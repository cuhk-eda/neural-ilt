import time, os
from collections import defaultdict
import numpy as np

import torch
import torchvision
from torchvision import transforms


def calc_loss_l2(pred, target, metrics):
    pred = torch.sigmoid(pred)

    criterion = torch.nn.MSELoss()
    loss = criterion(pred, target)
    metrics['loss'] += loss.item() * target.size(0)

    return loss

def print_metrics(epoch, metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("%s: %.4f" % (k, metrics[k] / epoch_samples))

    print("Epoch {}\taverage {} {}".format(epoch, phase, ", ".join(outputs)))

def train_cycle_model(model, alpha, beta, optimizer, scheduler, dataloaders, device, out_root, model_root, cplx_loss_layer, num_epochs):
    r"""
    Domain specific training recipe for Neural-ILT, section 3.5 (Jiang et al., ICCAD'20):
          Loss = supervised_loss_term + \alpha * ilt_loss_term + \beta * cplx_loss_term
    Args:
        model: Neural-ILT backbone model
        alpha: hyper-parameter for ilt_loss_term in the objective
        beta: hyper-parameter for cplx_loss_term in the objective
        dataloaders: dataset for train/test/validate
        num_epochs: number of training epochs
    """
    best_loss = 1e20
    sigmoid_layer = torch.nn.Sigmoid()
    for epoch in range(num_epochs):
        print('----- Epoch %d/%d -----' % (epoch, num_epochs - 1))

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("Learning Rate:", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            dataloader_len = len(dataloaders[phase])
            iter_since = time.time()
            loss_left_total = 0.0 
            loss_right_total = 0.0 
            loss_cplx_total = 0.0 
            for idx, data in enumerate(dataloaders[phase]):
                inputs, labels, _, new_cord, _, layouts = data # input = target wafer (layout), label = optimized mask
                inputs = inputs.to(device)
                labels = labels.to(device)
                layouts = layouts.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    loss_right, outputs, _ = model(inputs, layouts, new_cord)
                    loss_left = calc_loss_l2(outputs, labels, metrics)
                    sig_out = sigmoid_layer(outputs)
                    cplx_loss, _ = cplx_loss_layer(sig_out, layouts, new_cord)
                    
                    # Domain specific training recipe for Neural-ILT, section 3.5 (Jiang et al., ICCAD'20):
                    #       Loss = supervised_loss_term + \alpha * ilt_loss_term + \beta * cplx_loss_term
                    # where,  
                    #       supervised_loss_term = ||phi(z_t, w) - m||_2 
                    #       ilt_loss_term = ||litho(phi(z_t, w), P_nom) - z_t||_gamma
                    #       cplx_loss_term = ||litho(phi(z_t, w), P_max) - litho(phi(z_t, w), P_min)||_gamma
                    # and by default,    
                    #       \alpha = 1, \beta = 0 
                    loss = loss_left + loss_right.mul(alpha).div(inputs.size(2) ** 2).div(inputs.size(0)) + cplx_loss.mul(beta).div(inputs.size(2) ** 2).div(inputs.size(0))
                    
                    # For statistics
                    loss_left_total = loss_left_total + loss_left.item()
                    loss_right_total = loss_right_total + loss_right.item()
                    loss_cplx_total = loss_cplx_total + cplx_loss.item() / ((inputs.size(2) ** 2) * inputs.size(0))

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    if idx % 20 == 0:
                        print("%s time: %.2fs \tepoch: [%d/%d] \titer: [%d/%d] \tsl_loss: %.4f \tilt_loss:%.4f \tcplx_loss:%.4f" %
                            (phase, (time.time() - iter_since), epoch, (num_epochs -1) , idx, 
                            dataloader_len, loss_left_total / (idx + 1), loss_right_total / (idx + 1), loss_cplx_total / (idx + 1)))
                        iter_since = time.time()
                
                if idx == (dataloader_len - 2):
                    # Save last batch output images, assume batchsize = 2
                    with torch.set_grad_enabled(False):
                        preds = torch.sigmoid(outputs)
                        batch_preds = (preds > 0.5).type(torch.cuda.FloatTensor)
                        imagename = '%s_epoch_%d.png' % (phase, epoch)
                        imagePath = os.path.join(out_root, imagename)
                        torchvision.utils.save_image(batch_preds, imagePath, pad_value=1)
                        
                # For statistics
                epoch_samples += inputs.size(0)

            print_metrics(epoch, metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
                scheduler.step()

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                print("Best Val Loss: %.4f" % best_loss)
                best_epoch = epoch

        print("Saving Current Model...")
        torch.save(model.state_dict(), os.path.join(model_root, "cycle_%03d_wts.pth" % epoch))

        time_elapsed = time.time() - since
        print('Epoch %d\tTotal Time: %.0fm %.2fs\n' % (epoch, time_elapsed // 60, time_elapsed % 60))
        
    print('Best Val Loss: %.4f, Best Epoch: %d\n' % (best_loss, best_epoch))

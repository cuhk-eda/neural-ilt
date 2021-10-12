import torch
import torch.nn as nn
import utils.ilt_utils as ilt
from torch.autograd import Function
import lithosim.lithosim_cuda as litho
from utils.epe_checker import report_epe_violations, get_epe_checkpoints
import numpy as np
import os

class ilt_loss_function(Function):
    r"""
    The forward/backward implementation of conventional ILT functionality
    """
    @staticmethod
    def forward(ctx, mask_pred, target, kernels, kernels_ct, weight):
        r"""
        Calculate the ILT loss with respect to the forward prediction of the nn
        Args:
            mask_pred: the predicted mask of nn
            target: the target layout
            kernels/kernels_ct: SOCS kernels
            weight: weights for SOCS Kernels
        Return:
            Loss tensor
        """
        gamma = 4
        mask_pred = ilt.bit_mask_to_two_value_mask(mask_pred) # Change mask_pred from \in {0,1} to \in {-1,1}
        mask_pred_sig = ilt.sigmoid_ilt_mask(mask_pred, theta_m=4)
        result, _ = litho.lithosim(mask_pred_sig, 0.225, kernels, weight, None, False, return_binary_wafer=False)
        result = (result - target).pow(gamma).sum()

        ctx.save_for_backward(mask_pred_sig, target, kernels, kernels_ct, weight)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        r"""
        Compute the corresponding ILT gradient
        Return:
            Gradient tensor
        """
        mask_pred_sig, target, kernels, kernels_ct, weight = ctx.saved_tensors
        grad_input = ilt.compute_gradient(mask_pred_sig, target, kernels, 
                        kernels_ct, weight, dose=1.0, gamma=4.0, theta_z = 50, 
                        theta_m=4, avgpool_size=None)
        grad_input = grad_input * 2 # NOTE: backward for bit_mask_to_two_value_mask function
        return grad_output * grad_input, None, None, None, None

class ilt_loss_scale_function(Function):
    r"""
    A wrapper class of ilt_loss and cplx_loss calculations with scaling scheme
    Scaling scheme:
        Input mask (nn prediction) -> scale back to the cropped bbox size -> fit the cropped bbox into original size
                                   -> do litho-simulation -> compute ilt_loss/cplx_loss
                                   -> calculate the gradient of ilt_loss/cplx_loss -> crop and scale the gradient to input size -> gradient backward to nn
        [input_size, input_size] -> [cropped_bbox_size, cropped_bbox_size] -> [original_size, original_size] -> [cropped_bbox_size, cropped_bbox_size] -> [input_size, input_size]
    """
    @staticmethod
    def forward(ctx, mask_pred, target, kernels, kernels_ct, kernel_def, kernel_def_ct, weight, weight_def, new_cord, cycle_mode=False, cplx_obj=False, report_epe=False):
        lx, ly, rx, ry = new_cord
        output_litho_l2_loss = True

        # The exact ILT forward loss
        # ilt_loss = ||litho(threshold(Phi(Z_t, w)), P_nom) - Z_t||_gamma
        mask_pred_sig = ilt.bit_mask_to_two_value_mask(mask_pred) # change mask_pred \in {0,1} to \in {-1,1}
        mask_pred_sig = ilt.sigmoid_ilt_mask(mask_pred_sig, theta_m=4)
        mask_pred_sig_backup = torch.clone(mask_pred_sig)
        if output_litho_l2_loss: 
            # It is easier to monitor thru L2 loss, rather than the exact ILT loss
            mask_pred_sig = (mask_pred > 0.5).type(mask_pred.dtype)

        batch_size = mask_pred_sig.shape[0]
        channel_size = mask_pred_sig.shape[1]

        mask_pred_sig_orig_size = torch.zeros((batch_size, channel_size, 2048, 2048), 
                        dtype=mask_pred_sig.dtype, layout=mask_pred_sig.layout, device=mask_pred_sig.device)
        for i in range(batch_size):
            cur_mask = mask_pred_sig[i].unsqueeze(0) # 1 * 1 * H * W

            mask_crop = torch.nn.functional.interpolate(cur_mask, 
                            size=(abs(rx - lx)[i].item(), abs(rx - lx)[i].item()), 
                            mode='nearest') # 1 * 1 * H * W
            mask_origin = torch.zeros((mask_crop.shape[0], channel_size, 2048, 2048), 
                        dtype=mask_crop.dtype, layout=mask_crop.layout, device=mask_crop.device)
            mask_origin[..., ly[i].item():ry[i].item(), lx[i].item():rx[i].item()] = mask_crop
            mask_pred_sig_orig_size[i] = mask_origin.squeeze(0)

        avgpool_size = None
        mask_pred_bin_orig_size = (mask_pred_sig_orig_size > 0.5).type(torch.cuda.FloatTensor)

        result, bin_mask = litho.lithosim(mask_pred_bin_orig_size, 0.225, kernels, weight, 
                            None, False, avgpool_size=avgpool_size, return_binary_wafer=True)
        
        if cycle_mode:
            # It's easier to monitor the L2 loss using mean function but not the sum
            ilt_loss = (result - target).pow(4).mean()
            l2_loss = (bin_mask - target).abs().mean()
        elif cplx_obj:
            result_inner, bin_mask_inner = litho.lithosim(mask_pred_bin_orig_size, 0.225, kernel_def, weight_def, 
                                None, False, avgpool_size=avgpool_size, return_binary_wafer=True, dose=0.98)
            result_outer, bin_mask_outer = litho.lithosim(mask_pred_bin_orig_size, 0.225, kernels, weight, 
                                None, False, avgpool_size=avgpool_size, return_binary_wafer=True, dose=1.02)
            ilt_loss = ((result - target).pow(4) + (result_inner - target).pow(4) + (result_outer - target).pow(4)).sum().div(3.0)
            l2_loss = (bin_mask_outer - bin_mask_inner).abs().sum() # Image cplx loss
        else:
            ilt_loss = (result - target).pow(4).sum()
            l2_loss = (bin_mask - target).abs().sum()

        new_cord = torch.stack((new_cord[0], new_cord[1], new_cord[2], new_cord[3]), dim=0)
        cycle_mode = torch.tensor(cycle_mode)
        cplx_obj = torch.tensor(cplx_obj)
        ctx.save_for_backward(mask_pred_sig_backup, target, kernels, kernels_ct, kernel_def, kernel_def_ct, weight, weight_def, new_cord, cycle_mode, cplx_obj, torch.tensor(report_epe))

        if report_epe:
            checkpoints = get_epe_checkpoints((target.detach().data.cpu().numpy()[0][0] * 255).astype(np.uint8))
            epe_violation = report_epe_violations((bin_mask.detach().data.cpu().numpy()[0][0] * 255).astype(np.uint8), checkpoints)
            epe_violation = torch.tensor(epe_violation, requires_grad=False)
        place_holder = -1
        place_holder = torch.tensor(place_holder)

        if output_litho_l2_loss:
            # l2_loss is easier for us to monitor the training and on-nn-ilt correction, it is NOT the exact forward loss of ilt_loss_layer
            if report_epe:
                return l2_loss, epe_violation
            return l2_loss, place_holder
        else:
            # The exact ILT forward loss
            return ilt_loss, place_holder

    @staticmethod
    def backward(ctx, grad_output, place_holder):
        mask_pred_sig, target, kernels, kernels_ct, kernel_def, kernel_def_ct, weight, weight_def, new_cord, _, cplx_obj, _ = ctx.saved_tensors
        new_cord = [new_cord[0], new_cord[1], new_cord[2], new_cord[3]]
        
        grad_input = ilt.compute_gradient_scale(mask_pred_sig, target, kernels, kernels_ct, kernel_def, kernel_def_ct, weight, weight_def, new_cord, cplx_obj.item())
        grad_input = grad_input * 2 # NOTE: backward for bit_mask_to_two_value_mask function

        return grad_output * grad_input, None, None, None, None, None, None, None, None, None, None, None

class ilt_loss_layer(nn.Module):
    r"""
    The ILT loss layer of Neural-ILT
    """
    def __init__(self, kernels, kernels_ct, kernel_def, kernel_def_ct, weight, weight_def, cycle_mode=False, cplx_obj=False, report_epe=False):
        super(ilt_loss_layer, self).__init__()

        self.kernels = kernels
        self.kernels_ct = kernels_ct
        self.kernel_def = kernel_def
        self.kernel_def_ct = kernel_def_ct
        self.weight = weight
        self.weight_def = weight_def
        self.cycle_mode = cycle_mode
        self.cplx_obj = cplx_obj
        self.report_epe = report_epe
    
    def forward(self, preds, target, new_cord=None):
        if new_cord is None:
            return ilt_loss_function.apply(preds, target, self.kernels, self.kernels_ct, self.weight)
        else: 
            # Scale up original input from [N * C * 512 * 512] to [N * C * 2048 * 2048] and call ilt to get the gradient
            # Backward gradient size is [N * C * 512 * 512], which is the same as input
            return ilt_loss_scale_function.apply(preds, target, self.kernels, self.kernels_ct, self.kernel_def, self.kernel_def_ct, self.weight, self.weight_def,new_cord, self.cycle_mode, self.cplx_obj, self.report_epe)

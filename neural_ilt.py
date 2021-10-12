import os, time, sys, argparse
import numpy as np
from utils.utils import str2bool, dir_parser

import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader
from dataloader.refine_data_loader import ILTRefineDataset

import utils.unet_torch as unet_torch
from ilt_loss_layer import ilt_loss_layer
from neural_ilt_backbone import ILTNet

parser = argparse.ArgumentParser(description="take parameters")
parser.add_argument("--gpu_no", type=int, default=0)
parser.add_argument("--load_model_name", type=str, default="iccad_32nm_m1_wts.pth")
parser.add_argument("--beta", type=float, default=1.45)
parser.add_argument("--select_by_obj", type=str2bool, default=True)
args = parser.parse_args()


class Neural_ILT_Wrapper:
    r"""
    A wrapper class for Neural-ILT instance
    On-nerual-network ILT correction as a member function, and can be executed with customized lithography settings and datasets
    """
    def __init__(self, exp_para, image_para, lithosim_para):
        r"""
        Initialization of Neural_ILT_Wrapper
        Args:
            exp_para: experiment-relevant parameters
            image_para: image-relevant parameters
            lithosim_para: lithosim-relevant parameters        
        """

        # Set up the basic parameters
        print("Launching Neural-ILT on device:", exp_para["device"])
        self.exp_para = exp_para
        self.image_para = image_para
        self.lithosim_para = lithosim_para

        self.device = exp_para["device"]
        self.save_mask = exp_para["save_mask"]
        self.dynamic_beta = exp_para["dynamic_beta"]
        self.lr = exp_para["lr"]
        self.beta = exp_para["beta"]
        self.gamma = exp_para["gamma"]
        self.refine_iter_num = exp_para["refine_iter_num"]
        self.step_size = exp_para["step_size"]
        self.select_by_obj = exp_para["select_by_obj"]
        self.max_l2 = 1e15
        self.max_epe = 1e5
        if exp_para["max_l2"]:
            self.max_l2 = exp_para["max_l2"]
        if exp_para["max_epe"]:
            self.max_epe = exp_para["max_epe"]

        print("-------- Loading Neural-ILT Model & Data --------")

        # Litho-simulation kernels initialization
        print("MODEL:", self.exp_para["ilt_model_path"])
        if self.exp_para["data_set_name"]:
            print("DATASET:", self.exp_para["data_set_name"])
        self.kernels_root = self.lithosim_para["kernels_root"]
        self.kernels = torch.load(
            os.path.join(self.kernels_root, "kernel_focus_tensor.pt"),
            map_location=self.device,
        )
        self.kernels_ct = torch.load(
            os.path.join(self.kernels_root, "kernel_ct_focus_tensor.pt"),
            map_location=self.device,
        )
        self.kernels_def = torch.load(
            os.path.join(self.kernels_root, "kernel_defocus_tensor.pt"),
            map_location=self.device,
        )
        self.kernels_def_ct = torch.load(
            os.path.join(self.kernels_root, "kernel_ct_defocus_tensor.pt"),
            map_location=self.device,
        )
        self.weight = torch.load(
            os.path.join(self.kernels_root, "weight_focus_tensor.pt"),
            map_location=self.device,
        )
        self.weight_def = torch.load(
            os.path.join(self.kernels_root, "weight_defocus_tensor.pt"),
            map_location=self.device,
        )

        # Init the Unet & parse in the pretrained weights
        self.load_in_backone_model = unet_torch.UNet(n_class=1, in_channels=1).to(
            self.device
        )
        self.load_in_backone_model.load_state_dict(
            torch.load(self.exp_para["ilt_model_path"], map_location=self.device)
        )

        # Init the Neural-ILT backbone model
        self.refine_backbone_model = ILTNet(
            1,
            self.kernels,
            self.kernels_ct,
            self.kernels_def,
            self.kernels_def_ct,
            self.weight,
            self.weight_def,
            cplx_obj=False,
            report_epe=True,
            in_channels=1,
        ).to(self.device)

        # Init the complexity refinement layer
        self.cplx_loss_layer = ilt_loss_layer(
            self.kernels,
            self.kernels_ct,
            self.kernels_def,
            self.kernels_def_ct,
            self.weight,
            self.weight_def,
            cplx_obj=True,
        ).to(self.device)

        # Parse in the pretrained UNet weights into Neural-ILT
        pretrain_dict = self.load_in_backone_model.state_dict()
        self.model_dict = self.refine_backbone_model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in self.model_dict}

        for param in self.refine_backbone_model.parameters():
            param.requires_grad = True

        self.model_dict.update(pretrain_dict)
        self.refine_backbone_model.load_state_dict(self.model_dict)

        # Set up optimizer
        self.optimizer_ft = optim.Adam(
            self.refine_backbone_model.parameters(), lr=self.lr
        )
        self.opt_init_state = self.optimizer_ft.state_dict()


    def neural_ilt_correction(self, dataloaders):
        sys.setrecursionlimit(10000)
        start_time = time.time()
        online_train_loss_list = {}
        for idx, data in enumerate(dataloaders): # For each input target layout, conduct the on-neural-network ILT correction
            best_counter = 0
            inputs, labels, _, new_cord, layout_name = data
            print("\n--- Initializing Model for %s ---" % layout_name[0])
            self.refine_backbone_model.load_state_dict(self.model_dict)
            self.refine_backbone_model.train()
            self.optimizer_ft.load_state_dict(self.opt_init_state)
            step_lr_scheduler = lr_scheduler.StepLR(
                self.optimizer_ft, step_size=self.step_size, gamma=self.gamma
            )

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_size = inputs.size(0)
            assert batch_size == 1  # Online learning only for one specific layout

            x1_, y1_, x2_, y2_ = new_cord
            x1 = x1_[0].item()
            y1 = y1_[0].item()
            x2 = x2_[0].item()
            y2 = y2_[0].item()
            layout_name = layout_name[0].split(".")[0]

            best_epe_vios = 1e5
            best_loss = 1e15
            best_l2_loss = 1e15
            best_iter = 0
            best_masks = None
            my_beta = self.beta

            cur_image_start_time = time.time()
            iter_since = time.time()
            for iteration in range(self.refine_iter_num): # Maximum on-neural-network ILT correction iterations
                self.optimizer_ft.zero_grad()
                with torch.set_grad_enabled(True):
                    # Forward inference of Neural-ILT, calculate the corresponding losses
                    l2_loss, masks, epe_violation = self.refine_backbone_model(
                        inputs, labels, new_cord
                    )
                    sig_masks = torch.sigmoid(masks)
                    cplx_loss, _ = self.cplx_loss_layer(sig_masks, labels, new_cord)

                    # The on-neural-network ILT correction (backward of gradient)
                    loss = l2_loss.div(inputs.size(0)) + cplx_loss.mul(my_beta).div(
                        inputs.size(0)
                    )
                    loss.backward()
                    self.optimizer_ft.step()

                    cur_epe_vios = epe_violation.item()
                    cur_loss = my_beta * cplx_loss.item() + l2_loss.item() # select best solution by objective score = alpha * l2_loss + beta * cplx_loss
                    if not self.select_by_obj: # select best solution by printability score = l2_loss + cplx_loss
                        cur_loss = cplx_loss.item() + l2_loss.item()
                    update_best = cur_epe_vios <= best_epe_vios and cur_loss <  best_loss # consider EPE violation concurrently
                    if update_best and l2_loss.item() < self.max_l2 and cur_epe_vios < self.max_epe:
                        best_loss = cur_loss
                        best_l2_loss = l2_loss.item()
                        best_epe_vios = cur_epe_vios
                        best_cplx_loss = cplx_loss.item()
                        best_iter = iteration
                        best_masks = masks.detach()
                        best_counter = 0
                    else:
                        best_counter += 1
                    if best_counter > 20 and iteration > 25: # Early break if needed
                        break
                    if iteration % 2 == 0:
                        print(
                            "time: %.2fs\tImage_num: [%d/%d]\titer: [%d/%d]\tobj_loss: %.2f\tl2_loss: %.2f\tcplx_loss: %.2f\tepe_vio: %d"
                            % (
                                (time.time() - iter_since),
                                (idx + 1),
                                len(dataloaders),
                                iteration,
                                (self.refine_iter_num - 1),
                                loss.item(),
                                l2_loss.item(),
                                cplx_loss.item(),
                                cur_epe_vios
                            )
                        )
                        iter_since = time.time()
                step_lr_scheduler.step()

            print("Early stop counter:", best_counter)

            cur_runtime = time.time() - cur_image_start_time
            online_train_loss_list[layout_name + "_total_loss"] = [
                best_loss,
                best_l2_loss,
                best_epe_vios,
                best_cplx_loss,
                cur_runtime,
            ]

            best_image_path = None
            if self.save_mask:
                best_pred = torch.sigmoid(best_masks)
                cur_mask = (best_pred > 0.5).type(
                    torch.cuda.FloatTensor
                )  # 1 * 1 * H * W
                mask_crop = torch.nn.functional.interpolate(
                    cur_mask, size=(abs(y2 - y1), abs(x2 - x1)), mode="nearest"
                )  # 1 * 1 * H * W
                mask_origin = torch.zeros(
                    (mask_crop.shape[0], cur_mask.shape[0], 2048, 2048),
                    dtype=mask_crop.dtype,
                    layout=mask_crop.layout,
                    device=mask_crop.device,
                )
                mask_origin[..., y1:y2, x1:x2] = mask_crop
                best_image_path = os.path.join(
                    "output/refine_net_output", "%s_res.png" % (layout_name)
                )
                print("Saving best mask in %s" % best_image_path)
                torchvision.utils.save_image(mask_origin, best_image_path)
            print(
                "ImageName: %s\tTime: %.2fs\tbest_iter: %.2f\tbest_loss-> total:%.2f,\tl2:%.2f,\tcplx:%.2f,\tEPEV:%d"
                % (
                    layout_name,
                    cur_runtime,
                    best_iter,
                    best_loss,
                    best_l2_loss,
                    best_cplx_loss,
                    best_epe_vios
                )
            )

        print("\nTotal Time: %.4fs\n" % (time.time() - start_time))
        for key in online_train_loss_list:
            print(
                "%s: total:%d\tl2:%d\tepev:%d\tcplx:%d\truntime:%.4f"
                % (
                    key,
                    online_train_loss_list[key][0],
                    online_train_loss_list[key][1],
                    online_train_loss_list[key][2],
                    online_train_loss_list[key][3],
                    online_train_loss_list[key][4],
                )
            )
        l2_list = []
        epe_list = []
        pv_list = []
        runtime_list = []
        for key in sorted(online_train_loss_list):
            l2_list.append(online_train_loss_list[key][1])
            epe_list.append(online_train_loss_list[key][2])
            pv_list.append(online_train_loss_list[key][3])
            runtime_list.append(online_train_loss_list[key][4])
        l2_avg = np.array(l2_list).mean()
        epe_avg = np.array(epe_list).mean()
        pv_avg = np.array(pv_list).mean()
        runtime_avg = np.array(runtime_list).mean()
        print(
            "Average L2 Loss: %.4f\tPVBand: %.4f\tEPEV: %.4f\tRun Time:%.4f"
            % (l2_avg, pv_avg, epe_avg, runtime_avg)
        )

        return l2_avg, pv_avg, epe_avg, runtime_avg


def run_neural_ilt_ibm_bench():
    exp_para = {
        "device": "cuda:%s" % args.gpu_no if torch.cuda.is_available() else "cpu",
        "phase": "test",
        "beta": args.beta, # hyper-parameter for cplx_loss
        "lr": 2e-3,
        "gamma": 0.1,
        "refine_iter_num": 60,
        "step_size": 35,
        "max_l2": 95000,
        "max_epe": 55,
        "save_mask": True,
        "dynamic_beta": False,
        # "ilt_model_path": os.path.join("models/unet/", "iccad_32nm_m1_wts.pth"),
        "ilt_model_path": os.path.join("models/unet/", args.load_model_name),
        "data_set_name": "ICCAD2013-IBM-Benchmark",
        "select_by_obj": args.select_by_obj,
    }

    image_para = {
        "original_size": 2048,
        "scale_size": 512,
        "bbox_margin": 256,
    }

    lithosim_para = {
        "kernels_root": "lithosim/lithosim_kernels/torch_tensor",
        "kernel_num": 24,
    }

    # Obtain data_loader from a list of masks & obtain the corresponding bboxes on-the-fly
    nerual_ilt = Neural_ILT_Wrapper(exp_para, image_para, lithosim_para)
    refine_dataset = ILTRefineDataset(
        data_root=dir_parser("./", "dataset"),
        split="ibm_opc_test",
        margin=image_para["bbox_margin"],
        scale_dim_w=image_para["scale_size"],
        scale_dim_h=image_para["scale_size"],
        read_ref=False,
    )
    refine_data_loader = DataLoader(
        dataset=refine_dataset, num_workers=0, batch_size=1, shuffle=False
    )

    # Conduct on-neural-network ILT correction for the ICCAD-2013 IBM contest dataset
    l2_avg, pv_avg, epe_avg, runtime_avg = nerual_ilt.neural_ilt_correction(refine_data_loader)

    # Report results, baselines quoted from GAN-OPC (Yang et al., TCAD'20)
    mosaic_avg = [44012.7, 50899.5, 788.5]
    ganopc_avg = [40094.6, 50568.1, 384.7]
    pganopc_avg = [39948.9, 49957.2, 371.3]
    eganopc_avg = [39500.8, 48917.8, 262]
    print("Ratio_to_mosaic:\tL2:%.2f%%\tPVBand:%.2f%%\tPrintability(L2+PVB):%.2f%%\tRuntime:%.2f%%" % 
    (l2_avg / mosaic_avg[0] * 100, pv_avg / mosaic_avg[1] * 100, (l2_avg+pv_avg)/(mosaic_avg[0]+mosaic_avg[1]) * 100, runtime_avg/mosaic_avg[2] * 100))
    print("Ratio_to_ganopc:\tL2:%.2f%%\tPVBand:%.2f%%\tPrintability(L2+PVB):%.2f%%\tRuntime:%.2f%%" % 
    (l2_avg / ganopc_avg[0] * 100, pv_avg / ganopc_avg[1] * 100, (l2_avg+pv_avg)/(ganopc_avg[0]+ganopc_avg[1]) * 100, runtime_avg/ganopc_avg[2] * 100))
    print("Ratio_to_pganopc:\tL2:%.2f%%\tPVBand:%.2f%%\tPrintability(L2+PVB):%.2f%%\tRuntime:%.2f%%" % 
    (l2_avg / pganopc_avg[0] * 100, pv_avg / pganopc_avg[1] * 100, (l2_avg+pv_avg)/(pganopc_avg[0]+pganopc_avg[1]) * 100, runtime_avg/pganopc_avg[2] * 100))
    print("Ratio_to_eganopc:\tL2:%.2f%%\tPVBand:%.2f%%\tPrintability(L2+PVB):%.2f%%\tRuntime:%.2f%%" % 
    (l2_avg / eganopc_avg[0] * 100, pv_avg / eganopc_avg[1] * 100, (l2_avg+pv_avg)/(eganopc_avg[0]+eganopc_avg[1]) * 100, runtime_avg/eganopc_avg[2] * 100))


def run_neural_ilt_ibm_ext_bench():
    exp_para = {
        "device": "cuda:%s" % args.gpu_no if torch.cuda.is_available() else "cpu",
        "phase": "test",
        "beta": args.beta,
        "lr": 2e-3,
        "gamma": 0.1,
        "refine_iter_num": 60,
        "step_size": 35,
        "max_l2": 150000,
        "max_epe": 75,
        "save_mask": True,
        "dynamic_beta": False,
        "ilt_model_path": os.path.join("models/unet/", args.load_model_name),
        "data_set_name": "ICCAD2013-IBM-ext-Benchmark",
        "select_by_obj": args.select_by_obj,
    }

    image_para = {
        "original_size": 2048,
        "scale_size": 512,
        "bbox_margin": 256,
    }

    lithosim_para = {
        "kernels_root": "lithosim/lithosim_kernels/torch_tensor",
        "kernel_num": 24,
    }

    # Obtain data_loader from a list of masks & obtain the corresponding bboxes on-the-fly
    nerual_ilt = Neural_ILT_Wrapper(exp_para, image_para, lithosim_para)
    refine_dataset = ILTRefineDataset(
        data_root=dir_parser("./", "dataset"),
        split="ibm_opc_test_ext",
        margin=image_para["bbox_margin"],
        scale_dim_w=image_para["scale_size"],
        scale_dim_h=image_para["scale_size"],
        read_ref=False,
    )
    refine_data_loader = DataLoader(
        dataset=refine_dataset, num_workers=0, batch_size=1, shuffle=False
    )

    # Conduct on-neural-network ILT correction for the ICCAD-2013 IBM ext dataset
    l2_avg, pv_avg, epe_avg, runtime_avg = nerual_ilt.neural_ilt_correction(refine_data_loader)

    # Report results, baselines quoted from Neural-ILT 2.0 (Jiang et al., in submission to TCAD)
    mosaic_avg = [90486.3, 109842.7, 455]
    ganopc_avg = [89556.5, 120882.2, 364]
    pganopc_avg = [86697.4, 110330.5, 364]
    eganopc_avg = [86105.7, 108690.7, 273]
    print("Ratio_to_mosaic:\tL2:%.2f%%\tPVBand:%.2f%%\tPrintability(L2+PVB):%.2f%%\tRuntime:%.2f%%" % 
    (l2_avg / mosaic_avg[0] * 100, pv_avg / mosaic_avg[1] * 100, (l2_avg+pv_avg)/(mosaic_avg[0]+mosaic_avg[1]) * 100, runtime_avg/mosaic_avg[2] * 100))
    print("Ratio_to_ganopc:\tL2:%.2f%%\tPVBand:%.2f%%\tPrintability(L2+PVB):%.2f%%\tRuntime:%.2f%%" % 
    (l2_avg / ganopc_avg[0] * 100, pv_avg / ganopc_avg[1] * 100, (l2_avg+pv_avg)/(ganopc_avg[0]+ganopc_avg[1]) * 100, runtime_avg/ganopc_avg[2] * 100))
    print("Ratio_to_pganopc:\tL2:%.2f%%\tPVBand:%.2f%%\tPrintability(L2+PVB):%.2f%%\tRuntime:%.2f%%" % 
    (l2_avg / pganopc_avg[0] * 100, pv_avg / pganopc_avg[1] * 100, (l2_avg+pv_avg)/(pganopc_avg[0]+pganopc_avg[1]) * 100, runtime_avg/pganopc_avg[2] * 100))
    print("Ratio_to_eganopc:\tL2:%.2f%%\tPVBand:%.2f%%\tPrintability(L2+PVB):%.2f%%\tRuntime:%.2f%%" % 
    (l2_avg / eganopc_avg[0] * 100, pv_avg / eganopc_avg[1] * 100, (l2_avg+pv_avg)/(eganopc_avg[0]+eganopc_avg[1]) * 100, runtime_avg/eganopc_avg[2] * 100))


if __name__ == "__main__":
    print(args)
    run_neural_ilt_ibm_bench()
    # run_neural_ilt_ibm_ext_bench()
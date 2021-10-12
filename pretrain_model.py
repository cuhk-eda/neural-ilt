import os, time, argparse
from utils.utils import str2bool, dir_parser

import torch
torch.manual_seed(1)

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader
from dataloader.train_data_loader import OPCDataset
import utils.train_utils as train_utils
import neural_ilt_backbone

# Arguments
parser = argparse.ArgumentParser(description="take parameters")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--gpu_no", type=int, default=0)
parser.add_argument("--num_worker", type=int, default=0)
parser.add_argument("--num_epoch", type=int, default=10)
parser.add_argument("--root", type=str, default=os.getcwd())
parser.add_argument("--data_dir", type=str, default="dataset")
parser.add_argument("--train_out_dir", type=str, default="output/train")
parser.add_argument("--test_out_dir", type=str, default="output/test")
parser.add_argument(
    "--torch_data_path", type=str, default="lithosim/lithosim_kernels/torch_tensor"
)
parser.add_argument("--model_dir", type=str, default="models/unet")
parser.add_argument("--train_mode", type=str2bool, default=True)
parser.add_argument("--alpha", type=float, default=1, help="cycle loss weight for l2")
parser.add_argument("--beta", type=float, default=0, help="cycle loss weight for cplx")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--gamma", type=float, default=0.1, help="lr decay rate")
parser.add_argument("--step_size", type=int, default=5, help="lr decay step size")
parser.add_argument(
    "--scale_size",
    type=int,
    default=512,
    help="The target scaling size of the crop bbox, i.e., [corp_bbox_size, corp_bbox_size] -> [scale_size, scale_size]",
)
parser.add_argument(
    "--margin",
    type=int,
    default=256,
    help="The margin of the crop bbox, i.e., corp_bbox_size = max(margin + bbox_size_w, margin + bbox_size_h)",
)
parser.add_argument(
    "--read_ref",
    type=str2bool,
    default=False,
    help="Read the pre-computed crop bbox for each layout from csv file",
)

args = parser.parse_args()

for arg in vars(args):
    print("%s: %s" % (arg, getattr(args, arg)))

data_root = dir_parser(args.root, args.data_dir)
train_out_root = dir_parser(args.root, args.train_out_dir)
test_out_root = dir_parser(args.root, args.test_out_dir)
model_root = dir_parser(args.root, args.model_dir)

device = torch.device("cuda:%s" % args.gpu_no if torch.cuda.is_available() else "cpu")

print("data_root: %s" % data_root)
print("train_out_root: %s" % train_out_root)
print("test_out_root: %s" % test_out_root)
print("model_root: %s" % model_root)
print("device: %s" % device)

# Load in the datasets for training
train_dataset = OPCDataset(
    data_root=data_root,
    split="train",
    margin=args.margin,
    scale_dim_w=args.scale_size,
    scale_dim_h=args.scale_size,
    read_ref=args.read_ref,
)
test_dataset = OPCDataset(
    data_root=data_root,
    split="test",
    margin=args.margin,
    scale_dim_w=args.scale_size,
    scale_dim_h=args.scale_size,
    read_ref=args.read_ref,
)
val_dataset = OPCDataset(
    data_root=data_root,
    split="val",
    margin=args.margin,
    scale_dim_w=args.scale_size,
    scale_dim_h=args.scale_size,
    read_ref=args.read_ref,
)

print("Number of train set: %d " % len(train_dataset))
print("Number of test set: %d " % len(test_dataset))
print("Number of val set: %d " % len(val_dataset))


train_data_loader = DataLoader(
    dataset=train_dataset,
    num_workers=args.num_worker,
    batch_size=args.batch_size,
    shuffle=False,
)
test_data_loader = DataLoader(
    dataset=test_dataset, num_workers=args.num_worker, batch_size=1, shuffle=False
)
val_data_loader = DataLoader(
    dataset=val_dataset,
    num_workers=args.num_worker,
    batch_size=args.batch_size,
    shuffle=False,
)

dataloaders = {
    "train": train_data_loader,
    "test": test_data_loader,
    "val": val_data_loader,
}


if __name__ == "__main__":
    r"""
    Domain specific training recipe for Neural-ILT, section 3.5 (Jiang et al., ICCAD'20):
          Loss = supervised_loss_term + \alpha * ilt_loss_term + \beta * cplx_loss_term
    where,  
          supervised_loss_term = ||phi(z_t, w) - m||_2 
          ilt_loss_term = ||litho(phi(z_t, w), P_nom) - z_t||_gamma
          cplx_loss_term = ||litho(phi(z_t, w), P_max) - litho(phi(z_t, w), P_min)||_gamma
    By default,    
          \alpha = 1, \beta = 0 
    """
    if args.train_mode:
        # Load in the lithography kernels
        kernels_path = os.path.join(args.torch_data_path, "kernel_focus_tensor.pt")
        kernels_ct_path = os.path.join(
            args.torch_data_path, "kernel_ct_focus_tensor.pt"
        )
        kernels_def_path = os.path.join(
            args.torch_data_path, "kernel_defocus_tensor.pt"
        )
        kernels_def_ct_path = os.path.join(
            args.torch_data_path, "kernel_ct_defocus_tensor.pt"
        )
        weight_path = os.path.join(args.torch_data_path, "weight_focus_tensor.pt")
        weight_def_path = os.path.join(args.torch_data_path, "weight_defocus_tensor.pt")

        kernels = torch.load(kernels_path, map_location=device)
        kernels_ct = torch.load(kernels_ct_path, map_location=device)
        kernels_def = torch.load(kernels_def_path, map_location=device)
        kernels_def_ct = torch.load(kernels_def_ct_path, map_location=device)
        weight = torch.load(weight_path, map_location=device)
        weight_def = torch.load(weight_def_path, map_location=device)

        # Init the neural-ilt train model and optimizer/scheduler
        train_model = neural_ilt_backbone.ILTNet(
            1,
            kernels,
            kernels_ct,
            kernels_def,
            kernels_def_ct,
            weight,
            weight_def,
            cycle_mode=True,
            in_channels=1,
        ).to(device)
        from ilt_loss_layer import ilt_loss_layer
        
        cplx_loss_layer = ilt_loss_layer(
            kernels,
            kernels_ct,
            kernels_def,
            kernels_def_ct,
            weight,
            weight_def,
            cplx_obj=True,
        ).to(device)
        optimizer_ft = optim.Adam(train_model.parameters(), lr=args.lr)

        step_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=args.step_size, gamma=args.gamma
        )

        # Domain specific pre-train of Neural-ILT
        # See in Neural-ILT section 3.5 (Jiang et al., ICCAD'20)
        print("\n--------Start Training--------\n")
        start = time.time()
        train_utils.train_cycle_model(
            train_model,
            args.alpha,
            args.beta,
            optimizer_ft,
            step_lr_scheduler,
            dataloaders,
            device,
            train_out_root,
            model_root,
            cplx_loss_layer,
            num_epochs=args.num_epoch,
        )
        print("Finish training. Total training time: %.4f" % (time.time() - start))

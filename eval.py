import lithosim.lithosim_cuda as litho
import os, argparse
import torch
import torchvision

parser = argparse.ArgumentParser(description='take parameters')
parser.add_argument('--kernels_root', type=str,
                    default='lithosim/lithosim_kernels/bin_data')
parser.add_argument('--mask_root', type=str, default='output/refine_net_output/')
parser.add_argument('--mask_file_name', type=str, default=None)
parser.add_argument('--layout_root', type=str, default='dataset/ibm_opc_test/')
parser.add_argument('--layout_file_name', type=str, default=None)
parser.add_argument('--output_root', type=str,
                    default='output/refine_litho_out')
parser.add_argument('--kernel_num', type=int, default=24, help='24 SOCS kernels')
parser.add_argument('--gpu_no', type=int, default=0, help='GPU device id')
args = parser.parse_args()


def eval():
    r"""
    Evaluate the printability of the input mask
    Return:
        L2 error
        PVBand
    """

    print("------ Loading Kernels Data ------")
    kernel_torch_data_path = 'lithosim/lithosim_kernels/torch_tensor'

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    device = torch.device('cuda:' + str(args.gpu_no)
                          if torch.cuda.is_available() else 'cpu')
    kernels_path = os.path.join(
        kernel_torch_data_path, 'kernel_focus_tensor.pt')
    weight_path = os.path.join(
        kernel_torch_data_path, 'weight_focus_tensor.pt')
    kernels_def_path = os.path.join(
        kernel_torch_data_path, 'kernel_defocus_tensor.pt')
    weight_def_path = os.path.join(
        kernel_torch_data_path, 'weight_defocus_tensor.pt')
    kernels = torch.load(kernels_path, map_location=device)
    weight = torch.load(weight_path, map_location=device)
    kernels_def = torch.load(kernels_def_path, map_location=device)
    weight_def = torch.load(weight_def_path, map_location=device)
    threshold = 0.225

    if args.mask_file_name is not None and args.layout_file_name is not None:
        save_name = os.path.join(args.output_root, args.mask_file_name)
        mask_data = litho.load_image(os.path.join(args.mask_root, args.mask_file_name)).to(device)
        layout_data = litho.load_image(os.path.join(args.layout_root, args.layout_file_name)).to(device)

        print("------ Start lithography simulation for %s ------" % args.mask_file_name)
        _, wafer_nom = litho.lithosim(mask_data, threshold, kernels, weight, save_name, save_bin_wafer_image=False, kernels_number=args.kernel_num, dose=1.0)
        _, wafer_min = litho.lithosim(mask_data, threshold, kernels_def, weight_def, save_name, save_bin_wafer_image=False, kernels_number=args.kernel_num, dose=0.98)
        _, wafer_max = litho.lithosim(mask_data, threshold, kernels, weight, save_name, save_bin_wafer_image=False, kernels_number=args.kernel_num, dose=1.02)

        L2_error = (wafer_nom - layout_data).abs().sum()
        PVB = (wafer_min - wafer_max).abs().sum()
        
        torchvision.utils.save_image(wafer_nom, save_name)
        print("%s: L2 = %d, PVB = %d" % (args.mask_file_name, L2_error.item(), PVB.item()))
    else:
        raise NotImplementedError("Please specify correct {mask_file_name} and {layout_file_name}.")
    print("------ Done! ------")
    return L2_error.item(), PVB.item()

if __name__ == "__main__":
    eval()

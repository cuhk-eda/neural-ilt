import lithosim_cuda as litho
import os, argparse
from fnmatch import fnmatch
import torch

parser = argparse.ArgumentParser(description='take parameters')
parser.add_argument('--kernels_root', type=str,
                    default='lithosim_kernels/bin_data')
parser.add_argument('--kernel_type', type=str,
                    default='focus', help='[focus] or [defocus]')
parser.add_argument('--input_root', type=str, default='../output/refine_net_output/')
parser.add_argument('--output_root', type=str,
                    default='../output/refine_litho_out')
parser.add_argument('--kernel_num', type=int, default=24, help='24 SOCS kernels')
parser.add_argument('--device_id', type=int, default=0, help='GPU device id')
args = parser.parse_args()


def run_litho_sim_batch():
    r"""
    Run lithography simulation for a batch of masks (within the arg.input_root folder)
    """

    kernels_root = args.kernels_root
    print("------ Start Preprocessing Kernels Data ------")
    _, _, _ = litho.kernel_bin_preprocess(kernels_root, 'focus')
    _, _, _ = litho.kernel_bin_preprocess(kernels_root, 'defocus')
    print("------ Finish Preprocessing Kernels Data ------")

    litho_kernel_type = args.kernel_type
    torch_data_path = 'lithosim_kernels/torch_tensor'
    image_root = args.input_root
    output_root = args.output_root

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    threshold = 0.225
    kernels_path = os.path.join(
        torch_data_path, 'kernel_' + litho_kernel_type + '_tensor.pt')
    weight_path = os.path.join(
        torch_data_path, 'weight_' + litho_kernel_type + '_tensor.pt')

    device = torch.device('cuda:' + str(args.device_id)
                          if torch.cuda.is_available() else 'cpu')
    save_bin_wafer_image = True
    kernel_number = args.kernel_num

    kernels = torch.load(kernels_path, map_location=device)
    weight = torch.load(weight_path, map_location=device)

    # tensor image data
    for path, subdirs, files in os.walk(image_root):
        for name in files:
            if fnmatch(name, '*.png'):
                png_file = os.path.join(path, name)
                save_name = os.path.join(output_root, name)
                image_data = litho.load_image(png_file)
                image_data = image_data.to(device)
                print("------ Start lithography simulation for %s ------" % name)
                _, _ = litho.lithosim(image_data, threshold, kernels, weight, save_name, save_bin_wafer_image, kernel_number)
                print("------ Finish! ------\n")
    print("------ Done! ------")

if __name__ == "__main__":
    run_litho_sim_batch()
import lithosim.lithosim_cuda as litho
from utils.epe_checker import get_epe_checkpoints
import torch, torchvision
import numpy as np
import os 
from PIL import Image


def sigmoid_ilt_z(intensity_map, theta_z=50, threhold_z=0.225):
    sigmoid_layer = torch.nn.Sigmoid()
    intensity_map = (intensity_map - threhold_z) * theta_z
    return sigmoid_layer(intensity_map)

def sigmoid_ilt_mask(mask, theta_m=4, threhold_m=0.0):
    sigmoid_layer = torch.nn.Sigmoid()
    mask = (mask - threhold_m) * theta_m
    return sigmoid_layer(mask)

def compute_common_term(mask, target, kernels, weight, dose=1.0, gamma=4.0, theta_z=50, theta_m=4):
    r"""
    Compute the common term for graident calculation
    """
    intensity_map, _ = litho.lithosim(mask, None, kernels, weight, 
                    wafer_output_path=None, save_bin_wafer_image=False, 
                    kernels_number=None, avgpool_size=None,
                    dose=dose, return_binary_wafer=False)

    z_nom = sigmoid_ilt_z(intensity_map, theta_z) # Nomial wafer image with continuous value between (0,1)
    z_t = target # Binary target image
    common_term = (z_nom - z_t).pow(gamma-1) * (1 - z_nom) * z_nom # [1 * H * W]
    return common_term

def compute_gradient(mask, target, kernels, kernels_ct, weight, 
                    dose=1.0, gamma=4.0, theta_z=50, theta_m=4, epe_offset = 15, avgpool_size=None):
    r"""
    Main function of ILT loss gradient calculation
    Args:
        mask: input mask image
        target: target layout
        kernels (kernels_ct): SOCS kernels
        weight: weights for SOCS kernels
        dose: +-2% process condition \in {0.98, 1.0, 1.02} -> min/nomial/max
        gamma, theta_z, theta_m: hyper-parameters for gradient calculation, same as MOSAIC (Gao et al., DAC'14)
    Return:
        Gradient tensor of ilt_loss
    """
    common_term = compute_common_term(mask, target, kernels, weight, dose, gamma, theta_z, theta_m)

    checkpoints = get_epe_checkpoints((target.detach().data.cpu().numpy()[0][0] * 255).astype(np.uint8))
    mask_epe_roi = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]),
                               dtype=mask.dtype, layout=mask.layout, device=mask.device)
    # mask_epe_roi = target                       
    for cp in checkpoints['h_pts']:
        mask_epe_roi[:, :, (cp[1]-epe_offset):(cp[1]+epe_offset), cp[0]] = 1
    for cp in checkpoints['v_pts']:
        mask_epe_roi[:, :, cp[1], (cp[0]-epe_offset):(cp[0]+epe_offset)] = 1
    
    mask_convolve_kernel_ct_output = litho.convolve_kernel(mask, kernels_ct, weight, dose) #[1 * H * W]
    mask_convolve_kernel_ct_output = mask_convolve_kernel_ct_output * common_term # real_part = real_part * common_term, imagine_part = imagine_part * common_term

    # Here we need to flip the kernels and kernels_ct => rotate H and H* by 180 degrees
    kernels_flip = torch.flip(kernels, [1,2])
    kernels_ct_flip = torch.flip(kernels_ct, [1,2])
    gradient_right_term = litho.convolve_kernel(mask_convolve_kernel_ct_output, kernels_flip, weight, dose).real #[1 * H * W], take the real part
    gradient_right_term_epe = gradient_right_term * mask_epe_roi  # [1 * H * W], take the real part

    mask_convolve_kernel_output = litho.convolve_kernel(mask, kernels, weight, dose) #[1 * H * W]
    mask_convolve_kernel_output = mask_convolve_kernel_output * common_term # real_part = real_part * common_term, imagine_part = imagine_part * common_term
    gradient_left_term = litho.convolve_kernel(mask_convolve_kernel_output, kernels_ct_flip, weight, dose).real #[1 * H * W], take the real part
    gradient_left_term_epe = gradient_left_term * mask_epe_roi  # [1 * H * W], take the real part

    sigma = 2
    constant = gamma * theta_z * theta_m
    constant_epe = gamma * theta_z * theta_m * sigma
    discrete_penalty_mask = 0.025 * (-8 * mask + 4) # From the MOSAIC's (GAO et al. DAC'14) source code
    discrete_penalty_mask_epe = 0.025 * (-8 * mask * mask_epe_roi + 4) * sigma
    gradient = (constant * (gradient_right_term + gradient_left_term) + theta_m * discrete_penalty_mask) * mask * (1 - mask)
    gradient_epe = (constant_epe * (gradient_right_term_epe + gradient_left_term_epe) +
                    theta_m * discrete_penalty_mask_epe) * mask * (1 - mask)
    gradient += gradient_epe

    if avgpool_size is not None:
        avg_layer = torch.nn.AvgPool2d(
            kernel_size=(avgpool_size, avgpool_size), 
            stride=(avgpool_size, avgpool_size))
        gradient = avg_layer(gradient)

    return gradient

def compute_convolve_sigmoid_gradient(mask, kernels, kernels_ct, weight, dose=1.0, theta_z=50, theta_m=4):
    r"""
    Calculate the gradient of the sig(convolve(:,:)) operation
    e.g., Z = Sig(I) = Sig(convolve(mask, kernels))
    Args:
        mask: input mask image
        kernels/kernels_ct: SOCS kernels (focus)
        weight/weight_def: weights for SOCS kernels (focus/defocus)
        dose: +-2% process condition \in {0.98, 1.0, 1.02} -> min/nomial/max
        theta_z, theta_m: hyper-parameters for gradient calculation
    Return:
        Gradient tensor of sig(convolve(:,:))
    """
    intensity_map, _ = litho.lithosim(mask, None, kernels, weight, 
                    wafer_output_path=None, save_bin_wafer_image=False, 
                    kernels_number=None, avgpool_size=None,
                    dose=dose, return_binary_wafer=False) # Calculate convolve(mask, kernels)
    z = sigmoid_ilt_z(intensity_map, theta_z) # Nomial/min/max wafer image with continuous value between (0,1)
    common_term = (1 - z) * z

    # Flip your kernels here
    kernels_flip = torch.flip(kernels, [1,2])
    kernels_ct_flip = torch.flip(kernels_ct, [1,2])

    mask_convolve_kernel_ct_output = litho.convolve_kernel(mask, kernels_ct, weight, dose) # convolve(mask, kernels_ct)
    mask_convolve_kernel_ct_output = mask_convolve_kernel_ct_output * common_term # convolve(mask, kernels_ct) * z * (1 - z)
    gradient_right_term = litho.convolve_kernel(mask_convolve_kernel_ct_output, kernels_flip, weight, dose).real # convolve((convolve(mask, kernels_ct) * z * (1 - z)), kernels), [1 * H * W], take the real part
    mask_convolve_kernel_output = litho.convolve_kernel(mask, kernels, weight, dose) # convolve(mask, kernels)
    mask_convolve_kernel_output = mask_convolve_kernel_output * common_term # convolve(mask, kernels) * z * (1 - z)
    gradient_left_term = litho.convolve_kernel(mask_convolve_kernel_output, kernels_ct_flip, weight, dose).real # convolve((convolve(mask, kernels) * z * (1 - z)), kernels_ct), [1 * H * W], take the real part
    discrete_penalty_mask = 0.025 * (-8 * mask + 4) # From the MOSAIC's (GAO et al. DAC'14) source code
    gradient = (theta_z * theta_m * (gradient_right_term + gradient_left_term) + theta_m * discrete_penalty_mask) * mask * (1 - mask)

    return gradient, z

def compute_cplx_gradient(mask, kernels, kernels_ct, kernel_def, kernel_def_ct, weight, weight_def, theta_z=50, theta_m=4, gamma=4):
    r"""
    Calculate the gradient of cplx objective: loss_cplx = ||Z_outer - Z_inner||_gamma
    Args:
        mask: input mask image
        kernels/kernels_ct: SOCS kernels (focus)
        kernel_def/kernel_def_ct: SOCS kernels (defocus)
        weight/weight_def: weights for SOCS kernels (focus/defocus)
        gamma, theta_z, theta_m: hyper-parameters for gradient calculation
    Return:
        Gradient tensor of loss_cplx
    """
    MAX_DOSE = 1.02
    MIN_DOSE = 0.98
    mask_inner_convolve_sigmoid_gradient, z_inner = compute_convolve_sigmoid_gradient(mask, kernel_def, kernel_def_ct, 
                                                                weight_def, dose=MIN_DOSE,  theta_z=theta_z, theta_m=theta_m)
    mask_outer_convolve_sigmoid_gradient, z_outer = compute_convolve_sigmoid_gradient(mask, kernels, kernels_ct, 
                                                                weight, dose=MAX_DOSE,  theta_z=theta_z, theta_m=theta_m)
    gradient = gamma * (z_outer - z_inner).pow(gamma-1) * (mask_outer_convolve_sigmoid_gradient - mask_inner_convolve_sigmoid_gradient)
    
    return gradient

def compute_gradient_scale(mask, target, kernels, kernels_ct, kernel_def, kernel_def_ct, weight, weight_def, new_cord, cplx_obj=False, origin_size=2048):
    r"""
    A wrapper function for gradient calculations of both loss_ilt and loss_cplx with scaling scheme
    Flow:
        Input mask (nn prediction) -> scale back to the cropped bbox size -> fit the cropped bbox into original size
                                   -> calculate the gradient -> crop and scale the gradient to input size -> gradient backward to nn
        [input_size, input_size] -> [cropped_bbox_size, cropped_bbox_size] -> [original_size, original_size] -> [cropped_bbox_size, cropped_bbox_size] -> [input_size, input_size]
    Args:
        mask: input mask image of size N * 1 * input_size * input_size, input_size = 512 by default
        target: target layout of size N * 1 * origin_size * origin_size, origin_size = 2048 by default
        kernels/kernels_ct: SOCS kernels (focus)
        kernel_def/kernel_def_ct: SOCS kernels (defocus)
        weight/weight_def: weights for SOCS kernels (focus/defocus)
    Return:
        Gradient tensor
    """
    assert len(mask.shape) == 4
    assert len(target.shape) == 4
    
    # NOTE: this function is only suitable with batch-size input
    # target is N * 1 * 2048 * 2048
    # input mask size is N * 1 * H * W (N * 1 * 512 * 512)

    # new_cord is a N * 4 tensor, cropped_bbox_size[i] = [rx(i) - lx(i), ry(i) - ly(i)], locate at new_cord[i]
    lx, ly, rx, ry = new_cord # lx, ly, rx, ry is N dim tensor
    
    batch_size = mask.shape[0]
    channel_size = mask.shape[1]
    cur_h = mask.shape[2]
    cur_w = mask.shape[3]

    assert batch_size == len(lx)
    assert cur_h == cur_w
    assert abs(rx - lx)[0].item() == abs(ry - ly)[0].item()

    # Gradient size is N * 1 * H * W
    gradient = torch.zeros((batch_size, channel_size, cur_h, cur_w), 
                    dtype=mask.dtype, layout=mask.layout, device=mask.device)
    
    for i in range(batch_size):
        cur_mask = mask[i].unsqueeze(0)
        cur_target = target[i].unsqueeze(0)
        mask_crop = torch.nn.functional.interpolate(cur_mask, 
                            size=(abs(rx - lx)[i].item(), abs(rx - lx)[i].item()), 
                            mode='nearest')
        mask_origin = torch.zeros((1, channel_size, origin_size, origin_size), 
                        dtype=mask.dtype, layout=mask.layout, device=mask.device)

        mask_origin[..., ly[i].item():ry[i].item(), lx[i].item():rx[i].item()] = mask_crop     
        if cplx_obj: # Objective = cplx
            gradient_origin = compute_cplx_gradient(mask_origin, kernels, kernels_ct, kernel_def, kernel_def_ct, weight, weight_def)
        else: # Objective = l2
            gradient_origin = compute_gradient(mask_origin, cur_target, kernels, kernels_ct, weight)
        gradient_crop = gradient_origin[..., ly[i].item():ry[i].item(), lx[i].item():rx[i].item()]
        gradient_tmp = torch.nn.functional.interpolate(gradient_crop, size=(cur_h, cur_w), 
                                        mode='bilinear', align_corners=False)
        gradient[i] = gradient_tmp.squeeze(0)

    return gradient # Return gradient size is N * 1 * H * W

def bit_mask_to_two_value_mask(mask):
    r"""
    Bridging the network prediction to the input of ILT loss layer
    Args:
        input mask \in {0, 1}
    Return:
        output mask \in {-1, 1}
    """
    return (mask.mul(2.0)).add(-1.0)

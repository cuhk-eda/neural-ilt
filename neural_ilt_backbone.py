import torch
import torch.nn as nn
from ilt_loss_layer import ilt_loss_layer


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )

# U-Net part is based on the implementation of
# https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py

class ILTNet(nn.Module):
    r"""
    ILTNet: 
    The main backbone model of Neural-ILT, a standard U-Net + ILT correction layer
    """
    def __init__(
        self,
        n_class,
        kernels,
        kernels_ct,
        kernels_def,
        kernels_def_ct,
        weight,
        weight_def,
        cycle_mode=False,
        cplx_obj=False,
        report_epe=False,
        in_channels=1,
    ):
        super(ILTNet, self).__init__()

        # Standard U-Net
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.sigmoid = nn.Sigmoid()

        # ILT loss layer
        self.ilt_loss_layer = ilt_loss_layer(
            kernels,
            kernels_ct,
            kernels_def,
            kernels_def_ct,
            weight,
            weight_def,
            cycle_mode=cycle_mode,
            cplx_obj=cplx_obj,
            report_epe=report_epe
        )
        self.report_epe = report_epe

    def forward(self, x, y, new_cord):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        # U-Net prediction
        mask = self.conv_last(x)
        x = self.sigmoid(mask)

        # Calculate the ILT loss with respect to the predicted mask
        out_loss = self.ilt_loss_layer(x, y, new_cord)

        if self.report_epe:
            out_loss, epe_violation = self.ilt_loss_layer(x, y, new_cord)
            return out_loss, mask, epe_violation
        else:
            out_loss, placeholder = self.ilt_loss_layer(x, y, new_cord)
            return out_loss, mask, placeholder

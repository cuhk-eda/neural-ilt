import torch
import torch.nn as nn


# U-Net implementation for PyTorch
# from https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    r"""
    Standard U-Net implementation based on https://github.com/usuyama/pytorch-unet
    """
    def __init__(self, n_class, in_channels=1):
        super(UNet, self).__init__()

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)  # 512
        x = self.maxpool(conv1)  # 256

        conv2 = self.dconv_down2(x)  # 256
        x = self.maxpool(conv2)  # 128

        conv3 = self.dconv_down3(x)  # 128
        x = self.maxpool(conv3)  # 64

        x = self.dconv_down4(x)  # 64

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )  # 128
        x = torch.cat([x, conv3], dim=1)  # 128

        x = self.dconv_up3(x)  # 128
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )  # 256
        x = torch.cat([x, conv2], dim=1)  # 256

        x = self.dconv_up2(x)  # 256
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )  # 512
        x = torch.cat([x, conv1], dim=1)  # 512

        x = self.dconv_up1(x)  # 512

        out = self.conv_last(x)

        return out

##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

######################################################################################
# U-Net: Convolutional Networks for BiomedicalImage Segmentation
# Paper-Link: https://arxiv.org/pdf/1505.04597.pdf
######################################################################################

class DoubleConvolution(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int
        ) -> 'Down':
        """
            Input Convlution Layer

            Parameters:
                in_ch (int): Number of input convolution channels
                out_ch (int): Number of output convolution channels
        """
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        return self.conv(x)


class InConvolution(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int
        ) -> 'Down':
        """
            Input Convlution Layer

            Parameters:
                in_ch (int): Number of input convolution channels
                out_ch (int): Number of output convolution channels
        """
        super(InConvolution, self).__init__()
        self.conv = DoubleConvolution(in_ch, out_ch)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        return self.conv(x)


class Down(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int
        ) -> 'Down':
        """
            Encoder Layer or Downsample Layer

            Parameters:
                in_ch (int): Number of input convolution channels
                out_ch (int): Number of output convolution channels
        """
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvolution(in_ch, out_ch)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        return self.conv(x)


class Up(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            bilinear: bool = True
        ) -> 'Up':
        """
            Decoder Layer or Upsample Layer

            Parameters:
                in_ch (int): Number of input convolution channels
                out_ch (int): Number of output convolution channels
        """
        super(Up, self).__init__()
        self.bilinear = bilinear

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConvolution(in_ch, out_ch)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x1 (torch.Tensor): Last layer output
                x2 (torch.Tensor): Skip connection output
            Returns:
                (torch.Tensor): Output Tensor
        """
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            input = x1,
            pad = [
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2
            ]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvolution(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int
        ) -> 'Down':
        """
            Output Convlution Layer

            Parameters:
                in_ch (int): Number of input convolution channels
                out_ch (int): Number of output convolution channels
        """
        super(OutConvolution, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        return self.conv(x)


class UNet(nn.Module):

    def __init__(
            self,
            n_classes: int
        ) -> 'UNet':
        """
            U-NET Network

            Parameters:
                n_classes (int): Number of classes
        """
        super(UNet, self).__init__()
    
        # Input layer
        self.inc = InConvolution(3, 64)
        # Encoder layers
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        # Decoder layers
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        # Output layers
        self.outc = OutConvolution(64, n_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Decode
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x



if __name__ == '__main__':

    # Choose the most optimal device available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = 10
    image_size = (3, 256, 512)

    # Initialize model
    model = UNet(
        classes = n_classes
    ).to(device)

    summary(model, image_size)

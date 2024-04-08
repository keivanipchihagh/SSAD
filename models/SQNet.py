##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

###################################################################
# SQNet: Speeding up semantic segmentation for autonomous driving
# Paper-Link:   https://openreview.net/pdf?id=S1uHiFyyg
###################################################################


class Fire(nn.Module):

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand_planes: int
    ) -> 'Fire':
        """
            Initialize Fire module.

            Parameters:
                inplanes (int): Number of input channels.
                squeeze_planes (int): Number of squeeze planes.
                expand_planes (int): Number of expand planes.

            Returns:
                None
        """
        super(Fire, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ELU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        x = self.conv1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out2 = self.conv3(x)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class ParallelDilatedConv(nn.Module):

    def __init__(
        self,
        inplanes,
        planes
    ) -> 'ParallelDilatedConv':
        """
            Parallel dilated convolution block.
            
            This module applies four dilated convolutions with different dilation
            factors to the input. The dilation factors are 1, 2, 3 and 4. The
            dilated convolutions are applied in parallel and the outputs are
            concatenated and then passed through a ReLU activation function.

            Paramters:
                inplanes (int): Number of input channels.
                planes (int): Number of output channels.
        """
        super(ParallelDilatedConv, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=4, dilation=4)
        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        self.relu3 = nn.ELU(inplace=True)
        self.relu4 = nn.ELU(inplace=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        out1 = self.dilated_conv_1(x)
        out2 = self.dilated_conv_2(x)
        out3 = self.dilated_conv_3(x)
        out4 = self.dilated_conv_4(x)
        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        out3 = self.relu3(out3)
        out4 = self.relu4(out4)
        out = out1 + out2 + out3 + out4
        return out


class SQNet(nn.Module):

    def __init__(
            self,
            n_classes: int
    ) -> 'SQNet':
        """
            SQNet Network

            Parameters:
                n_classes (int): Number of classes
        """
        super(SQNet, self).__init__()

        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1) # 32
        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire1_1 = Fire(96, 16, 64)
        self.fire1_2 = Fire(128, 16, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire2_1 = Fire(128, 32, 128)
        self.fire2_2 = Fire(256, 32, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire3_1 = Fire(256, 64, 256)
        self.fire3_2 = Fire(512, 64, 256)
        self.fire3_3 = Fire(512, 64, 256)
        self.parallel = ParallelDilatedConv(512, 512)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ELU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)
        self.relu3 = nn.ELU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(256, 96, 3, stride=2, padding=1, output_padding=1)
        self.relu4 = nn.ELU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(192, self.n_classes, 3, stride=2, padding=1, output_padding=1)

        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 32
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1) # 32
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 32
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 32
        self.conv1_1 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)   # 32
        self.conv1_2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1) # 32

        self.relu1_1 = nn.ELU(inplace=True)
        self.relu1_2 = nn.ELU(inplace=True)
        self.relu2_1 = nn.ELU(inplace=True)
        self.relu2_2 = nn.ELU(inplace=True)
        self.relu3_1 = nn.ELU(inplace=True)
        self.relu3_2 = nn.ELU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        x = self.conv1(x)
        x_1 = self.relu1(x)
        x = self.maxpool1(x_1)
        x = self.fire1_1(x)
        x_2 = self.fire1_2(x)
        x = self.maxpool2(x_2)
        x = self.fire2_1(x)
        x_3 = self.fire2_2(x)
        x = self.maxpool3(x_3)
        x = self.fire3_1(x)
        x = self.fire3_2(x)
        x = self.fire3_3(x)
        x = self.parallel(x)
        y_3 = self.deconv1(x)
        y_3 = self.relu2(y_3)
        x_3 = self.conv3_1(x_3)
        x_3 = self.relu3_1(x_3)
        x_3 = F.interpolate(x_3, y_3.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x_3, y_3], 1)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        y_2 = self.deconv2(x)
        y_2 = self.relu3(y_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.relu2_1(x_2)
        y_2 = F.interpolate(y_2, x_2.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x_2, y_2], 1)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        y_1 = self.deconv3(x)
        y_1 = self.relu4(y_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.relu1_1(x_1)
        x = torch.cat([x_1, y_1], 1)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.deconv4(x)
        return x



if __name__ == '__main__':

    # Choose the most optimal device available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = 10
    image_size = (3, 256, 512)

    # Initialize model
    model = SQNet(
        n_classes = n_classes
    ).to(device)

    summary(model, image_size)

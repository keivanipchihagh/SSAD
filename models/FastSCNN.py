##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

import torch
from typing import Any, Tuple
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

##################################################################################
# Fast-SCNN: Fast Semantic Segmentation Network
# Paper-Link: https://arxiv.org/pdf/1902.04502.pdf
##################################################################################


class _ConvBNReLU(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        **kwargs: Any
    ) -> '_ConvBNReLU':
        """
            Conv-BN-ReLU module.

            Parameters:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
                kernel_size (int, optional): Kernel size. Defaults to 3.
                stride (int, optional): Stride. Defaults to 1.
                padding (int, optional): Padding. Defaults to 0.
                **kwargs (Any): Additional keyword arguments.
        """
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
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


class _DSConv(nn.Module):

    def __init__(
        self,
        dw_channels: int,
        out_channels: int,
        stride: int = 1,
        **kwargs: Any
    ) -> None:
        """
            Initializes a Depthwise Separable Convolutions module.

            Parameters:
                dw_channels (int): The number of depthwise channels.
                out_channels (int): The number of output channels.
                stride (int): The stride of the convolution. Default is 1.
                **kwargs (dict): Additional keyword arguments.
        """
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
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


class _DWConv(nn.Module):

    def __init__(
        self,
        dw_channels: int,
        out_channels: int,
        stride: int = 1,
        **kwargs: Any
    ) -> None:
        """
            Initializes a Depthwise Convolutions module.

            Parameters:
                dw_channels (int): The number of depthwise channels.
                out_channels (int): The number of output channels.
                stride (int): The stride of the convolution. Default is 1.
                **kwargs (dict): Additional keyword arguments.
        """
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
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


class LinearBottleneck(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t: int = 6,
        stride: int = 2,
        **kwargs: Any
    ) -> 'LinearBottleneck':
        """
        Initializes a LinearBottleneck module used in MobileNetV2.

        Parameters:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            t (int): The expansion ratio of depthwise separable convolution. Default is 6.
            stride (int): The stride of the first convolution. Default is 2.
            **kwargs (dict): Additional keyword arguments.
        """
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PyramidPooling module.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            **kwargs (dict): Additional keyword arguments.
        """
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)


    def pool(
        self,
        x: torch.Tensor,
        size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Apply pooling operation to the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.
            size (Tuple[int, int]): The size for the pooling operation.

        Returns:
            torch.Tensor: The result of the pooling operation.
        """
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)


    def upsample(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """
        Upsample the input tensor to the specified size using bilinear interpolation.

        Parameters:
            x (torch.Tensor): The input tensor to upsample.
            size (Tuple[int, int]): The size to upsample the tensor to.

        Returns:
            torch.Tensor: The upsampled tensor.
        """
        return F.interpolate(x, size, mode='bilinear', align_corners=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):

    def __init__(
        self,
        dw_channels1: int = 32,
        dw_channels2: int = 48,
        out_channels: int = 64,
        **kwargs: Any,
    ) -> None:
        """
            Initialize Learning to downsample module.

            Parameters:
                dw_channels1 (int): Number of downsampled channels for the first conv layer.
                dw_channels2 (int): Number of downsampled channels for the second conv layer.
                out_channels (int): Number of output channels.
                **kwargs (dict): Additional keyword arguments.
        """
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(
        self,
        in_channels: int = 64,
        block_channels: Tuple[int, int, int] = (64, 96, 128),
        out_channels: int = 128,
        t: int = 6,
        num_blocks: Tuple[int, int, int] = (3, 3, 3),
        **kwargs: Any,
    ) -> 'GlobalFeatureExtractor':
        """
            Initialize the Global Feature Extractor.

            Parameters:
                in_channels (int): Number of input channels.
                block_channels (Tuple[int, int, int]): Number of channels for each block.
                out_channels (int): Number of output channels.
                t (int): Number of layers in each block.
                num_blocks (Tuple[int, int, int]): Number of blocks in each stage.
                **kwargs (dict): Additional keyword arguments.
        """
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)


    def _make_layer(
        self,
        block: nn.Module,
        inplanes: int,
        planes: int,
        blocks: int,
        t: int = 6,
        stride: int = 1,
    ) -> nn.Sequential:
        """Make a layer of blocks.

        Args:
            block (Type[nn.Module]): The type of the block.
            inplanes (int): The number of input channels.
            planes (int): The number of output channels.
            blocks (int): The number of blocks in the layer.
            t (int, optional): The number of layers in each block. Defaults to 6.
            stride (int, optional): The stride of the convolution. Defaults to 1.

        Returns:
            nn.Sequential: The constructed layer.
        """
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):

    def __init__(
        self,
        highter_in_channels: int,
        lower_in_channels: int,
        out_channels: int,
        scale_factor: int = 4,
        **kwargs: Any,
    ) -> 'FeatureFusionModule':
        """
            Initialize the Feature Fusion Module.

            Parameters:
                highter_in_channels (int): Number of input channels from the higher resolution.
                lower_in_channels (int): Number of input channels from the lower resolution.
                out_channels (int): Number of output channels.
                scale_factor (int, optional): Scale factor for downsampling. Defaults to 4.
                **kwargs (dict): Additional keyword arguments.
        """
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)


    def forward(
        self,
        higher_res_feature: torch.Tensor,
        lower_res_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
            higher_res_feature (torch.Tensor): Features from the higher resolution.
            lower_res_feature (torch.Tensor): Features from the lower resolution.

        Returns:
            torch.Tensor: Fused feature tensor.
        """
        _, _, h, w = higher_res_feature.size()
        lower_res_feature = F.interpolate(lower_res_feature, size=(h,w), mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(
        self,
        dw_channels: int,
        num_classes: int,
        stride: int = 1,
        **kwargs: Any,
    ) -> None:
        """
            Initialize the Classifer.

            Parameters:
                dw_channels (int): Number of input channels.
                num_classes (int): Number of output classes.
                stride (int, optional): Stride for the dilated convolution. Defaults to 1.
                **kwargs (dict): Additional keyword arguments.
        """
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class FastSCNN(nn.Module):

    def __init__(
        self,
        classes: int,
        aux: bool = False,
        **kwargs: Any,
    ) -> 'FastSCNN':
        """
            Initialize the FastSCNN model.

            Parameters:
                classes (int): Number of output classes.
                aux (bool, optional): Whether to use Auxiliary Classifier. Defaults to False.
                **kwargs (dict): Additional keyword arguments.
        """
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, classes, 1)
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the layer

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return x



if __name__ == '__main__':

    # Choose the most optimal device available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = 10
    image_size = (3, 256, 512)

    # Initialize model
    model = FastSCNN(
        classes = n_classes
    ).to(device)

    summary(model, image_size)
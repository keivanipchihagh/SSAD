##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

# Standard
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


# Paper:    https://arxiv.org/abs/2006.01413
# GitHub:   https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
class CrossEntropyLoss2d(_WeightedLoss):

    def __init__(
            self,
            weights: torch.Tensor = None,
            ignore_index: int = 255,
            reduction: str = 'mean'
        ) -> "CrossEntropyLoss2d":
        """
            Cross Entropy Loss Function for 2D vectors

            Parameters:
                weights (torch.Tensor): Dataset's Weights scheme
                ignore_index (int): Mask ID to ignore in calculations
                reduction (str): Reduction Algorithms
        """
        super(CrossEntropyLoss2d, self).__init__()
        self.name = "CrossEntropyLoss2d"

        # Fix Typing of weights
        if isinstance(weights, list):
            weights = torch.FloatTensor(weights)
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)

        self.nll_loss = nn.CrossEntropyLoss(weights, ignore_index=ignore_index, reduction=reduction)


    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
            Forward pass
        
            Parameters:
                outout (torch.Tensor): Model Output
                target (torch.Tensor): Desired Output
            Returns:
                (torch.Tensor): Loss Vector
        """
        return self.nll_loss(output, target)



# Paper:    https://arxiv.org/abs/1708.02002
# GitHub:   https://github.com/clcarwin/focal_loss_pytorch
class FocalLoss2d(nn.Module):

    def __init__(
            self,
            alpha: float = 0.5,
            gamma: float = 2,
            weights: torch.Tensor = None,
            ignore_index: int = 255,
            size_average: bool = True
        ):
        """
            Focal Loss Loss Function for 2D vectors

            Parameters:
                alpha (float): Alpha
                gamma (float): Gamma
                weights (torch.Tensor): Dataset's Weights scheme
                ignore_index (int): Mask ID to ignore in calculations
                size_average (str): Reduction method
        """
        super().__init__()
        self.name = "FocalLoss2d"
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

        if isinstance(weights, list):
            # List to torch.Tensor
            weights = torch.FloatTensor(weights)
        elif isinstance(weights, np.ndarray):
            # Numpy to torch.Float32
            weights = torch.from_numpy(weights)
            weights = weights.type(torch.FloatTensor)

        self.ce_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)


    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
            Forward pass
        
            Parameters:
                outout (torch.Tensor): Model Output of shape (N, C, H, W)
                target (torch.Tensor): Desired Output
            Returns:
                (torch.Tensor): Loss Vector
        """
        # Transform output
        if output.dim() > 2:
            # (N, C, H, W) -> (N*H*W, C)
            output = output.contiguous().view(output.size(0), output.size(1), -1)   # (N, C, H, W) -> (N, C, H*W)
            output = output.transpose(1, 2)                                         # (N, C, H*W) -> (N, H*W, C)
            output = output.contiguous().view(-1, output.size(2)).squeeze()         # (N, H*W, C) -> (N*H*W, C)

        # Transform target
        if target.dim() == 4:
            # (N, C, H, W) -> (N*H*W, C)
            target = target.contiguous().view(target.size(0), target.size(1), -1)   # (N, C, H, W) -> (N, C, H*W)
            target = target.transpose(1, 2)                                         # (N, C, H*W) -> (N, H*W, C)
            target = target.contiguous().view(-1, target.size(2)).squeeze()         # (N, H*W, C) -> (N*H*W, C)
        elif target.dim() == 3:
            # (N, H, W) -> (N*H*W)
            target = target.view(-1)

        logpt: torch.Tensor = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss: torch.Tensor = ((1-pt) ** self.gamma) * self.alpha * logpt

        return loss.mean() if self.size_average else loss.sum()

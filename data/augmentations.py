##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

# Standard
import numpy as np
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    PILToTensor,
)


class CityScapesTransforms(object):

    def __init__(self, size: int, mean: np.array, std: np.array):
        self.image_transform = Compose([
            Resize((size, size), antialias=True),
            ToTensor(),
        ])
        self.label_transform = Compose([
            Resize((size, size), antialias=True),
            PILToTensor(),  # Keep type for mask IDs
        ])


    def __call__(self, img, is_label: bool = False):
        if is_label: return self.label_transform(img)
        return self.image_transform(img)



class CamVidTransforms(object):

    def __init__(self, size: int, mean: np.array, std: np.array):
        self.image_transform = Compose([
            Resize((size, size), antialias=True),
            ToTensor(),
        ])
        self.label_transform = Compose([
            Resize((size, size), antialias=True),
            PILToTensor(),  # Keep type for mask IDs
        ])


    def __call__(self, img, is_label: bool = False):
        if is_label: return self.label_transform(img)
        return self.image_transform(img)

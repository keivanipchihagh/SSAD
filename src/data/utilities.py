##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

import numpy as np
import pandas as pd


def image_to_segmentation(image: np.ndarray, map: pd.DataFrame) -> np.ndarray:
    """
        Converts RGB image to Segmentation image

        Parameters:
            image (np.ndarray): RGB image
            map (pd.DataFrame): Segmentation map
        Returns:
            (np.ndarray): Segmentation image
    """
    segmentation = np.zeros(shape = image.shape).astype('int')

    def _(row: pd.Series):
        dims = image.shape[-1]
        is_r = image[:, :, 0] == row['r']   # Red
        is_g = image[:, :, 1] == row['g']   # Green
        is_b = image[:, :, 2] == row['b']   # Blue
        cond = (is_r & is_g & is_b)         # RGB
        segmentation[cond] = np.array([[row['id'] for _ in range(dims)]])

    map.apply(lambda row: _(row), axis = 1)
    return segmentation[:, :, 0]



def segmentation_to_image(segmentation: np.ndarray, map: pd.DataFrame) -> np.ndarray:
    """
        Converts Segmentation image to RGB image

        Parameters:
            image (np.ndarray): Segmentation image
            map (pd.DataFrame): Segmentation map
        Returns:
            (np.ndarray): RGB image
    """
    image = np.zeros(shape = (*segmentation.shape, 3)).astype('int')

    def _(row: pd.Series):
        cond = (segmentation == row['id'])
        image[cond, :] = np.array([row['r'], row['g'], row['b']])

    map.apply(lambda row: _(row), axis = 1)
    return image

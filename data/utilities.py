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
from typing import Tuple
from torch.utils.data import DataLoader
import torchvision.transforms as transform

# Third-party
from data.datasets import BaseDataset


def train_valid_loaders(
    root_dir: str,
    dataset: BaseDataset,
    batch_size: int,
    num_workers: int,
    transform: transform = None,
    shuffle: bool = True,
    n_images: int = None,
    ) -> Tuple[DataLoader, DataLoader]:
    """
        Returns training and validation DataLoaders

        Parameters:
            - root_dir (str): Root Directory of the dataset
            - dataset (BaseDataset): Dataset instance
            - transform (transform): List of Transformations to apply on Datasets
            - batch_size (int): DataLoader batch size
            - num_workers (int): How many processes to use for loading the data
            - shuffle (bool): Whether to shuffle the Datasets or not (default: True)
            - n_images (int): Number of images to use for training and validation sets (If None, all images will be used)
        Returns:
            - (Tuple[DataLoader, DataLoader]): Training and Validation DataLoaders
    """
    def _(root_dir: str, split: str):
        """ Wrapper """
        # Dataset
        _dataset = dataset(
            root_dir = root_dir,
            split = split,
            transform = transform,
            n_images = n_images,
        )
        # DataLoader
        loader = DataLoader(
            dataset = _dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = shuffle,
            pin_memory = True,
        )
        return loader

    train_loader = _(root_dir, split='train')   # Train loader
    valid_loader = _(root_dir, split='valid')   # Valid loader
    return train_loader, valid_loader



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

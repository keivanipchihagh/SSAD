##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

import os
from PIL import Image
from torch import Tensor
from typing import List, Tuple
from torch.utils.data import Dataset
import torchvision.transforms as transform

# Third-party
from utilities import walk


class BaseDataset(Dataset):

    def __init__(self) -> 'CityScapesDataset':
        super().__init__()

        self.transform = None
        self.images_dir: str = None
        self.labels_dir: str = None
        self.images_paths: List[str] = None
        self.labels_paths: List[str] = None


    def __len__(self) -> int:
        """ Returns number of images """
        return len(self.images_paths)


    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
            Returns image and its mask

            Parameters:
                - ids (int): ID of iamge and mask to return
            Returns:
                - (Tuple[Tensor, Tensor]): Image and Mask
        """
        image_path = self.images_paths[idx]
        label_path = self.labels_paths[idx]

        # Load images
        image = Image.open(image_path)
        label = Image.open(label_path).convert('L')

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            label = self.transform(label, is_label=True)

        return image, label



class CityScapesDataset(BaseDataset):

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            transform: transform = None,
            n_images: int = None,
        ) -> 'CityScapesDataset':
        """
            CityScapes Dataset

            Parameters:
                - root_dir (str): Root Directory of the dataset
                - split (str): Data split. Options: 'train' and 'valid'
                - transform (transform): Transformations to apply to the dataset
                - n_images (int): Number of images to use for training and validation sets (If None, all images will be used)
        """
        super().__init__()
        self.transform = transform
        split = 'val' if split == 'valid' else split

        # Get image directories
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.labels_dir = os.path.join(root_dir, 'gtFine', split)

        # Get image paths
        self.images_paths = walk(self.images_dir)
        self.labels_paths = walk(self.labels_dir, img_postfix='segment')

        # Select only a subset
        if n_images:
            self.images_paths = self.images_paths[:n_images]
            self.labels_paths = self.labels_paths[:n_images]



class CamVidDataset(BaseDataset):

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            transform: transform = None,
            n_images: int = None,
        ) -> 'CityScapesDataset':
        """
            CityScapes Dataset

            Parameters:
                - root_dir (str): Root Directory of the dataset
                - split (str): Data split. Options: 'train', 'valid' and 'test'
                - transform (transform): Transformations to apply to the dataset
                - n_images (int): Number of images to use for training and validation sets (If None, all images will be used)
        """
        super().__init__()
        self.transform = transform
        split = 'val' if split == 'valid' else split

        # Get image directories
        self.images_dir = os.path.join(root_dir, split)
        self.labels_dir = os.path.join(root_dir, split + "_labels")

        # Get image paths
        self.images_paths = walk(self.images_dir)
        self.labels_paths = walk(self.labels_dir, img_postfix='segment')

        # Select only a subset
        if n_images:
            self.images_paths = self.images_paths[:n_images]
            self.labels_paths = self.labels_paths[:n_images]

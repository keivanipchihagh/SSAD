##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

import os
import torch
import pandas as pd
from torch import nn
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

# Third-party
from data.utilities import train_valid_loaders
from utils.utilities import setup_seed, print_args
from data.datasets import CamVidDataset, CityScapesDataset
from data.augmentations import CityScapesTransforms, CamVidTransforms



def load_args() -> ArgumentParser:
    """
        Loads CMD Arguments

        Returns:
            - (ArgumentParser): Arguments
    """
    parser = ArgumentParser(description='')
    # Global
    parser.add_argument('--use_cuda',       type=bool,  default=True,           help="Run on CUDA (default: True)")
    parser.add_argument('--use_tb',         type=bool,  default=True,           help="Use Tensorboard (default: True)")
    parser.add_argument('--seed',           type=int,   default=42,             help="Random Seed (default: 42)")
    # Datasets
    parser.add_argument('--dataset',        type=str,   default="cityscapes",   help="Dataset Name (default: cityscapes)",  choices=["cityscapes", "camvid"])
    parser.add_argument('--batch_size',     type=int,   default=32,             help="Batch Size (default: 32)")
    parser.add_argument('--shuffle',        type=bool,  default=False,          help="Shuffle Dataset (default: True)")
    parser.add_argument('--num_workers',    type=int,   default=2,              help="Number of Workers (defualt: 4)")
    parser.add_argument('--n_images',       type=int,   default=2000,           help="Number of images to load for training and validation (default: None)")
    # Model
    parser.add_argument('--model',          type=str,   default="unet",         help="Model Architecture (default: unet)")
    # Training
    parser.add_argument('--max_epochs',     type=int,   default=100,            help="Maximum Number of Epochs (default: 100)")
    parser.add_argument('--resume',         type=str,   default=None,           help="Checkpoint to resume from (default: None)")
    parser.add_argument('--optim',          type=str,   default="adam",         help="Optimizer to use (default: adam)", choices=["sgd", "adam"])
    parser.add_argument('--lr',             type=float, default=5e-4,           help="Initial Learning Rate")
    parser.add_argument('--lr_schedule',    type=str,   default="poly",         help="Learning Rate Scheduler Policy (default: poly)", choices = ["poly"])

    return parser.parse_args()



if __name__ == '__main__':
    args = load_args()      # Load Arguments
    print_args(args)        # Print Arguments
    setup_seed(args.seed)   # Setup Random Seed for reproduction

    # -------- Identifier ----------
    identifier = f"{args.dataset}_{args.batch_size}_{args.n_images}_{args.model}_{args.optim}_{args.lr}_{args.lr_schedule}"
    os.makedirs(f'history/{identifier}/weights', exist_ok = True)
    os.makedirs(f'history/{identifier}/tensorboard', exist_ok = True)

    # -------- Tensorboard ---------
    tb_writer = None
    if args.use_tb:
        tb_writer = SummaryWriter(f"history/tensorboard/{identifier}")

    # ------- Configure CUDA -------
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')

    # ------------ Data ------------
    if args.dataset == 'cityscapes':
        # cityscapes
        n_classes = 34
        ignore_label = 255
        root_dir = "data/datasets/Cityspaces"
        dataset = CityScapesDataset
        augmentation = CityScapesTransforms(256)
    elif args.dataset == 'camvid':
        # camvid
        n_classes = 11
        ignore_label = 11
        root_dir = "data/datasets/CamVid"
        dataset = CamVidDataset
        augmentation = CamVidTransforms(256)

    train_loader, valid_loader = train_valid_loaders(
        root_dir = root_dir,
        dataset = dataset,
        batch_size = args.batch_size,
        num_workers =  args.num_workers,
        transform = augmentation,
        shuffle =  args.shuffle,
        n_images =  args.n_images,
    )
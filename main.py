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
from utils import logger
from models.unet import UNet
from models.train import Trainer
from data.utilities import train_valid_loaders
from utils.utilities import setup_seed, print_args
from models.loss import CrossEntropyLoss2d, FocalLoss2d
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
    parser.add_argument('--num_workers',    type=int,   default=4,              help="Number of Workers (defualt: 4)")
    parser.add_argument('--n_images',       type=int,   default=5000,           help="Number of images to load for training and validation (default: None)")
    # Model
    parser.add_argument('--model',          type=str,   default="unet",         help="Model Architecture (default: unet)")
    parser.add_argument('--criteria',       type=str,   default="focal_loss",   help="Criteria function (default: focal_loss)")
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
    identifier = f"{args.dataset}_{args.batch_size}_{args.n_images}_{args.model}_{args.criteria}_{args.optim}_{args.lr}_{args.lr_schedule}"
    os.makedirs(f'history/{identifier}/weights', exist_ok = True)
    os.makedirs(f'history/{identifier}/tensorboard', exist_ok = True)

    # -------- Tensorboard ---------
    tb_writer = None
    if args.use_tb:
        tb_writer = SummaryWriter(f"history/{identifier}/tensorboard")

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
    segmentation_map = pd.read_csv(f"{root_dir}/class_dict.csv")    # Segmentation mapper

    # ---------- Model -----------
    if args.model == 'unet':
        # U-Net
        model = UNet(n_classes)
    else:
        raise Exception("Unknown Model Name")

    # --------- Criteria ---------
    if args.criteria == 'focal_loss':
        criteria = FocalLoss2d(
            weights = None,
            ignore_index = ignore_label
        )
    elif args.criteria == 'cross_entropy':
        criteria = CrossEntropyLoss2d(
            weights = None,
            ignore_index = ignore_label
        )
    else:
        raise Exception("Unknown Criteria Name")

    # -------- Optimizer --------
    if args.optim == 'sgd':
        optim = torch.optim.SGD(
            params = filter(lambda p: p.requires_grad, model.parameters()),
            lr = args.lr,
            momentum = 0.9,
            weight_decay = 1e-4
        )
    elif args.optim == 'adam':
        optim = torch.optim.Adam(
            params = filter(lambda p: p.requires_grad, model.parameters()),
            lr = args.lr,
            betas = (0.9, 0.999),
            eps = 1e-08,
            weight_decay = 1e-4
        )
    else:
        raise Exception("Unknown Optimizer Name")

    # ------- Move to GPU -------
    if args.use_cuda:
        model = nn.DataParallel(model).cuda()   # Model
        criteria = criteria.cuda()              # Criteria

    # ------ Resume Training -------
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)        # Load checkpoint
            start_epoch = checkpoint['epoch']           # Load last epoch
            model.load_state_dict(checkpoint['model'])  # Load last state
            logger.info(f"Resuming from checkpoint '{args.checkpoint}' (Epoch {start_epoch})...")
        else:
            raise Exception(f"Checkpoint '{args.resume}' not found.")

    # ------- Train -------
    Trainer(
        model = model,
        criteria = criteria,
        optimizer = optim,
        scheduler = None,
        device = device,
        segmentation_map = segmentation_map
    ).run(
        start_epoch = start_epoch,
        end_epoch = args.max_epochs,
        train_loader = train_loader,
        valid_loader = valid_loader,
        tb_writer = tb_writer,
        identifier = identifier,
    )

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
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Third-party
from data.utilities import segmentation_to_image


class Trainer():

    def __init__(
            self,
            model: nn.Module,
            criteria,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler._LRScheduler,
            segmentation_map: pd.DataFrame,
            device: torch.device = torch.device("cuda"),
        ) -> 'Trainer':
        """
            Trainer

            Parameters:
                model (nn.Module): Model to train
                criteria: Loss function
                optimizer (optim.Optimizer): Optimization strategy
                device (torch.device): Device to mount the training on
                segmentation_map (pd.DataFrame): DataFrame of Segmentation mappings
        """
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.segmentation_map = segmentation_map


    def validate(self, loader: DataLoader) -> float:
        """
            Evaluates the model on the given DataLoader

            Parameters:
                loader (DataLoader): DataLoader to use
            Returns:
                (float): Loss
        """
        loss = 0
        self.model.eval()
        print("Validating", end = "\t")

        with torch.no_grad():
            for batch in loader:
                # Load batches
                images: torch.Tensor = batch[0]
                labels: torch.Tensor = batch[1]
                # Fix types
                images = images.type(torch.FloatTensor)
                labels = labels.type(torch.LongTensor)
                # Load into device
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.model(images)             # Get model predictions
                _loss = self.criteria(output, labels)   # Compute Batch loss
                loss += _loss                           # Cumulative loss
                print("=", end = "")

        print(f"> Loss: {loss}")
        return loss.item() / len(loader)   # Return epoch loss


    def train(self, loader: DataLoader):
        """
            Trains the model on the given DataLoader

            Parameters:
                loader (DataLoader): DataLoader to use
            Returns:
                (float): Loss
        """
        loss = 0
        self.model.train()
        print("Training", end = "\t")

        for _, batch in enumerate(loader):
            # Load batches
            images: torch.Tensor = batch[0]
            labels: torch.Tensor = batch[1]
            # Fix types
            images = images.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)
            # Load into device
            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images)             # Get model predictions
            _loss = self.criteria(output, labels)   # Compute Batch loss
            self.optimizer.zero_grad()              # Reset gradients
            _loss.backward()                        # Compute gradients loss
            self.optimizer.step()                   # Update parameters
            # self.scheduler.step()
            loss += _loss                           # Cumulative loss
            print("=", end = "")

        print(f"> Loss: {loss}")
        return loss.item() / len(loader)   # Return epoch loss


    def run(
        self,
        start_epoch: int,
        end_epoch: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        tb_writer: SummaryWriter = None,
        identifier: str = '',
    ) -> None:
        """
            Main train/eval Loop

            Parameters:
                start_epoch (int): Starting Epoch
                end_epoch (int): Ending Epoch
                train_loader (DataLoader): Training DataLoader
                valid_loader (DataLoader): Validation DataLoader
                tb_writer (SummaryWriter): Tensorboard Writer
                save_plot (bool): Whether to save plots or display them interactively
                identifier (str): Unique identifier for subfolder name
            Returns:
                None
        """
        for epoch in range(start_epoch, end_epoch):
            print(f"\nEPOCH {epoch})")

            # Training
            train_loss = self.train(train_loader)
            image = self.build_plot(train_loader, 5)
            tb_writer.add_image('training_sample', image, epoch)

            # Validation
            valid_loss = self.validate(valid_loader)
            image = self.build_plot(valid_loader, 5)
            tb_writer.add_image('validation_sample', image, epoch)

            # Save weights
            torch.save(obj = {"epoch": epoch + 1, "model": self.model.state_dict()}, f = f"history/{identifier}/weights/epoch_{epoch}.pth")
            # Plot metrics
            if tb_writer:
                tb_writer.add_scalars('loss', {'training': train_loss, 'validation': valid_loss}, epoch)


    def build_plot(
        self,
        loader: DataLoader,
        n_images: int = 5,
    ) -> np.ndarray:
        """
            Builds and returns a plot of input DataLoader for original, mask and labels

            Parameters:
                loader (DataLoader): DataLoader to get images from
                n_images (int): Number of images to plot
            Returns:
                (np.ndarray): Concatenated image
        """
        images = []
        train_tuples = [loader.dataset[idx] for idx in range(n_images)]

        with torch.no_grad():
            for _, (image, label) in enumerate(train_tuples):
                image = image.unsqueeze(0)                  # Add 0th dimention
                output: torch.Tensor = self.model(image)    # Make prediction

                # Convert original image to Numpy
                image = image.cpu().squeeze().permute(1, 2, 0).numpy()

                # Convert predicted segmentation to Numpy image
                probs = F.softmax(output, dim=1)
                mask = torch.argmax(probs, dim=1)
                mask = mask.cpu().squeeze().numpy()
                mask = segmentation_to_image(mask, self.segmentation_map)

                # Convert segemtnation label to Numpy image
                label = label.cpu().squeeze().numpy()
                label = segmentation_to_image(label, self.segmentation_map)

                image = np.concatenate([image, mask / 255, label / 255], axis = 0)
                images.append(image)

        # Return final image
        image = np.concatenate(images, axis = 1)
        return np.transpose(image, (2, 0, 1))   # (W, H, D) -> (D, W, H)

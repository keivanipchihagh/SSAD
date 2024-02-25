##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

import os
import json
import torch
import random
import numpy as np
from torch import nn
from glob import glob
from typing import List
from argparse import ArgumentParser


def setup_seed(seed: int) -> None:
    """
        Sets the seed for generating numbers

        Parameters:
            - seed (int): Seed Number
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def print_args(args: ArgumentParser) -> None:
    """
        Prints command line arguments

        Parameters:
            - args (ArgumentParser): Arguments
        Returns:
            None
    """
    for arg in vars(args):
        sep = '\t\t' if len(arg) < 7 else '\t'
        print(arg, getattr(args, arg), sep = ':' + sep)



def get_model_params(model: nn.Module) -> int:
    """
        Calculate total number of Model Parameters

        Parameters:
            - model (nn.Module): Model
        Returns:
            - (int): Number of Parameters
    """
    params = 0
    for parameter in model.parameters():
        param = 1
        for j in range(len(parameter.size())):
            param *= parameter.size(j)
        params += param
    return params



def walk(dir: str, img_postfix: str = '', extension: str = 'png') -> List[str]:
    """
        Walks through a directory and retrieves images from the subdirectories

        Parameters:
            - dir (str): Parent directory
            - img_postfix (str): Image postfix
            - extension (str): Image extention (default: png) 
        Returns:
            - (List[str]): List of relative paths for the images
    """
    paths: List[str] = []
    for dir, _, _ in os.walk(dir):
        sub_paths: List[str] = glob(f"{dir}/*{img_postfix}.{extension}")
        paths.extend(sub_paths)
    return paths



class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

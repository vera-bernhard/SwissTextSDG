import logging
import random

import numpy as np
import torch

from src.helpers.logging_helper import setup_logging

setup_logging()


def initialize_gpu_seed(seed: int):
    device, n_gpu = setup_gpu()

    init_seed_everywhere(seed, n_gpu)

    return device, n_gpu


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(f'Set seed for random, numpy and torch to {seed}')

def init_seed_everywhere(seed, n_gpu):
    init_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_gpu():
    # Setup GPU parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.warn('=' * 50)
    logging.warn('')
    logging.warn(f"\t[PyTorch]\tWe are using {str(device).upper()} on {n_gpu} gpu's.")
    logging.warn('')
    logging.warn('=' * 50)

    return device, n_gpu

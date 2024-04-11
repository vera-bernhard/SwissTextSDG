import csv
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.helpers.logging_helper import setup_logging
from src.data.dataset import SwissTextDataset
sys.path.append(os.getcwd())

setup_logging()


class PyTorchModel:

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #TODO: Implement the model 

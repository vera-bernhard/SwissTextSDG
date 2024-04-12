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
from src.models.config import Config
sys.path.append(os.getcwd())

setup_logging()


class PyTorchModel:

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = SwissTextDataset.create_instance(name=self.args.experiment_name, model_name=self.args.model_name,
                                                        use_val=self.args.use_val, seed=self.args.seed,
                                                        do_lower_case=self.args.do_lower_case,
                                                        max_seq_length=self.args.max_seq_length,
                                                        train_frac=self.args.train_frac)
        

        self.seed = self.args.seed
        self.model_seed = self.args.model_seed
        self.use_val = self.args.use_val

        self.model_config = Config.MODELS[self.args.model_name]

        self.model = self._get_pretrained_network(self.args.model_name, self.model_config.pretrained_model) 


    def _get_pretrained_network(self, model_name, model_name_or_path):
        config_class = Config.MODELS[model_name].model_config
        model_class = Config.MODELS[model_name].model_class

        config = config_class.from_pretrained(model_name_or_path)
        network = model_class.from_pretrained(model_name_or_path, config=config)

        # (Log)SoftmaxLayer to get softmax probabilities as output of our network, rather than logits
        if self.args.use_softmax_layer:
            new_clf = nn.Sequential(
                network.classifier,
                nn.LogSoftmax(dim=self.no_of_classes),
            )
            network.classifier = new_clf

        return network


import copy
import os
import sys

import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod

from src.helpers.logging_helper import setup_logging
from src.helpers.seed_helper import init_seed
from src.helpers.path_helper import *
from src.data.preprocessor import OSDGPreprocessor
from src.data.tokenizer import SwissTextTokenizer
from src.models.config import Config, DEFAULT_SEED, DEFAULT_SEQ_LENGTH, DEFAULT_TRAIN_FRAC, DEFAULT_NONMATCH_RATIO
sys.path.append(os.getcwd())


setup_logging()


# Wrapper class for datasets. The goal is to have all relevant CSVs accessed through this class, so that
# we do not have to wrangle with files and paths directly, but rather get what's' needed easily.
#
# To add a new dataset, simply add the Config in models/config.py

class SwissTextDataset(ABC):
    # Static method to expose available datasets
    @staticmethod
    def available_datasets():
        return Config.DATASETS.keys()
    
    @staticmethod
    def create_instance(name: str,
                        model_name: str,
                        use_val: bool,
                        seed: int = DEFAULT_SEED,
                        do_lower_case=True,
                        max_seq_length: int = DEFAULT_SEQ_LENGTH,
                        train_frac: float = DEFAULT_TRAIN_FRAC,):
        
        if name == 'OSDG':
            return OSDGDataset(model_name, use_val, seed, do_lower_case, max_seq_length, train_frac)
        elif name == 'enlarged_OSDG':
            return OSDGDataset(model_name, use_val, seed, do_lower_case, max_seq_length, train_frac)
        else:
            raise ValueError(f"Dataset {name} not supported")

    def __init__(self, name: str, model_name: str, use_val: bool,
                seed: int = DEFAULT_SEED, do_lower_case=True, max_seq_length: int = DEFAULT_SEQ_LENGTH,
                train_frac: float = DEFAULT_TRAIN_FRAC):
        self.name = self._check_dataset_name(name)
        self.raw_file_path = dataset_raw_file_path(Config.DATASETS[self.name])
        self.model_name = self._check_model_name(model_name)
        self.use_val = use_val
        self.seed = seed
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.train_frac = train_frac
        self.tokenizer = None

        # Set the seed on all libraries
        init_seed(self.seed)

    def _check_dataset_name(self, name):
        configured_dataset_names = Config.DATASETS.keys()

        if name not in configured_dataset_names:
            raise ValueError(f':dataset_name {name} should be one of [{configured_dataset_names}]')
        return name

    def _check_model_name(self, model_name):
        configured_model_names = Config.MODELS.keys()

        if model_name not in configured_model_names:
            raise ValueError(f':model_name {model_name} should be one of [{configured_model_names}]')
        return model_name
    
    def _get_label_list(self):
        data_df = self.preprocessor.get_raw_df()
        return data_df['label'].unique().tolist()
    
    def get_raw_df(self):
        return self.preprocessor.get_raw_df()
    
    def get_tokenized_data(self):
        try:
            return self.tokenized_data
        except AttributeError:
            tokenized_file_path = dataset_processed_file_path(self.name, 'tokenized_data__' + self.model_name + '.json',
                                                              seed=self.seed)

            if file_exists_or_create(tokenized_file_path):
                self.tokenized_data = pd.read_json(tokenized_file_path)

            else:
                self.tokenized_data, _ = self.tokenizer.tokenize_df(self.get_entity_data())
                self.tokenized_data.to_json(tokenized_file_path)

        return self.tokenized_data
    
        # returns the PyTorch dataloaders ready for use
    def get_data_loaders(self, batch_size: int = 8):
        train_df, test_df, validation_df = self.get_train_test_val()

        train_ds = PytorchDataset(model_name=self.name, idx_df=train_df, data_df=self.get_tokenized_data(),
                                  tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        test_ds = PytorchDataset(model_name=self.name, idx_df=test_df, data_df=self.get_tokenized_data(),
                                 tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

        if validation_df.empty:
            val_dl = None
        else:
            val_ds = PytorchDataset(model_name=self.name, idx_df=validation_df, data_df=self.get_tokenized_data(),
                                 tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
            val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size)

        return train_dl, test_dl, val_dl


class OSDGDataset(SwissTextDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = OSDGPreprocessor(raw_file_path= self.raw_file_path, dataset_name=self.name, seed=self.seed, tf_idf=False)
        self.tokenizer = SwissTextTokenizer(model_name=self.model_name, do_lower_case=self.do_lower_case,
                                            max_seq_length=self.max_seq_length)
        self.label_list = self._get_label_list()


class PytorchDataset(Dataset):
    def __init__(self, model_name: str, data_df: pd.DataFrame, tokenizer: SwissTextTokenizer,
                 max_seq_length: int):
        self.model_name = model_name
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.idx_df)

    def __getitem__(self, idx):
        seq = self.tokenized_data.iloc[idx]['tokenized']
        return seq

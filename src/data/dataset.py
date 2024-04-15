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
from src.data.dataset_utils import SplitMethod
from src.data.preprocessor import OSDGPreprocessor
from src.data.tokenizer import SwissTextTokenizer
from src.models.config import Config, DEFAULT_SEED, DEFAULT_SEQ_LENGTH
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
    def create_instance(dataset: str,
                        model_name: str,
                        use_val: bool,
                        seed: int = DEFAULT_SEED,
                        max_seq_length: int = DEFAULT_SEQ_LENGTH,
                        do_lower_case: bool = True,
                        train_frac: float = 0.8
                        ):
        
        if dataset == 'OSDG':
            return OSDGDataset(name = dataset, model_name = model_name, split_method = SplitMethod.RANDOM, 
                               use_val = use_val, seed = seed, max_seq_length = max_seq_length, 
                               do_lower_case = do_lower_case, train_frac = train_frac)
        elif dataset == 'enlarged_OSDG':
            return OSDGDataset(name = dataset, model_name = model_name, split_method = SplitMethod.RANDOM, 
                               use_val = use_val, seed = seed, max_seq_length = max_seq_length, 
                               do_lower_case = do_lower_case, train_frac = train_frac)
        else:
            raise ValueError(f"Dataset {dataset} not supported")

    def __init__(self, name: str, model_name: str, split_method: SplitMethod, use_val: bool,
                seed: int = DEFAULT_SEED, max_seq_length: int = DEFAULT_SEQ_LENGTH, do_lower_case: bool = True, train_frac: float = 0.8,
                ):
        self.name = self._check_dataset_name(name)
        self.raw_file_path = dataset_raw_file_path(Config.DATASETS[self.name])
        self.model_name = self._check_model_name(model_name)
        self.use_val = use_val
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.tokenizer = None
        self.split_method = split_method
        self.do_lower_case = do_lower_case
        self.train_frac = train_frac

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
        return data_df['sdg'].astype(int).unique().tolist()
    
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
                self.tokenized_data, _ = self.tokenizer.tokenize_df(self.preprocessor.get_entity_data())
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
    
    def get_split_method_name(self):
        return str(self.split_method).split('.')[1].lower()

    def get_train_test_val(self):
        try:
            return self.train_df, self.test_df, self.validation_df
        except AttributeError:
            method_name = self.get_split_method_name()
            train_file_path = dataset_processed_file_path(self.name, f'train__{method_name}__samples.csv',
                                                          seed=self.seed)
            test_file_path = dataset_processed_file_path(self.name, f'test__{method_name}__samples.csv',
                                                         seed=self.seed)
            validation_file_path = dataset_processed_file_path(self.name, f'val__{method_name}__samples.csv',
                                                               seed=self.seed)

            check_val = file_exists_or_create(validation_file_path) if self.use_val else True

            if file_exists_or_create(train_file_path) and file_exists_or_create(train_file_path) and check_val:
                self.train_df = pd.read_csv(train_file_path)
                self.test_df = pd.read_csv(test_file_path)
                self.validation_df = pd.read_csv(validation_file_path) if self.use_val else pd.DataFrame()
            else:
                # Split the dataset into train, test, and validation

                self.train_df, self.test_df, self.validation_df = \
                    self.get_train_test_val__implementation(self.train_frac)
                
                self.train_df.to_csv(train_file_path, index=False)
                self.test_df.to_csv(test_file_path, index=False)
                if not self.validation_df.empty:
                    self.validation_df.to_csv(validation_file_path, index=False)

        return self.train_df, self.test_df, self.validation_df
    

    def _random_split(self):
        def split_fn(df: pd.DataFrame, train_frac: float):
            train_df = df.sample(frac=train_frac, random_state=self.seed)
            test_df = df.drop(train_df.index)
            val_df = pd.DataFrame()
            if self.use_val:
                # split the validation set as half of the test set, i.e.
                # both test and valid sets will be of the same size
                #
                val_df = test_df.sample(frac=0.5, random_state=self.seed)
                test_df = test_df.drop(val_df.index)
            # TODO: Check if we need to return the dfs or just the indices (see model.train/model.test)
            return train_df, test_df, val_df

        return split_fn

    
    def pre_split(self):
        raise NotImplementedError("Needs to be implemented on subclass.")


    def get_train_test_val__implementation(self, train_frac: float):
        if self.split_method == SplitMethod.RANDOM:
            split_fn = self._random_split()
        elif self.split_method == SplitMethod.PRE_SPLIT:
            split_fn = self.pre_split()
        else:
            raise NotImplementedError(
                f"Split method '{self.split_method}' not implemented. \
                Make sure to include the seed when implementing a new one.")
        
        try:
            if self.use_val:
                return self.train_given, self.test_given, self.val_given
            else:
                return self.train_given, self.test_given, pd.DataFrame()
        except AttributeError:
            method_name = self.get_split_method_name()
            train_file_path = dataset_processed_file_path(self.name, f'train__{method_name}__samples.csv',
                                                          seed=self.seed)
            test_file_path = dataset_processed_file_path(self.name, f'test__{method_name}__samples.csv',
                                                         seed=self.seed)
            validation_file_path = dataset_processed_file_path(self.name, f'val__{method_name}__samples.csv',
                                                         seed=self.seed)

            check_val = file_exists_or_create(validation_file_path) if self.use_val else True

            if file_exists_or_create(train_file_path) and file_exists_or_create(train_file_path) and check_val:
                self.train_given = pd.read_csv(train_file_path)
                self.test_given = pd.read_csv(test_file_path)
                self.validation_given = pd.read_csv(validation_file_path) if self.use_val else None
            else:

                self.train_given, self.test_given, self.validation_given = split_fn(self.get_raw_df(), train_frac)
                self.train_given.to_csv(train_file_path, index=False)
                self.test_given.to_csv(test_file_path, index=False)
                if not self.validation_given.empty:
                    self.validation_given.to_csv(validation_file_path, index=False)

        return self.train_given, self.test_given, self.validation_given





class OSDGDataset(SwissTextDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = OSDGPreprocessor(raw_file_path= self.raw_file_path, dataset_name=self.name, seed=self.seed, tf_idf=False)
        self.tokenizer = SwissTextTokenizer(model_name=self.model_name,
                                            max_seq_length=self.max_seq_length,
                                            do_lower_case= self.do_lower_case)
        self.label_list = self._get_label_list()
        self.label_list.append(0) # Add the 0 label for the 'non-relevant' class
        self.no_labels = len(self.label_list)


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

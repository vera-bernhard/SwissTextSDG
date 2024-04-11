import os
import sys
sys.path.append(os.getcwd())

from transformers import AutoTokenizer
import torch

from src.helpers.logging_helper import setup_logging
import logging
from src.models.config import Config
import copy
import swifter

import pandas as pd
from abc import ABC, abstractmethod


setup_logging()


class SwissTextTokenizer(ABC):
    def __init__(self, model_name, do_lower_case, max_seq_length):
        self.model_name = model_name
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.tokenizer = self._setup_tokenizer()

    def __len__(self):
        return self.tokenizer.__len__()

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def _setup_tokenizer(self):
        '''
        Loads the corresponding tokenizer according to the configuration above
        '''
        pretrained_model = Config.MODELS[self.model_name].pretrained_model

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=self.do_lower_case)

        return tokenizer


    def tokenize_df(self, df_original: pd.DataFrame) -> pd.DataFrame:
        df = df_original.copy()

        logging.info(f'Start tokenizing.')

        tokenized_df = df \
            .swifter \
            .allow_dask_on_strings(enable=True) \
            .progress_bar(desc='Concatenating rows...') \
            .apply(self._get_concat_fn(), axis=1) \
            .to_frame(name='concat_data')

        tokenized_df['tokenized'] = tokenized_df \
            .swifter \
            .progress_bar(desc='Tokenizing rows...') \
            .apply(lambda row: self.tokenizer.tokenize(row['concat_data']), axis=1)

        logging.info('Done tokenizing.')

        return tokenized_df, df

    def _get_concat_fn(self):
        '''
        Provides the method for concatenating a single row in the dataframe
        '''
        def concat_fn(row):
            def string_conversion(cell):
                col_name, val = cell
                return '' if pd.isna(val) else str(val)

            return " ".join(map(string_conversion, row['text'])).strip()

        return concat_fn

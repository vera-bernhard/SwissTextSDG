import os
import sys
sys.path.append(os.getcwd())

from transformers import AutoTokenizer
import torch

from src.helpers.logging_helper import setup_logging
import logging
from src.data.data_config import Config
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


        if self.model_name.startswith('qlora'):
            tokenizer = Config.MODELS[self.model_name].tokenizer.from_pretrained(pretrained_model)            
            # For Qlora: set the padding token to the eos token
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_eos_token = True
            
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=self.do_lower_case)
        

        return tokenizer


    def tokenize_df(self, df_original: pd.DataFrame) -> pd.DataFrame:
        df = df_original.copy()

        logging.info(f'Start tokenizing.')

        tokenized_df = copy.deepcopy(df)

        tokenized_df['tokenized'] = tokenized_df \
            .swifter \
            .progress_bar(desc='Tokenizing rows...') \
            .apply(lambda row: self.tokenizer.tokenize(row['text']), axis=1)

        logging.info('Done tokenizing.')

        return tokenized_df, df
    
    def generate_input(self, token_seq, label):
        '''
        Generates the input for the model from a token sequence
        '''
        token_seq = copy.deepcopy(token_seq)
        # Truncate if necessary
        token_seq = self.truncate_sequence(token_seq)
        
        # Add special tokens
        if self.model_name.startswith('qlora'):
            token_seq = [self.tokenizer.bos_token] + token_seq + [self.tokenizer.eos_token]
        else:
            token_seq = ['[CLS]'] + token_seq + ['[SEP]']
        # Convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(token_seq)
        # Pad sequence
        input_ids, pad_tokens = self.pad_sequence(input_ids)
        # Input mask
        input_mask = [1] * len(token_seq) + [0] * pad_tokens
        # Segment ids
        segment_ids = [0] * len(token_seq) + [0] * pad_tokens

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        label = torch.tensor([label], dtype=torch.long)

        return (input_ids, input_mask, segment_ids, label)
    
    def truncate_sequence(self, tokens):
        '''
        Truncates the sequence to the maximum sequence length, we remove the last tokens on the right
        '''
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:self.max_seq_length - 2]
        return tokens
    
    def pad_sequence(self, input_ids):
        '''
        Pads the sequence to the maximum sequence length
        '''
        pad_tokens = self.max_seq_length - len(input_ids)
        input_ids += [0] * pad_tokens
        return input_ids, pad_tokens

from abc import ABC, abstractmethod
import pandas as pd
import copy

import logging
from src.helpers.path_helper import *
from src.models.config import DEFAULT_SEED
from src.data.text_preprocessing import TextProcessor


class Preprocessor(ABC):
    def __init__(self, raw_file_path: str, dataset_name: str, seed: int=DEFAULT_SEED):
        self.raw_file_path = raw_file_path
        self.dataset_name = dataset_name
        self.seed = seed


    def get_raw_df(self):
        try:
            self.raw_df
        except AttributeError:
            df = pd.read_csv(self.raw_file_path)
            self.raw_df = self._preprocess_raw_df(df)
        return self.raw_df


    @abstractmethod
    def _preprocess_raw_df(self, df: pd.DataFrame):
        """Function that deals with making the raw_df consistent if neeeded, i.e. changing the labels to integers, etc."""
        raise NotImplementedError("Should be implemented in the respective subclasses.")


    def get_entity_data(self):
        try:
            self.entity_data_df
        except AttributeError:
            entity_data_file_path = dataset_processed_file_path(self.dataset_name, 'entity_data.csv', seed=self.seed)

            if file_exists_or_create(entity_data_file_path):
                self.entity_data_df = pd.read_csv(entity_data_file_path)
            else:
                raw_df = self.get_raw_df()
                df = self._get_entity_data__implementation(raw_df)

                self.entity_data_df = df
                self.entity_data_df.to_csv(entity_data_file_path, index=False)

        return self.entity_data_df


    @abstractmethod
    def _get_entity_data__implementation(self, df):
        raise NotImplementedError("Needs to be implemented on subclass.")


class OSDGPreprocessor(Preprocessor):
    def __init__(self, raw_file_path: str, dataset_name: str, seed: int=DEFAULT_SEED, tf_idf: bool=False):
        super().__init__(raw_file_path, dataset_name, seed)
        self.tf_idf = tf_idf

    def _preprocess_raw_df(self, df: pd.DataFrame):
        return df

    def _get_entity_data__implementation(self, raw_df: pd.DataFrame):
        df = copy.deepcopy(raw_df)
        df = self.preprocess_text_samples(df)
        return df

    def preprocess_text_samples(self, df: pd.DataFrame):
        
        # if too slow -> multiple jobs/cores
        # We remove the punctuation + stopwords from samples
        df.loc[pd.notnull(df['text']), 'text'] = df.loc[pd.notnull(df['text']), 'text'] \
            .swifter \
            .allow_dask_on_strings(enable=True) \
            .progress_bar(desc='[Preprocessing] Normalizing text...') \
            .apply(lambda x: TextProcessor.normalize_text(x))

        if self.tf_idf:
            df = TextProcessor.tf_idf_ordering(df=df)

        return df


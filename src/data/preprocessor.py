from abc import ABC, abstractmethod
import pandas as pd
import copy

import logging
from src.helpers.path_helper import *
from src.models.mbert.config import DEFAULT_SEED
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
        """This function should implement the preprocessing of the raw df before tokenization."""
        raise NotImplementedError("Needs to be implemented on subclass.")


class OSDGPreprocessor(Preprocessor):
    def __init__(self, raw_file_path: str, dataset_name: str, seed: int=DEFAULT_SEED, tf_idf: bool=False):
        super().__init__(raw_file_path, dataset_name, seed)
        self.tf_idf = tf_idf

    def _preprocess_raw_df(self, df: pd.DataFrame):
        # Drop rows with label < 0.5
        df = copy.deepcopy(df)
        df = df.loc[pd.notnull(df['label'])]
        df = df.loc[df['label'] >= 0.5]
        # Drop rows with less than 10 words
        df = df.loc[df['text'].str.split().str.len() >= 10]
        # Drop rows with empty text
        df = df.loc[pd.notnull(df['text'])]
        return df

    def _get_entity_data__implementation(self, raw_df: pd.DataFrame):
        df = copy.deepcopy(raw_df)
        # Drop the paperId, sgd and label columns (not needed for tokenization)
        df = df.drop(columns=['paperId', 'sdg', 'label'])
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

class TrainSwissTextPreprocessor(Preprocessor):
    def __init__(self, raw_file_path: str, dataset_name: str, seed: int = DEFAULT_SEED, tf_idf: bool=False):
        super().__init__(raw_file_path, dataset_name, seed)
        self.tf_idf = tf_idf

    def get_raw_df(self):
        # Load the jsonl file
        try:
            self.raw_df
        except AttributeError:
            df = pd.read_json(self.raw_file_path, lines=True)
            self.raw_df = self._preprocess_raw_df(df)
        return self.raw_df
    
    def _preprocess_raw_df(self, df: pd.DataFrame):
        # Rename SDG column to sdg
        df = df.rename(columns={'SDG': 'sdg'})
        return df
    
    def _get_entity_data__implementation(self, raw_df: pd.DataFrame):
        df = copy.deepcopy(raw_df)
        # Drop the ID, URL and sdg columns (not needed for tokenization)
        df = df.drop(columns=[ 'ID', 'URL', 'sdg'])
        # Merge rows TITLE and ABSTRACT into one column called text
        df['text'] = df['TITLE'] + ' ' + df['ABSTRACT']
        df = df.drop(columns=['TITLE', 'ABSTRACT'])
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

class CombinedOSDGSwissTextPreprocessor(Preprocessor):

    def __init__(self, raw_file_path: str, dataset_name: str, seed: int = DEFAULT_SEED, tf_idf: bool=False):
        super().__init__(raw_file_path, dataset_name, seed)
        self.tf_idf = tf_idf
    
    def get_raw_df(self):
        try: 
            self.raw_df
        except AttributeError:
            osdg_path = dataset_raw_file_path('OSDG/osdg-community-data-v2024-01-01.csv')
            osdg_df = pd.read_csv(osdg_path, header=0, delimiter = '\t', encoding='utf-8')
            osdg_df = osdg_df[['text', 'sdg']]

            enlarged_osdg_path = dataset_raw_file_path('OSDG/citing_works_OSDG.csv')
            enlarged_osdg_df = pd.read_csv(enlarged_osdg_path)
            enlarged_osdg_df = enlarged_osdg_df[['text', 'sdg']]

            swisstext_path = dataset_processed_folder_path('swisstext_task1_train/seed_44/train__random__samples.csv') # We take only the train split, so we can evaluate on the test samples
            swisstext_df = pd.read_csv(swisstext_path)
            swisstext_df['text'] = swisstext_df['TITLE'] + ' ' + swisstext_df['ABSTRACT']
            swisstext_df = swisstext_df[['text', 'sdg']]

            enlarged_swisstext_path = dataset_raw_file_path('swisstext/citing_works_swisstext.csv')
            enlarged_swisstext_df = pd.read_csv(enlarged_swisstext_path)
            enlarged_swisstext_df = enlarged_swisstext_df[['text', 'sdg']]

            self.raw_df = pd.concat([osdg_df, enlarged_osdg_df, swisstext_df, enlarged_swisstext_df])
            self.raw_df = self.raw_df.reset_index(drop=True)

        return self.raw_df        
        
    def _preprocess_raw_df(self, df: pd.DataFrame):
        df = copy.deepcopy(df)
        df = df.loc[pd.notnull(df['text'])]

        # Drop rows with less than 10 words
        df = df.loc[df['text'].str.split().str.len() >= 10]

        # Get the min number of samples per class, we want all classes to have the same number of samples
        samples_per_class = df['sdg'].value_counts().min()

        # Downsample the overrepresented classes

        df = df.groupby('sdg').apply(lambda x: x.sample(samples_per_class, random_state=DEFAULT_SEED)).reset_index(drop=True)

        return df
    
    def _get_entity_data__implementation(self, raw_df):
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





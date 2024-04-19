import os
import sys
import re
import requests
import pickle
from tqdm import tqdm
import time

import pandas as pd


class SwissTextLoader:
    
    def __init__(self, swisstext_dataset_path, responses_dict_path):
        self.osdg_dataset_path = swisstext_dataset_path
        if os.path.exists(swisstext_dataset_path):
            self.osdg_data = pd.read_csv(swisstext_dataset_path, header=0, delimiter = '\t', encoding='utf-8')
        else:
            self.osdg_data = self.load_osdg_data()
        
        self.responses_dict_path = responses_dict_path
        if os.path.exists(responses_dict_path):
            with open(responses_dict_path, 'rb') as file:
                self.responses_dict = pickle.load(file)
        else:
            self.responses_dict = {}

    def save_responses_dict(self):
        with open(self.responses_dict_path, 'wb') as file:
            pickle.dump(self.responses_dict, file)

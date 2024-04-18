
import pandas as pd
import json

def read_in_data(json_file: str) -> pd.DataFrame:
    '''Read in the train data and return a pandas dataframe.'''
    data = []
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def add_textual_label(json_file: str, train_data: pd.DataFrame) -> pd.DataFrame:
    '''Add the textual label to the train data.'''
    with open(json_file, 'r', encoding='utf-8') as file:
        labels_dict = json.load(file)
        # Convert the keys to integers
        labels_dict = {int(k): v for k, v in labels_dict.items()}
    train_data['Label'] = train_data['SDG'].map(labels_dict)
    return train_data
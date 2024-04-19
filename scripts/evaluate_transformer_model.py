import os
import sys
import json

sys.path.append(os.getcwd())
from models.mbert.pytorch_model import PyTorchModel
from models.mbert.config import read_arguments_train
from src.helpers.logging_helper import setup_logging


setup_logging()

def main(args):

    model = PyTorchModel(args)

    # Load the evaluation data from the file
    f = open(eval_dataset, 'r')
    eval_data = f.readlines()
    f.close()

if __name__ == '__main__':
    args = read_arguments_train()
    eval_dataset = 'data/raw/task1_train.jsonl'
    main(args)

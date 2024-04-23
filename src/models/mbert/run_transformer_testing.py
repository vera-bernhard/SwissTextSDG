import os
import sys

sys.path.append(os.getcwd())
from models.mbert.pytorch_model import PyTorchModel
from src.data.dataset import PytorchDataset
from src.data.tokenizer import SwissTextTokenizer
from torch.utils.data import Dataset, DataLoader

from src.helpers.seed_helper import initialize_gpu_seed
from models.mbert.config import read_arguments_test
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import *


setup_logging()


def main(args):
    initialize_gpu_seed(args.model_seed)

    checkpoint_suffix = '__epoch' + str(args.epoch)
    if args.epoch == 0:
        checkpoint_suffix += '__zeroshot'

    file_name = "".join([args.model_name, checkpoint_suffix, '.pt'])
    checkpoint_path = experiment_file_path(args.experiment_name, file_name)

    model = PyTorchModel.load_from_checkpoint(args, checkpoint_path)
    
    # We always want to compare models to mbert trained only on the train samples of the swisstext_task1_train dataset
  
    # Modify the args to load the swisstext_task1_train dataset
    args.experiment_name = 'mbert_seed_0_swisstext_task1_train'
    args.epoch = 5
    swisstext_model = PyTorchModel(args)

    # Now set the test data loader of our model to that of the swisstext_model
    model.test_data_loader = swisstext_model.test_data_loader
    model.test(args.epoch)

if __name__ == '__main__':
    args = read_arguments_test()
    main(args)

import os
import sys

sys.path.append(os.getcwd())
from src.models.pytorch_model import PyTorchModel
from src.helpers.seed_helper import initialize_gpu_seed
from src.models.config import read_arguments_train
from src.helpers.logging_helper import setup_logging


setup_logging()


def main(args):
    initialize_gpu_seed(args.model_seed)

    model = PyTorchModel(args)
    model.train()

if __name__ == '__main__':
    args = read_arguments_train()
    main(args)
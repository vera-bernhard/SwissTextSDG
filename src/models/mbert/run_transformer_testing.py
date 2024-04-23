import os
import sys
import copy

sys.path.append(os.getcwd())
from src.models.mbert.pytorch_model import PyTorchModel
from src.data.dataset import PytorchDataset
from src.data.tokenizer import SwissTextTokenizer
from torch.utils.data import Dataset, DataLoader

from src.helpers.seed_helper import initialize_gpu_seed
from src.models.mbert.config import read_arguments_test, update_args_with_config
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
    
    # We always want to compare models to mbert trained only on the train samples of the swisstext_task1_train dataset trained with the following launch.json: 

#     {
#     // Use IntelliSense to learn about possible attributes.
#     // Hover to view descriptions of existing attributes.
#     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python Debugger: osdg_utils",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "src/models/mbert/run_transformer_training.py",
#             "console": "integratedTerminal",
#             "justMyCode": true,
#             "env": {
#             "PYTHONPATH": "anaconda3/envs/swisstextenv/bin/python",
#             },
#             "cwd": "${workspaceFolder}",
#             "args": ["--experiment_name", "mbert_seed_0_swisstext_task1_train",
#                      "--model_name", "mbert",
#                      "--dataset", "swisstext_task1_train",
#                      "--train_frac", "0.33",
#                      "--save_model",
#                      "--save_config",
#                 ],
#         }
#     ]
# }

  
    # Modify the args to load the swisstext_task1_train dataset
    new_args = copy.deepcopy(args)
    new_args.experiment_name = 'mbert_seed_0_swisstext_task1_train'
    new_args.epoch = 5
    new_args = update_args_with_config(experiment_name = new_args.experiment_name, args = new_args)

    swisstext_model = PyTorchModel(new_args)

    # Now set the test data loader of our model to that of the swisstext_model
    model.test_data_loader = swisstext_model.test_data_loader
    model.test(args.epoch)

if __name__ == '__main__':
    args = read_arguments_test()
    main(args)

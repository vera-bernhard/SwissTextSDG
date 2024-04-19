import argparse
import datetime
import json
import os
from dataclasses import dataclass


import logging
from src.helpers.logging_helper import setup_logging
setup_logging()

from src.helpers.path_helper import experiment_config_path, file_exists_or_create



DEFAULT_SEED = 44
DEFAULT_MODEL_SEED = 44
DEFAULT_SEQ_LENGTH = 256

@dataclass
class LanguageModelConfig:
    model_class: object
    model_config: object
    pretrained_model: str
    tokenizer: object

def write_config_to_file(args):
    config_path = experiment_config_path(args.experiment_name)
    file_exists_or_create(config_path)

    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logging.info(f'\tSuccessfully saved configuration at {config_path}')


def read_config_from_file(experiment_name: str):
    config_path = experiment_config_path(experiment_name)

    with open(config_path, 'r') as f:
        args = json.load(f)

    logging.info(f'\tSuccessfully loaded configuration for {experiment_name}')
    return args

def read_arguments_train():
    parser = argparse.ArgumentParser(description='Train a model on a dataset')

    parser.add_argument('--experiment_name', type=str, default='default_experiment', help='Name of the experiment')
    parser.add_argument('--model_name', type=str, default='mbert', help='Model to use')
    parser.add_argument('--dataset', type=str, default='OSDG', help='Dataset to use')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Seed for random number generators')
    parser.add_argument('--model_seed', type=int, default=DEFAULT_MODEL_SEED, help='Seed for model initialization')
    parser.add_argument('--seq_length', type=int, default=DEFAULT_SEQ_LENGTH, help='Maximum sequence length')
    parser.add_argument('--train_frac', type=float, default=0.8, help='Fraction of data to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--max_seq_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--do_lower_case', action='store_true', default=True, help='Use lower case')

    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--use_softmax_layer', action='store_true', default=True)

    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_config', action='store_true')

    args = parser.parse_args()

    if args.save_config:
        write_config_to_file(args)

    return args    

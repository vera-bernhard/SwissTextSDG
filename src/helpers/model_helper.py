import src.models.config as cfg
from src.helpers.path_helper import *
from src.helpers.seed_helper import initialize_gpu_seed
from src.models.pytorch_model import PyTorchModel


def load_model(args, update_config=False):
    if update_config:
        args = cfg.update_args_with_config(args.experiment_name, args)
        
    initialize_gpu_seed(args.model_seed)

    checkpoint_suffix = '__epoch' + str(args.epoch)
    if args.epoch == 0:
        checkpoint_suffix += '__zeroshot'

    file_name = "".join([args.model_name, checkpoint_suffix, '.pt'])
    checkpoint_path = experiment_file_path(args.experiment_name, file_name)

    model = PyTorchModel.load_from_checkpoint(args, checkpoint_path)
    return model
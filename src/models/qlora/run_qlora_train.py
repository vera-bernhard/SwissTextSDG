import os
import sys

sys.path.append(os.getcwd())
from src.models.qlora.qlora_model import QloraModel
from src.helpers.seed_helper import initialize_gpu_seed
from src.models.mbert.config import read_arguments_train
from src.helpers.logging_helper import setup_logging

# python src/models/qlora/run_qlora_train.py --experiment_name qlora-mistral_swisstext_task1_train --model_name qlora-mistral --dataset swisstext_task1_train --save_model --save_config --train_frac 0.33 --batch_size 1 --num_epochs 5 --no_stopword_removal

# python src/models/qlora/run_qlora_train.py --experiment_name qlora-mistral_swisstext_task1_train_big --model_name qlora-mistral --dataset combined_OSDG_swisstext_enlarged_OSDG_enlarged_swisstext --save_model --save_config --train_frac 1 --batch_size 1 --num_epochs 5 --no_stopword_removal


setup_logging()


def main(args):
    initialize_gpu_seed(args.model_seed)

    model = QloraModel(args)
    model.train()

if __name__ == '__main__':
    args = read_arguments_train()
    main(args)
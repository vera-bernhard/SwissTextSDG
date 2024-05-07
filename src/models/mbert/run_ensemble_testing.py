import os
import sys
import copy
import pandas as pd

sys.path.append(os.getcwd())
from src.models.mbert.pytorch_model import PyTorchModel
from src.data.dataset import PytorchDataset
from src.data.tokenizer import SwissTextTokenizer
from torch.utils.data import Dataset, DataLoader

from src.helpers.seed_helper import initialize_gpu_seed
from src.models.mbert.config import read_arguments_ensemble_test, update_args_with_config
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import *


setup_logging()


def main(args, experiments_args_dict):

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


    # Modify one of the experiments_args_dict to be the mbert trained on the swisstext_task1_train dataset
    new_args = copy.deepcopy(experiments_args_dict[list(experiments_args_dict.keys())[0]])
    new_args.experiment_name = 'mbert_seed_0_swisstext_task1_train'
    new_args.epoch = 5
    new_args = update_args_with_config(experiment_name = new_args.experiment_name, args = new_args)

    checkpoint_suffix = '__epoch' + str(5)
    swisstext_file_name = "".join(['mbert', checkpoint_suffix, '.pt'])
    swisstext_model = PyTorchModel.load_from_checkpoint(new_args, experiment_file_path(new_args.experiment_name, swisstext_file_name))

    # Check whether the mbert_seed_0_swisstext_task1_train has already been evaluated on the test set
    if not os.path.exists(experiment_file_path('mbert_seed_0_swisstext_task1_train', 'mbert__prediction_log__ep5.csv')): 
        swisstext_model.test(new_args.epoch)

    predictions_dict = {}

    for model_name, model_args in experiments_args_dict.items():
        initialize_gpu_seed(model_args.model_seed)

        checkpoint_suffix = '__epoch' + str(model_args.epoch)
        if model_args.epoch == 0:
            checkpoint_suffix += '__zeroshot'

        file_name = "".join([model_args.model_name, checkpoint_suffix, '.pt'])
        checkpoint_path = experiment_file_path(model_args.experiment_name, file_name)

        model = PyTorchModel.load_from_checkpoint(model_args, checkpoint_path)

        if model.args.model_name == 'mbert':
            # Set the test data loader of our model to that of the swisstext_model
            model.test_data_loader = swisstext_model.test_data_loader
            predictions_dict[model.args.experiment_name], labels_list = model.ensemble_test()

        else:
            # If the model is different, we need to tokenize the swisstext data with the model's tokenizer first
            test_dataloader = swisstext_model.test_data_loader
            test_df = test_dataloader.dataset.data_df
            tokenized_test_df, _ = model.dataset.tokenizer.tokenize_df(test_df)
            swisstext_model.test_data_loader.dataset.data_df['tokenized'] = tokenized_test_df['tokenized']
            swisstext_model.test_data_loader.dataset.tokenizer = model.dataset.tokenizer

            model.test_data_loader = swisstext_model.test_data_loader
            predictions_dict[model.args.experiment_name], labels_list = model.ensemble_test()

    # Now we have the predictions of all models in the predictions_list, we do the ensemble prediction by majority vote
    predictions_df = pd.DataFrame(columns=['label', 'prediction'])
    labels_list = labels_list.tolist()

    for test_sample in range(len(labels_list)):
        sample_predictions = []
        for model_name, model_predictions in predictions_dict.items():
            sample_predictions.append(model_predictions[test_sample])

        # We do the majority vote, if there is a tie, we take the first one
        ensemble_prediction = max(set(sample_predictions), key=sample_predictions.count).item()

        # We add the ensemble prediction to the predictions_df
        predictions_df = pd.concat([predictions_df, pd.DataFrame({'label': [labels_list[test_sample][0]], 'prediction': [ensemble_prediction]})])
    
    # Now we save the predictions_df to a csv file
    if not os.path.exists(experiment_file_path('ensemble', '')):
        os.makedirs(experiment_file_path('ensemble', ''))
    predictions_df.to_csv(experiment_file_path('ensemble', 'ensemble__prediction_log__ep5.csv'), index=False)


if __name__ == '__main__':
    args, experiments_args_dict = read_arguments_ensemble_test()
    main(args, experiments_args_dict)

import os
import sys
import copy
import pandas as pd

sys.path.append(os.getcwd())
from src.models.mbert.pytorch_model import PyTorchModel
from src.models.qlora.qlora_model import QloraModel
from src.data.dataset import PytorchDataset, SwissTextDataset
from src.data.tokenizer import SwissTextTokenizer
from torch.utils.data import Dataset, DataLoader

from src.helpers.seed_helper import initialize_gpu_seed
from src.models.mbert.config import read_arguments_ensemble_test, update_args_with_config
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import *


setup_logging()


def main(args, experiments_args_dict):

    # We always want to compare models to mbert trained only on the train samples of the swisstext_task1_train dataset trained with the following launch.json: 

  
    predictions_dict = {}
    probs_dict = {}

    for model_name, model_args in experiments_args_dict.items():

        if 'bert' in model_name:
            initialize_gpu_seed(model_args.model_seed)

            checkpoint_suffix = '__epoch' + str(model_args.epoch)
            if model_args.epoch == 0:
                checkpoint_suffix += '__zeroshot'

            file_name = "".join([model_args.model_name, checkpoint_suffix, '.pt'])
            checkpoint_path = experiment_file_path(model_args.experiment_name, file_name)

            model = PyTorchModel.load_from_checkpoint(model_args, checkpoint_path)

            
        elif 'qlora' in model_name:
            initialize_gpu_seed(model_args.model_seed)

            checkpoint_suffix = '__epoch' + str(model_args.epoch)
            if model_args.epoch == 0:
                checkpoint_suffix += '__zeroshot'

            file_name = "".join([model_args.model_name, checkpoint_suffix])
            checkpoint_path = experiment_file_path(model_args.experiment_name, file_name)
            model = QloraModel.load_from_checkpoint(model_args, checkpoint_path)
        
        
        test_data_loader = SwissTextDataset.create_test_instance(model_args)
        model.test_data_loader = test_data_loader

        predictions_dict[model.args.experiment_name], probs_dict[model.args.experiment_name], labels_list = model.ensemble_test()

           
    # Now we have the predictions of all models in the predictions_list, we do the ensemble prediction by majority vote
    predictions_df = pd.DataFrame(columns=['labels', 'predictions'])
    # Concatenate the labels list (list of lists of arrays)
    labels_list = [label[0] for sublist in labels_list for label in sublist]
    # Do the same for each model's predictions and probabilities
    for model_name, model_predictions in predictions_dict.items():
        predictions_dict[model_name] = [prediction for sublist in model_predictions for prediction in sublist]
        probs_dict[model_name] = [prob for sublist in probs_dict[model_name] for prob in sublist]

    for test_sample in range(len(labels_list)):
        sample_predictions = []
        for model_name, model_predictions in predictions_dict.items():
            sample_predictions.append(model_predictions[test_sample])

        sample_probs = []
        for model_name, model_probs in probs_dict.items():
            sample_probs.append(model_probs[test_sample])

        # Distribute the probabilities across classes
        class_probs = {}
        for i, pred in enumerate(sample_predictions):
            if pred not in class_probs:
                class_probs[pred] = [sample_probs[i]]
            else:
                class_probs[pred].append(sample_probs[i])

        # We do the majority vote, first check if there is a tie across the multiclass predictions
        pred_counts = {pred: sample_predictions.count(pred) for pred in set(sample_predictions)}
        max_pred_count = max(pred_counts.values())
        
        if list(pred_counts.values()).count(max_pred_count) > 1:
            # If there is a tie, we take the average of the probabilities of the tied classes
            tied_preds = [pred for pred, count in pred_counts.items() if count == max_pred_count]
            tied_class_probs = {}
            for key in tied_preds:
                tied_class_probs[key] = sum(class_probs[key]) / len(class_probs[key])
                
            # We take the class with the highest average probability
            ensemble_prediction = max(tied_class_probs, key=tied_class_probs.get)
            ensemble_prob = tied_class_probs[ensemble_prediction]
                   
        else:
            # If there is no tie, we take the majority vote
            ensemble_prediction = max(set(sample_predictions), key=sample_predictions.count)
            # We take the average of the probabilities of the majority class
            ensemble_prob = sum(class_probs[ensemble_prediction]) / len(class_probs[ensemble_prediction])

        # We add the ensemble prediction to the predictions_df
        predictions_df = pd.concat([predictions_df, pd.DataFrame({'labels': [labels_list[test_sample]], 'predictions': [ensemble_prediction],
                                                                  'model_probs':[ensemble_prob]})])
    
    # Now we save the predictions_df to a csv file
    if not os.path.exists(experiment_file_path('ensemble', '')):
        os.makedirs(experiment_file_path('ensemble', ''))
    predictions_df.to_csv(experiment_file_path('ensemble', 'ensemble__prediction_log__ep5.csv'), index=False)


if __name__ == '__main__':
    args, experiments_args_dict = read_arguments_ensemble_test()
    main(args, experiments_args_dict)

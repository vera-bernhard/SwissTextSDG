import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from models.mbert.pytorch_model import PyTorchModel
from models.mbert.config import read_arguments_eval
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import experiment_file_path
from scripts.analyze_prediction_log import calculate_scores, plot_scores, save_label_scores, save_accuracy_score, plot_confusion_matrix


setup_logging()

def main(args):
    # Load the prediction log of the model
    prediction_log_path = 'models/{}'.format(args.experiment_name) + '/mbert__prediction_log_epoch{}.csv'.format(args.epoch)
    model_predictions = pd.read_csv(prediction_log_path)
    model_scores = calculate_scores(model_predictions)

    # Benchmark predictions

    benchmark_prediction_log_path = 'models/{}'.format('mbert_seed_0_swisstext_task1_train') + '/mbert__benchmark_prediction_log_epoch{}.csv'.format(args.epoch)
    benchmark_predictions = pd.read_csv(benchmark_prediction_log_path)
    benchmark_scores = calculate_scores(benchmark_predictions)

    # Plot the scores together



def plot_scores_vs_benchmark(model_scores, benchmark_scores, experiment_name, epoch):
    plt.figure(figsize=(10, 5))
    plt.bar(model_scores.keys(), [score['f1'] for score in model_scores.values()], label='{}'.format(experiment_name))
    plt.bar(benchmark_scores.keys(), [score['f1'] for score in benchmark_scores.values()], label='Benchmark')
    plt.xlabel('SDG')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores for {experiment_name} at epoch {epoch} vs Benchmark Model')
    plt.legend()
    plt.savefig(experiment_file_path(experiment_name, "".join(['mbert', '__f1_scores__ep', str(epoch), '.png'])))

if __name__ == '__main__':
    args = read_arguments_eval()

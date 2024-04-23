import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from src.models.mbert.config import read_arguments_eval
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import experiment_file_path
from scripts.analyze_prediction_log import calculate_scores, save_label_scores, plot_confusion_matrix


setup_logging()

def main(args):
    # Load the prediction log of the model
    prediction_log_path = 'models/{}'.format(args.experiment_name) + '/mbert__prediction_log__ep{}.csv'.format(args.epoch)
    model_predictions = pd.read_csv(prediction_log_path)
    model_acc, model_scores = calculate_scores(model_predictions)

    # Benchmark predictions

    benchmark_prediction_log_path = 'models/{}'.format('mbert_seed_0_swisstext_task1_train') + '/mbert__prediction_log__ep{}.csv'.format(args.epoch)
    benchmark_predictions = pd.read_csv(benchmark_prediction_log_path)
    benchmark_acc, benchmark_scores = calculate_scores(benchmark_predictions)

    # Print the accuracies and save the F1 scores for each label

    print('Model accuracy: {}'.format(model_acc))
    print('Benchmark accuracy: {}'.format(benchmark_acc))
    save_label_scores(model_scores, args.experiment_name, args.epoch)
    save_label_scores(benchmark_scores, args.experiment_name, args.epoch, benchmark=True)

    # Plot the scores together
    plot_scores_vs_benchmark(model_scores, benchmark_scores, args.experiment_name, args.epoch)


def plot_scores_vs_benchmark(model_scores, benchmark_scores, experiment_name, epoch):
    # Plot the F1 scores in a Fig with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))

    # Plot the F1 scores of the model
    model_scores = pd.DataFrame(model_scores)
    model_scores = model_scores.transpose()
    # Sort the labels by the SDG (i.e. 0 to 17)
    model_scores = model_scores.sort_index()
    model_scores.plot(kind='bar', ax=axs[0])
    axs[0].set_title('{} F1 scores'.format(experiment_name))
    axs[0].set_xlabel('Label')
    axs[0].set_ylabel('F1 score')

    # Plot the F1 scores of the benchmark
    benchmark_scores = pd.DataFrame(benchmark_scores)
    benchmark_scores = benchmark_scores.transpose()
    # Sort the labels by the SDG (i.e. 0 to 17)
    benchmark_scores = benchmark_scores.sort_index()
    benchmark_scores.plot(kind='bar', ax=axs[1])
    axs[1].set_title('Benchmark F1 scores')
    axs[1].set_xlabel('Label')
    axs[1].set_ylabel('F1 score')

    # Save the plot
    plot_path = experiment_file_path(experiment_name, 'mbert__scores_comparison__ep{}.png'.format(epoch))
    plt.savefig(plot_path)


if __name__ == '__main__':
    args = read_arguments_eval()
    main(args)

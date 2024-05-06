import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from matplotlib import pyplot as plt

from src.helpers.path_helper import experiment_file_path

def load_prediction_log(experiment_name, epoch):
    log_path = experiment_file_path(experiment_name, "".join(['mbert', '__prediction_log__ep', str(epoch), '.csv']))
    return pd.read_csv(log_path)

def calculate_scores(df):
    # Calculate the overall accuracy for the multilabel classification
    num_correct = (df['labels'] == df['predictions']).sum()
    accuracy = num_correct / len(df)

    # Now calculate the precision/recall/f1 for each label
    labels = df['labels'].unique()
    label_scores = {}
    for label in labels:
        label_df = df[df['labels'] == label]
        num_correct = (label_df['labels'] == label_df['predictions']).sum()
        num_true = len(df[df['predictions'] == label])
        if num_true == 0:
            precision = 0
        else:
            precision = num_correct / len(df[df['predictions'] == label])
        recall = num_correct / len(label_df)
        sum_precision_recall = precision + recall
        if sum_precision_recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / sum_precision_recall
        label_scores[label] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return accuracy, label_scores

def plot_scores(scores, experiment_name, epoch):
    plt.figure(figsize=(10, 5))
    plt.bar(scores.keys(), [score['f1'] for score in scores.values()])
    plt.xlabel('SDG')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores for {experiment_name} at epoch {epoch}')
    plt.savefig(experiment_file_path(experiment_name, "".join(['mbert', '__f1_scores__ep', str(epoch), '.png'])))

def save_label_scores(scores, experiment_name, epoch, benchmark: bool = False):
    if benchmark:
        scores_path = experiment_file_path(experiment_name, "".join(['mbert', '__scores__ep', str(epoch), '_benchmark.csv']))
    else:
        scores_path = experiment_file_path(experiment_name, "".join(['mbert', '__scores__ep', str(epoch), '.csv']))
    # Save the scores as a CSV with the following columns:
    # label, precision, recall, f1
    with open(scores_path, 'w') as f:
        f.write('label,precision,recall,f1\n')
        for label, score in scores.items():
            f.write(f'{label},{score["precision"]},{score["recall"]},{score["f1"]}\n')

def save_accuracy_score(score, experiment_name, epoch, benchmark: bool = False):
    if benchmark:
        scores_path = experiment_file_path(experiment_name, "".join(['mbert', '__accuracy_score__ep', str(epoch), '_benchmark.csv']))
    else:
        scores_path = experiment_file_path(experiment_name, "".join(['mbert', '__accuracy_score__ep', str(epoch), '.csv']))
    with open(scores_path, 'w') as f:
        f.write('accuracy\n')
        f.write(f'{score[0]}\n')

def plot_confusion_matrix(df, experiment_name, epoch):

    # Calculate the confusion matrix
    confusion_matrix = pd.crosstab(df['labels'], df['predictions'])
    # Normalize the confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.matshow(confusion_matrix, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {experiment_name} at epoch {epoch}')
    plt.savefig(experiment_file_path(experiment_name, "".join(['mbert', '__confusion_matrix__ep', str(epoch), '.png'])))

def main():
    experiment_name = 'mbert_seed_0_enlargedOSDG'
    epochs = [1, 2, 3, 4, 5]

    for epoch in epochs:
        df = load_prediction_log(experiment_name, epoch)
        scores = calculate_scores(df)
        save_accuracy_score(scores, experiment_name, epoch)
        save_label_scores(scores[-1], experiment_name, epoch)
        plot_scores(scores[-1], experiment_name, epoch)
        plot_confusion_matrix(df, experiment_name, epoch)
    

if __name__ == '__main__':
    main()
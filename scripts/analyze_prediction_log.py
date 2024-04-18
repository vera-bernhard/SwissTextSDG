import os

import pandas as pd
from matplotlib import pyplot as plt

from helpers.path_helper import *

def load_prediction_log(experiment_name, epoch):
    log_path = experiment_file_path(experiment_name, "".join(['mbert', '__prediction_log__ep', str(epoch), '.csv']))
    return pd.read_csv(log_path)

def calculate_scores(df):
    num_labels = df['label'].nunique()
    num_correct = (df['label'] == df['prediction']).sum()
    accuracy = num_correct / len(df)
    precision = num_correct / len(df[df['prediction'] == 1])
    recall = num_correct / len(df[df['label'] == 1])
    f1_score = 2 * precision * recall / (precision + recall)

    # Now calculate the precision/recall/f1 for each label
    label_scores = {}
    for label in range(num_labels):
        label_df = df[df['label'] == label]
        num_correct = (label_df['label'] == label_df['prediction']).sum()
        precision = num_correct / len(label_df[label_df['prediction'] == label])
        recall = num_correct / len(label_df)
        f1 = 2 * precision * recall / (precision + recall)
        label_scores[label] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return accuracy, precision, recall, f1_score, label_scores

def plot_scores(scores, experiment_name, epoch):
    plt.figure(figsize=(10, 5))
    plt.bar(scores.keys(), [score['f1'] for score in scores.values()])
    plt.xlabel('Label')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores for {experiment_name} at epoch {epoch}')
    plt.savefig(experiment_file_path(experiment_name, "".join(['mbert', '__f1_scores__ep', str(epoch), '.png'])))

def save_scores(scores, experiment_name, epoch):
    scores_path = experiment_file_path(experiment_name, "".join(['mbert', '__scores__ep', str(epoch), '.csv']))
    # Save the scores as a CSV with the following columns:
    # label, precision, recall, f1
    with open(scores_path, 'w') as f:
        f.write('label,precision,recall,f1\n')
        for label, score in scores.items():
            f.write(f'{label},{score["precision"]},{score["recall"]},{score["f1"]}\n')

def main():
    experiment_name = 'mbert_seed_0_enlargedOSDG'
    epoch = 5

    df = load_prediction_log(experiment_name, epoch)
    scores = calculate_scores(df)
    save_scores(scores[-1], experiment_name, epoch)


if __name__ == '__main__':
    main()
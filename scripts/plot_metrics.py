import pandas as pd
from matplotlib import pyplot as plt


def plot_metrics(models):

    for model in models:
        filepath = f'../{model}/eval_metrics.csv'
        df = pd.read_csv(filepath)

        # Plotting the accuracy and F1 score for each class
        plt.figure(figsize=(10, 6))
        plt.bar(df['Class'], df['Accuracy'], color='violet', label='Accuracy')
        plt.bar(df['Class'], df['F1_Score'], color='green', alpha=0.5, label='F1 Score')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title(f'Accuracy and F1 Score per Class: {model}')
        plt.xticks(df['Class'])
        plt.legend()
        plt.grid(axis='y')
        plt.savefig(f'metrics_plot_{model}.png', dpi=300, bbox_inches='tight')

        plt.show()



def main():
    models = ['Mistral']
    plot_metrics(models)


if __name__ == "__main__":
    main()
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(models):

    for model in models:
        filepath = f'../{model}/classification_results.csv'
        df = pd.read_csv(filepath)

        conf_matrix = confusion_matrix(df['Gold_Standard'], df['Response'])

        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)  # adjust font size
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=sorted(df['Gold_Standard'].unique()),
                    yticklabels=sorted(df['Gold_Standard'].unique()))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix: {model}')
        plt.savefig(f'confusion_matrix_{model}.png', dpi=300, bbox_inches='tight')

        plt.show()


def main():
    models = ['Mistral']
    create_confusion_matrix(models)


if __name__ == "__main__":
    main()
    
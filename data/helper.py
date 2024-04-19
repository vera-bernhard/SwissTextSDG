
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import json



def read_in_data(json_file: str) -> pd.DataFrame:
    '''Read in the train data and return a pandas dataframe.'''
    data = []
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def add_textual_label(json_file: str, train_data: pd.DataFrame) -> pd.DataFrame:
    '''Add the textual label to the train data.'''
    with open(json_file, 'r', encoding='utf-8') as file:
        labels_dict = json.load(file)
        # Convert the keys to integers
        labels_dict = {int(k): v for k, v in labels_dict.items()}
    train_data['Label'] = train_data['SDG'].map(labels_dict)
    return train_data


def load_test_set(data_json, label_json):
    data_json = data_json
    label_json = label_json
    df = read_in_data(data_json)
    df = add_textual_label(label_json, df)
    # add a column containing concatenated title and abstract -> full_text
    df['full_text'] = df.apply(lambda row: row['TITLE'] + ' ' + row['ABSTRACT'], axis=1)
    # Stratified sampling to create a subset with 20% of the original dataset
    df_subset, _ = train_test_split(df, test_size=0.7, stratify=df['SDG'], random_state=42)

    print("Subset info:")
    print(df_subset.info())

    print("\nClass distribution in the subset:")
    print(df_subset['SDG'].value_counts())

    return df_subset


def compute_and_plot_metrics(df, dir_name, output_file=None):

    predicted = df['Response']
    actual = df['Gold_Standard']

    # calculate overall metrics
    accuracy = accuracy_score(actual, predicted)
    f1 = f1_score(actual, predicted, average='weighted')

    # Calculate accuracy per class
    accuracy_per_class = {}
    for class_label in set(actual):
        indices = actual == class_label
        accuracy_per_class[class_label] = accuracy_score(actual[indices], predicted[indices])

    # Calculate F1 score per class
    f1_per_class = f1_score(actual, predicted, labels=list(set(actual)), average=None)

    # Create DataFrame for metrics
    metrics_df = pd.DataFrame({
        'Class': list(accuracy_per_class.keys()),
        'Accuracy': list(accuracy_per_class.values()),
        'F1_Score': f1_per_class
    })

    # Save metrics to CSV file
    if output_file:
        metrics_df.to_csv(output_file, index=False)

    print('Overall accuracy: ', accuracy)
    print('Overall F1-score: ', f1)
    print(metrics_df)
    # Plot accuracy per class
    plt.figure(figsize=(10, 5))
    plt.bar(accuracy_per_class.keys(), accuracy_per_class.values(), color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Class')
    plt.xticks(list(accuracy_per_class.keys()))
    plt.show()
    plt.savefig(f'./{dir_name}/Accuracy.png')

    # Plot F1 score per class
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(f1_per_class)), f1_per_class, color='lightgreen')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class')
    plt.xticks(range(len(f1_per_class)), list(set(actual)))
    plt.show()
    plt.savefig(f'./{dir_name}/F1.png')
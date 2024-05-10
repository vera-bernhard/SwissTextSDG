
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import json
import os


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
    
    
def reformat_prediction_output(orig_jsonl: str, pred_csv: str, output_jsonl: str):
        prediction_df = pd.read_csv(pred_csv)
        
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
        # open both files
        with open(orig_jsonl, 'r', encoding='utf-8') as json_file, open(output_jsonl, 'a', encoding='utf-8') as output_file:
            # remove any contetn in the output file
            output_file.seek(0)
            output_file.truncate()
            
            # iterate through prediction_df and json_file
            for csv_line, json_line in zip(prediction_df.iterrows(), json_file):
                json_dict = json.loads(json_line)
                # convert csv_line to dictionary
                csv_dict = dict(csv_line[1])
                sgd = int(csv_dict['predictions'])
                json_dict['SDG'] = sgd
                
                # append the json_dict to the output file
                output_file.write(json.dumps(json_dict,ensure_ascii = False, separators=(',', ':')))
                output_file.write('\n')
                

if __name__ == '__main__':
    test_set = 'raw/swisstext/task1_test-covered.jsonl'
    predictions = '../models/ensemble/ensemble__prediction_log__ep5.csv'
    pred_test_set = 'predictions/predictions_test.jsonl'
        
    reformat_prediction_output(test_set, predictions, pred_test_set)
                 
                            
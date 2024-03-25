import os
import sys
import re
import requests
import pickle

import pandas as pd


def load_osdg_data():
    # Implement a function to load data from the OSDG dataset and return it as a pandas dataframe with the following columns:
    # - doi (the doi of the sample)
    # - text (the text of the sample)
    # - sdg (the corresponding SDG label)
    # - label (the agreement value that we will use as the target variable)

    data = pd.read_csv('data/OSDG/osdg-community-data-v2024-01-01.csv', header=0, delimiter = '\t', encoding='utf-8')

    # The columns of the dataframe should be: 'doi', 'text', 'sdg', 'label', the csv has columns doi, text_id, text, sdg, labels_negative, labels_positive, agreement
    # Remove the columns 'text_id', 'labels_negative', 'labels_positive'
    data = data.drop(columns=['text_id', 'labels_negative', 'labels_positive'])
    # Rename the column 'agreement' to 'label'
    data = data.rename(columns={'agreement': 'label'})
    return data

def process_API_response(response):
    # Implement a function to process the API response and return the related works as a pandas dataframe with the following columns:
    # - doi (the doi of the related work)
    # - text (the text of the related work)
    # - sdg (the corresponding SDG label)
    # - label (the agreement value that we will use as the target variable)

    if response.status_code == 200:
        data = response.json()
        references_count = data['message']['reference-count']
        if references_count > 0:
            references = data['message']['reference']
            related_works = pd.DataFrame(columns=['doi', 'text', 'sdg', 'label'])
            for reference in references:
                related_works = related_works.append({'doi': reference['DOI'], 'text': reference['title'], 'sdg': sdg, 'label': label}, ignore_index=True)
            return related_works
        else:
            return None
    else:
        return None

def get_related_works(osdg_sample):
    # Implement a function to retrieve the related works for a given OSDG sample (i.e. a row from the OSDG dataset) using the Crossref API
    # The function should return a pandas dataframe with the same columns as the OSDG dataset, but with the related works as rows:
    # - doi (the doi of the related work)
    # - text (the text of the related work)
    # - sdg (the corresponding SDG label)
    # - label (the agreement value that we will use as the target variable)

    # Get the doi of the OSDG sample
    doi = osdg_sample['doi']
    # Get the SDG of the OSDG sample
    sdg = osdg_sample['sdg']
    # Get the label of the OSDG sample
    label = osdg_sample['label']

    # Load/create a dictionary to store the API responses (so that we call the API only once for each doi)
    responses_dict_path = 'data/OSDG/responses_dict.pkl'
    if os.path.exists(responses_dict_path):
        with open(responses_dict_path, 'rb') as file:
            responses_dict = pickle.load(file)
    else:
        responses_dict = {}

    # Check if the doi is already in the dictionary
    if doi in responses_dict:
        response = responses_dict[doi]
    else:
        # Call the API
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url)
        # Add the response to the dictionary
        responses_dict[doi] = response
        # Save the dictionary
        with open(responses_dict_path, 'wb') as file:
            pickle.dump(responses_dict, file)

    # Process the API response
        
    related_works = process_API_response(response)

    return related_works

def enlarge_osdg_dataset(osdg_data):
    # Implement a function to enlarge the OSDG dataset by adding related works for each OSDG sample using the get_related_works function
    # The function should return a pandas dataframe with the same columns as the OSDG dataset, but with the related works as rows

    # Create an empty dataframe to store the related works
    related_works_df = pd.DataFrame(columns=['doi', 'text', 'sdg', 'label'])
    # Iterate over the OSDG samples
    for index, osdg_sample in osdg_data.iterrows():
        # Get the related works for the OSDG sample
        related_works = get_related_works(osdg_sample)

        if related_works is None:
            continue
        
        # Add the related works to the dataframe
        related_works_df = pd.concat([related_works_df, related_works], ignore_index=True)

    return related_works_df



def main():
    print('Loading OSDG data...')
    # Load the OSDG data
    osdg_data = load_osdg_data()
    print('OSDG data loaded successfully!')
    print('OSDG data:')
    print(osdg_data.head())

    print('Enlarging OSDG dataset...')
    # Enlarge the OSDG dataset by adding related works

    enlarged_osdg_data = enlarge_osdg_dataset(osdg_data)
    print('OSDG dataset enlarged successfully, {} related works added!'.format(len(enlarged_osdg_data)))

if __name__ == '__main__':
    main()
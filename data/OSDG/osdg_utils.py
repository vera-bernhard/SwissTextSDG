import os
import sys
import re
import requests


import pandas as pd


def load_osdg_data():
    # Implement a function to load data from the OSDG dataset and return it as a pandas dataframe with the following columns:
    # - doi (the doi of the sample)
    # - text (the text of the sample)
    # - sdg (the corresponding SDG label)
    # - label (the agreement value that we will use as the target variable)

    data = pd.read_csv('data/OSDG/osdg-community-data-v2024-01-01.csv', header=0, delimiter = '\t', index_col=0, encoding='utf-8')

    # The columns of the dataframe should be: 'doi', 'text', 'sdg', 'label', the csv has columns doi, text_id, text, sdg, labels_negative, labels_positive, agreement
    # Remove the columns 'text_id', 'labels_negative', 'labels_positive'
    data = data.drop(columns=['text_id', 'labels_negative', 'labels_positive'])
    # Rename the column 'agreement' to 'label'
    data = data.rename(columns={'agreement': 'label'})
    return data

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

    # Get the related works for the given doi using the Crossref API
    url = f'https://api.crossref.org/works/{doi}/relations'
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the related works from the response
        related_works = response.json()['message']['items']
        # Create a dataframe with the related works
        related_works_df = pd.DataFrame(columns=['doi', 'text', 'sdg', 'label'])
        for related_work in related_works:
            # Get the doi of the related work
            related_doi = related_work['DOI']
            # Get the text of the related work
            related_text = related_work['title']
            # Add the related work to the dataframe
            related_works_df = related_works_df.append({'doi': related_doi, 'text': related_text, 'sdg': sdg, 'label': label}, ignore_index=True)
        return related_works_df
    else:
        print(f'Error: {response.status_code}')
        return None

def enlarge_osdg_dataset(osdg_data):
    # Implement a function to enlarge the OSDG dataset by adding related works for each OSDG sample using the get_related_works function
    # The function should return a pandas dataframe with the same columns as the OSDG dataset, but with the related works as rows

    # Create an empty dataframe to store the related works
    related_works_df = pd.DataFrame(columns=['doi', 'text', 'sdg', 'label'])
    # Iterate over the OSDG samples
    for index, osdg_sample in osdg_data.iterrows():
        # Get the related works for the OSDG sample
        related_works = get_related_works(osdg_sample)
        # Add the related works to the dataframe
        related_works_df = related_works_df.append(related_works, ignore_index=True)
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
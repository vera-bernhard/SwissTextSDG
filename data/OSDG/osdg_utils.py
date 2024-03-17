import os
import sys

import pandas as pd
from pypdf import PdfReader


def load_osdg_data():
    # Implement a function to load data from the OSDG dataset and return it as a pandas dataframe with the following columns:
    # - doi (the doi of the sample)
    # - text (the text of the sample)
    # - sdg (the corresponding SDG label)
    # - label (the agreement value that we will use as the target variable)

    data = pd.read_csv('data/OSDG/osdg-community-data-v2024-01-01.csv')

    # The columns of the dataframe should be: 'doi', 'text', 'sdg', 'label', the csv has columns doi, text_id, text, sdg, labels_negative, labels_positive, agreement
    # Remove the columns 'text_id', 'labels_negative', 'labels_positive'
    data = data.drop(columns=['text_id', 'labels_negative', 'labels_positive'])
    # Rename the column 'agreement' to 'label'
    data = data.rename(columns={'agreement': 'label'})
    return data


def load_sdg_descriptions():
    # Implement a function to load the SDG goal/subgoal descriptions from the file 'data/OSDG/sdg_descriptions.pdf' and return it as a pandas dataframe with the following columns:
    # - 'sdg' (the SDG label)
    # - 'description' (the description of the SDG goal/subgoal)

    reader = PdfReader('data/OSDG/sdg_descriptions.pdf')
    # Each SDG goal/subgoal is preceded by the corresponding identifier (e.g. "Goal 1." for SDG 1, "1.1" for subgoal 1.1, etc.)

    # Extract the text from the pdf and store it in a pandas dataframe with the columns 'SDG' and 'description' for each page of the pdf
    sdg_descriptions = pd.DataFrame(columns=['sdg', 'description'])
    for page in reader.pages:
        text = page.extract_text()
        # Split the text by the newline character
        lines = text.split('\n')
        # Iterate over the lines and extract the SDG identifier and the corresponding description
        for i, line in enumerate(lines):
            if line.startswith('Goal'):
                # Extract the SDG number
                sdg = line.split('.')[0]
                # Extract the description
                description = lines[i+1]
                # Append the data to the dataframe
                sdg_descriptions = sdg_descriptions.append({'sdg': sdg, 'description': description}, ignore_index=True)
    
    return sdg_descriptions



def main():
    print('Loading OSDG data...')
    # Load the OSDG data
    data = load_osdg_data()
    print('OSDG data loaded successfully!')
    print('Loading SDG descriptions...')
    # Load the SDG descriptions
    sdg_descriptions = load_sdg_descriptions()
    print('SDG descriptions loaded successfully!')
    print('OSDG data:')
    print(data.head())
    print('SDG descriptions:')
    print(sdg_descriptions.head())

if __name__ == '__main__':
    main()
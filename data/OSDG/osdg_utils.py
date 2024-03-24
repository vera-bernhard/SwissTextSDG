import os
import sys
import re
import requests


import pandas as pd
from pypdf import PdfReader


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


def load_sdg_descriptions():
    # Implement a function to load the SDG goal/subgoal descriptions from the file 'data/OSDG/sdg_descriptions.pdf' and return it as a pandas dataframe with the following columns:
    # - 'sdg' (the SDG label)
    # - 'description' (the description of the SDG goal/subgoal)

    # Check if we have already extracted the SDG descriptions and saved them to a csv file
    if os.path.exists('data/OSDG/sdg_descriptions.csv'):
        return pd.read_csv('data/OSDG/sdg_descriptions.csv')

    reader = PdfReader('data/OSDG/sdg_descriptions.pdf')
    # Each SDG goal/subgoal is preceded by the corresponding identifier (e.g. "Goal 1." for SDG 1, "1.1" for subgoal 1.1, etc.)

    # Extract the text from the pdf and store it in a pandas dataframe with the columns 'SDG' and 'description' for each page of the pdf
    sdg_descriptions = pd.DataFrame(columns=['sdg', 'description'])
    for page in reader.pages:
        text = page.extract_text()
        lines = text.split('\n')
        # Find the SDG of the current page, its described as Goal X. 
        sdg = re.findall(r'\bGoal \d+\.', text)
        if sdg: 
            # Extract the SDG identifier from the text (i.e. the X without the "Goal" prefix and the "." suffix)
            sdg = re.findall(r'Goal (\d+)\.', sdg[0])
            # Extract the description of the SDG, that is the text after the SDG identifier and before the next line
            description = re.findall(r'Goal \d+\.\s*(.*)\n', text)

            # Add the SDG and its description to the dataframe
            sdg_descriptions = sdg_descriptions._append({'sdg': sdg[0], 'description': description[0]}, ignore_index=True)

        # Find the subgoals of the current page, they are described as X.Y and are always at the beginning of a line
        # we don't want to catch X.Y.Z as those are the indicators or any other X.Y not at the beginning of a line
        subgoals = re.findall(r'\n(\d+\.\d+)\s', text)

        for subgoal in subgoals:

            # Extract the subgoal identifier from the text (i.e. the X.Y)
            subgoal = re.findall(r'(\d+\.\d+)', subgoal)
            # Get all the text in the page after the subgoal identifier and before the next subgoal or indicator (X.Y or X.Y.Z) or blank line, or line starting with -
            # the description is in multiple lines so we first find the line where the subgoal is and then we get all the text until the next subgoal or indicator
            line = [line for line in lines if subgoal[0] in line][0]
            index = lines.index(line)
            # Get the description of the subgoal (i.e remove the subgoal identifier)
            description = line.replace(subgoal[0], '')
            for i in range(index+1, len(lines)):
                if re.match(r'\d+\.\d+\.\d+', lines[i]) or re.match(r'\d+\.\d+', lines[i]) or lines[i] == '' or lines[i][0] == '-':
                    break
                description += lines[i]

            # Add the subgoal and its description to the dataframe
            sdg_descriptions = sdg_descriptions._append({'sdg': subgoal[0], 'description': description}, ignore_index=True)

    # Save the dataframe to a csv file
    sdg_descriptions.to_csv('data/OSDG/sdg_descriptions.csv', index=False)
    return sdg_descriptions

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
    print('Loading SDG descriptions...')
    # Load the SDG descriptions
    sdg_descriptions = load_sdg_descriptions()
    print('SDG descriptions loaded successfully!')
    print('OSDG data:')
    print(osdg_data.head())
    print('SDG descriptions:')
    print(sdg_descriptions.head())

if __name__ == '__main__':
    main()
import os
import sys
import re
import requests
import pickle

import pandas as pd

class OSDGDataLoader:

    def __init__(self, osdg_dataset_path, responses_dict_path):

        self.osdg_dataset_path = osdg_dataset_path
        if os.path.exists(osdg_dataset_path):
            self.osdg_data = pd.read_csv(osdg_dataset_path, header=0, delimiter = '\t', encoding='utf-8')
        else:
            self.osdg_data = None
        
        self.responses_dict_path = responses_dict_path
        if os.path.exists(responses_dict_path):
            with open(responses_dict_path, 'rb') as file:
                self.responses_dict = pickle.load(file)
        else:
            self.responses_dict = {}

    def save_responses_dict(self):
        with open(self.responses_dict_path, 'wb') as file:
            pickle.dump(self.responses_dict, file)

    def load_osdg_data(self):
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

    def process_API_response(self, response, doi):
        # Implement a function to process a works endpoint API response and return the citations of the work as a pandas dataframe with the following columns:
        # - doi (the doi of the related work)

        if response.status_code == 200:
            data = response.json()
            citations_count = data['message']['is-referenced-by-count']

            if citations_count > 0:
            # Get the citations of the work 
                citations_url = f"https://api.crossref.org/works/{doi}/citations"
                # Check if we have already called this API endpoint
                if citations_url in self.responses_dict:
                    citations_response = self.responses_dict[citations_url]
                else:
                    citations_response = requests.get(citations_url)
                    self.responses_dict[citations_url] = citations_response
                    self.save_responses_dict(self.responses_dict_path)

                if citations_response.status_code == 200:
                    citations_data = citations_response.json()
                    dois = [citation['doi'] for citation in citations_data['message']['items']]
                    related_works = pd.DataFrame({'doi': dois})
                    return  related_works
                else:
                    return None
        else:
            None

    def get_related_works(self, osdg_sample):
        # Implement a function to retrieve the related works for a given OSDG sample (i.e. a row from the OSDG dataset) using the Crossref API
        # The function should return a pandas dataframe with the same columns as the OSDG dataset, but with the related works as rows:
        # - doi (the doi of the related work)
        # - text (the text of the related work)

        # Get the doi of the OSDG sample
        doi = osdg_sample['doi']

        url = f"https://api.crossref.org/works/{doi}"

        # Check if the doi is already in the dictionary
        if url in self.responses_dict:
            response = self.responses_dict[url]
        else:
            # Call the API
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url)
            # Add the response to the dictionary
            self.responses_dict[url] = response
            # Save the dictionary
            self.save_responses_dict()

        # Process the API response
            
        related_works = self.process_API_response(response, doi)

        return related_works

    def enlarge_osdg_dataset(self, osdg_data):
        # Implement a function to enlarge the OSDG dataset by adding related works for each OSDG sample using the get_related_works function
        # The function should return a pandas dataframe with the same columns as the OSDG dataset, but with the related works as rows

        # Create an empty dataframe to store the related works dois
        related_works_df = pd.DataFrame(columns=['doi'])
        # Iterate over the OSDG samples
        for index, osdg_sample in osdg_data.iterrows():
            # Get the related works for the OSDG sample
            related_works = self.get_related_works(osdg_sample)

            if related_works is None:
                continue
            
            # Add the related works to the dataframe
            related_works_df = pd.concat([related_works_df, related_works], ignore_index=True)

        return related_works_df



def main():
    osdg_dataset_path = 'data/OSDG/osdg_dataset.csv'
    responses_dict_path = 'data/OSDG/responses_dict.pkl'

    osdg_data_loader = OSDGDataLoader(osdg_dataset_path, responses_dict_path)

    osdg_data = osdg_data_loader.load_osdg_data()

    related_works = osdg_data_loader.enlarge_osdg_dataset(osdg_data)

    print(related_works.head())

if __name__ == '__main__':
    main()
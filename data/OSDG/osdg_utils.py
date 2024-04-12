import os
import sys
import re
import requests
import pickle
from tqdm import tqdm

import pandas as pd

class OSDGDataLoader:

    def __init__(self, osdg_dataset_path, responses_dict_path):

        self.osdg_dataset_path = osdg_dataset_path
        if os.path.exists(osdg_dataset_path):
            self.osdg_data = pd.read_csv(osdg_dataset_path, header=0, delimiter = '\t', encoding='utf-8')
        else:
            self.osdg_data = self.load_osdg_data()
        
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
    
    def get_semantic_scholar_citations_url(self, doi):
        # Implement a function to get the Semantic Scholar API url to retrieve the citations of a given doi
        # The function should return the url as a string

        url = f'https://api.semanticscholar.org/graph/v1/paper/{doi}/citations?fields=title,abstract'
        return url
    
    def process_semantic_scholar_citation_response(self, response):
        # Implement a function to process the response from the Semantic Scholar API and return a pandas dataframe with the following columns:

        # - paperId (the Semantic Scholar paperId of the citing work)
        # - text (text of the citing work, title and/or abstract)

        if response.status_code == 200:
            data = response.json()
            # Check if the response has no data
            try:
                data['data']
            except KeyError:    
                return None
            
            if len(data['data']) == 0:
                return None

            related_works = pd.DataFrame(columns=['paperId', 'text'])
            for citation in data['data']:
                # data['data'] is a list of dictionaries with keys 'citingPaper'
                # each 'citingPaper' dictionary has keys 'paperId' (required), 'title' (optional), 'abstract' (optional)
                paperId = citation['citingPaper']['paperId']
                title = citation['citingPaper'].get('title', '')
                abstract = citation['citingPaper'].get('abstract', '')
                if abstract is not None:
                    # Remove the punctuation, newlines and special characters from the abstract
                    abstract = re.sub(r'[^\w\s]', '', abstract)
                    abstract = re.sub(r'\n', ' ', abstract)
                    abstract = re.sub(r'\t', ' ', abstract)

                    text = title + ' ' + abstract
                else:
                    text = title
                related_works = pd.concat([related_works, pd.DataFrame({'paperId': [paperId], 'text': [text]})], ignore_index=True)
                
            return related_works
        else:
            return None

    def get_related_works(self, osdg_sample):
        # Implement a function to retrieve the works that cite a given OSDG sample (i.e. a row from the OSDG dataset) using the Semantic Scholar Academic Graph API
        # The function should return a pandas dataframe with the following columns:

        # - doi (the doi of the citing work)
        # - text (text of the citing work, title and/or abstract)

        # Get the doi of the OSDG sample
        doi = osdg_sample['doi']

        url = self.get_semantic_scholar_citations_url(doi)

        # Check if we have queried this url before

        if url in self.responses_dict:
            response = self.responses_dict[url]
        else:
            # Call the API
            response = requests.get(url)
            # Add the response to the dictionary
            self.responses_dict[url] = response
            # Save the dictionary
            self.save_responses_dict()

        # Process the API response
            
        related_works = self.process_semantic_scholar_citation_response(response)

        return related_works

    def build_related_work_sample(self, osdg_sample, related_work):
        # Implement a function to build a related work sample using the OSDG sample and the related work
        # The function should return a pandas dataframe with the same columns as the OSDG dataset but with the related work as the only row
        # We will assign the same SDG label and agreement value to the related work as its corresponding OSDG sample

        related_work_sample = pd.DataFrame({'paperId': related_work['paperId'], 
                                                          'text': related_work['text'], 
                                                          'sdg': osdg_sample['sdg'], 
                                                          'label': osdg_sample['label']})
        
        return related_work_sample




    def enlarge_osdg_dataset(self, osdg_data):
        # Implement a function to enlarge the OSDG dataset by adding citing works for each OSDG sample using the get_related_works function
        # The function should return a pandas dataframe with the same columns as the OSDG dataset, but with the citing works as rows

        # Check if we have already enlarged the dataset
        enlarged_dataset_path = os.path.join('data', 'OSDG', 'citing_works_OSDG.csv')
        if os.path.exists(enlarged_dataset_path):
            return pd.read_csv(enlarged_dataset_path, header=0, delimiter = ',', encoding='utf-8')

        # Create an empty dataframe to store the citing works
        related_works_df = pd.DataFrame(columns=['paperId', 'text', 'sdg', 'label'])
        # Iterate over the OSDG samples
        for index, osdg_sample in tqdm(osdg_data.iterrows(), total=osdg_data.shape[0], desc='Enlarging OSDG dataset'):

            # Get the related works for the OSDG sample
            related_works = self.get_related_works(osdg_sample)

            # Check if related works is an empty dataframe
            if related_works is None:
                continue
            
            # Add the related works to the dataframe
            related_works = self.build_related_work_sample(osdg_sample, related_works)
            related_works_df = pd.concat([related_works_df, related_works], ignore_index=True)

        return related_works_df

    def post_process_related_works(self, related_works):
        # Implement a function to post-process the related works dataframe
        # The function should remove duplicates and return the post-processed dataframe

        related_works = related_works.drop_duplicates(subset=['paperId'], keep='first')

        # Remove rows with empty text
        related_works = related_works[related_works['text'].str.strip() != '']

        # Remove any rows with NaN values

        related_works = related_works.dropna()

        # Remove any rows where the sdg column is not a float

        related_works = related_works[related_works['sdg'].apply(lambda x: isinstance(x, float))]

        return related_works

def main():
    osdg_dataset_path = 'data/OSDG/osdg_dataset.csv'
    responses_dict_path = 'data/OSDG/responses_dict.pkl'

    osdg_data_loader = OSDGDataLoader(osdg_dataset_path, responses_dict_path)

    related_works = osdg_data_loader.enlarge_osdg_dataset(osdg_data_loader.osdg_data)

    related_works = osdg_data_loader.post_process_related_works(related_works)

    # Save the enlarged dataset
    related_works.to_csv('data/OSDG/citing_works_OSDG.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
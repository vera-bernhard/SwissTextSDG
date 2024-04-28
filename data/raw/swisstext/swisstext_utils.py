import os
import sys
import re
import requests
import pickle
from tqdm import tqdm
import time

import pandas as pd


class SwissTextLoader:
    
    def __init__(self, swisstext_dataset_path, responses_dict_path):
        self.swisstext_dataset_path = swisstext_dataset_path
        self.swisstext_data = self.load_preprocessed_swisstext_data()
        
        self.responses_dict_path = responses_dict_path
        if os.path.exists(responses_dict_path):
            with open(responses_dict_path, 'rb') as file:
                self.responses_dict = pickle.load(file)
        else:
            self.responses_dict = {}

    def save_responses_dict(self):
        with open(self.responses_dict_path, 'wb') as file:
            pickle.dump(self.responses_dict, file)

    def load_preprocessed_swisstext_data(self):
        # Implement a function to load the preprocessed swisstext jsonl and return a Dataframe with the following columns:
        # - paperID (the paperID of the sample)
        # - text (the text of the sample)
        # - sdg (the corresponding SDG label)

        data = pd.read_json('data/preprocessed/task1_train_doi.jsonl', lines=True)

        # Remove columns 'ID' and 'URL'
        data = data.drop(columns=['ID', 'URL'])

        # Merge columns 'TITLE' and 'ABSTRACT' into 'text'
        data['text'] = data['TITLE'] + ' ' + data['ABSTRACT']
        data = data.drop(columns=['TITLE', 'ABSTRACT'])

        # Rename column 'SDG' to 'sdg'
        data = data.rename(columns={'SDG': 'sdg'})
        return data
    
    def get_semantic_scholar_citations_url(self, doi):
        # Implement a function to get the Semantic Scholar API url to retrieve the citations of a given doi
        # The function should return the url as a string.
        # We want to query the fields: contextsWithIntent, isInfluential, title and abstract
        url = f'https://api.semanticscholar.org/graph/v1/paper/{doi}/citations?fields=contextsWithIntent,isInfluential,title,abstract'
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
        elif response.status_code == 429:
            return '429'
        else:
            return None
    def call_semantic_scholar_citations_api(self, url):
        # Implement a function to call the Semantic Scholar API and return the response

        response = requests.get(url)
        return response
    
    def extract_doi_from_url(self, url):
        # Implement a function to extract the doi from a given url
        # The function should return the doi as a string

        # Extract the doi code by taking all characters after the pattern 10. (including it) until the end of the string
        doi = re.search(r'10\..*', url)
        if doi is not None:
            return doi.group(0)
        else:
            return ''
        
    
    def get_related_works(self, swisstext_sample):
        # Implement a function to retrieve the works that cite a given swisstext sample (i.e. a row from the swisstext dataset) using the Semantic Scholar Academic Graph API
        # The function should return a pandas dataframe with the following columns:

        # - paperID (the paperID of the citing work)
        # - text (text of the citing work, title and/or abstract)

        # Get the doi of the swisstext sample
        doi = swisstext_sample['paperID']
        if 'https://doi.org/' in doi:
            doi = self.extract_doi_from_url(doi)

        if doi == '':
            return None


        url = self.get_semantic_scholar_citations_url(doi)

        # Check if we have queried this url before

        if url in self.responses_dict:
            response = self.responses_dict[url]
            related_works = self.process_semantic_scholar_citation_response(response)
        else:
            # Call the API
            response = self.call_semantic_scholar_citations_api(url)
            time.sleep(1)
            related_works = self.process_semantic_scholar_citation_response(response)

            # Check if the response is a 429 status code
            if type(related_works) == str:
                while related_works == '429':
                    # Call the API
                    response = self.call_semantic_scholar_citations_api(url)
                    related_works = self.process_semantic_scholar_citation_response(response)

            # Add the response to the dictionary
            self.responses_dict[url] = response
            # Save the dictionary
            self.save_responses_dict()

        return related_works

    def build_related_work_sample(self, swisstext_sample, related_work):
        # Implement a function to build a related work sample using the swisstext sample and the related work
        # The function should return a pandas dataframe with the same columns as the swisstext dataset but with the related work as the only row

        related_work_sample = pd.DataFrame({'paperId': related_work['paperId'], 
                                                          'text': related_work['text'], 
                                                          'sdg': swisstext_sample['sdg']})
        
        return related_work_sample

    def enlarge_swisstext_dataset(self, swisstext_data):
        # Implement a function to enlarge the swisstext dataset by adding citing works for each swisstext sample using the get_related_works function
        # The function should return a pandas dataframe with the same columns as the swisstext dataset, but with the citing works as rows

        # Check if we have already enlarged the dataset
        enlarged_dataset_path = os.path.join('data', 'swisstext', 'citing_works_swisstext.csv')
        if os.path.exists(enlarged_dataset_path):
            return pd.read_csv(enlarged_dataset_path, header=0, delimiter = ',', encoding='utf-8')

        # Create an empty dataframe to store the citing works
        related_works_df = pd.DataFrame(columns=['paperId', 'text', 'sdg'])
        # Iterate over the swisstext samples
        for index, swisstext_sample in tqdm(swisstext_data.iterrows(), total=swisstext_data.shape[0], desc='Enlarging swisstext dataset'):

            # Get the related works for the swisstext sample
            related_works = self.get_related_works(swisstext_sample)

            # Check if related works is None
            if related_works is None:
                continue
            
            # Add the related works to the dataframe
            related_works = self.build_related_work_sample(swisstext_sample, related_works)
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

        # Remove line separators from the text
        related_works['text'] = related_works['text'].str.replace('\n', '')
        related_works['text'] = related_works['text'].str.replace('\r', '')
        related_works['text'] = related_works['text'].str.replace('\t', '')

        related_works['sdg'] = related_works['sdg'].astype(int)

        return related_works

def main():
    swisstext_dataset_path = 'data/preprocessed/task1_train_doi.jsonl'
    responses_dict_path = 'data/raw/swisstext/responses_dict.pkl'

    swisstext_loader = SwissTextLoader(swisstext_dataset_path, responses_dict_path)

    related_works = swisstext_loader.enlarge_swisstext_dataset(swisstext_loader.swisstext_data)

    related_works = swisstext_loader.post_process_related_works(related_works)

    related_works.to_csv('data/raw/swisstext/citing_works_swisstext.csv', index=False, sep=',')

if __name__ == '__main__':
    main()
import json
import os

import PyPDF2
import requests
from bs4 import BeautifulSoup

from helper import read_in_data


def get_pdf_from_zora_url(url: str, id: str) -> str:
    '''Get the html from a url.'''
    try:
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        # find the link in <a> with class ep_document_link
        pdf_link = soup.find('a', class_='ep_document_link')
        if pdf_link:
            pdf_url = pdf_link.get('href')
            # download the pdf
            response = requests.get(pdf_url)
            with open(f'data/pdf/{id}.pdf', 'wb') as file:
                file.write(response.content)
        else:
            # write the url to a file
            with open('data/pdf/missing_pdfs.txt', 'a') as file:
                file.write(f'{url}\n')
    
    except requests.exceptions.RequestException as e:
        print(e)
        return None

def extract_text_from_pdf(pdf: str) -> str:
    text = ""
    with open(pdf, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for pag_num in range(1, num_pages):
            page = reader.pages[pag_num]
            text += page.extract_text()
    return text

def text_mine_pdfs(pdf_folder: str) -> None:
    '''Extract the text from the pdfs in the pdf_folder.'''
    pdf_text_dict = {}
    for pdf in os.listdir(pdf_folder):
        if pdf.endswith('.pdf'):
            id = pdf.split('.')[0]
            pdf_path = os.path.join(pdf_folder, pdf)
            try:
                text = extract_text_from_pdf(pdf_path)
                pdf_text_dict[id] = text
            except Exception as e:
                # write the pdfs and error to a file
                with open('data/pdf/faulty_pdf_text_extract.txt', 'a') as file:
                    file.write(f'{pdf}: {e}\n')
            
    # write all text to a json file
    with open('data/pdf/pdf_text_raw.json', 'w', encoding='utf-8') as file:
        json.dump(pdf_text_dict, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # read in the data
    data = read_in_data('data/task1_train.jsonl')
    # get the pdfs
    for i, row in data.iterrows():
        id = row['ID'].lstrip('oai:www.zora.uzh.ch:')
        get_pdf_from_zora_url(row['URL'], id)

    # text_mine_pdfs('data/pdf')
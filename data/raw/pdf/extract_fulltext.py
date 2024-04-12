import json
import os

import PyPDF2
import requests
import layoutparser as lp
import numpy as np

from pdf2image import convert_from_path
from bs4 import BeautifulSoup
from helper import read_in_data
from settings import *
from selenium import webdriver
from selenium.webdriver.common.by import By


def get_pdf_from_zora_url(url: str, id: str) -> str:
    '''Get the html from a url.'''
    try:
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        # find <span class="download_div_oa_text">Closed Access</span>
        access = soup.find('span', class_='download_div_oa_text').text
        downloadable = ['Hybrid Open Access',
                        'Gold Open Access', 'Green Open Access']

        if access == 'Closed Access':
            no_pdf = soup.find('div', class_='download_meta').text
            if no_pdf == 'Full text not available from this repository.':
                # write the url to a file
                with open('pdf/missing_pdfs.txt', 'a') as file:
                    file.write(f'{url}\n')
            else:
                # write the url to a file
                with open('pdf/closed_pdfs.txt', 'a') as file:
                    file.write(f'{url}\n')

        elif access in downloadable:
            # find the link in <a> with class ep_document_link
            pdf_link = soup.find('a', class_='ep_document_link')
            if pdf_link:
                id = url.lstrip('oai:www.zora.uzh.ch:')
                pdf_url = pdf_link.get('href')
                # download the pdf
                get_pdf(pdf_url, id)
                

    except requests.exceptions.RequestException as e:
        print(e)
        return None

def get_pdf(pdf_url: str, id: str, ext='zora', session=None):
    if session:
        response = session.get(pdf_url)
    else:
        response = requests.get(pdf_url)
    # check if response 200
    if response.status_code != 200:
        print(f'Error: {response.status_code}')
        return None
    else:
        with open(f'pdf/{id}_{ext}.pdf', 'wb') as file:
            file.write(response.content)

def get_closed_pdf_from_zora_url(file: str) -> str:
    """ Little bot to open zora, login, prompt sso code, and download the pdfs."""
    # Create a session object
    session = requests.Session()
    
    # Login to zora
    driver = webdriver.Chrome()
    driver.get('https://www.zora.uzh.ch/cgi/users/home')
    driver.implicitly_wait(5)
    username = driver.find_element(By.ID, 'username')
    username.send_keys(EMAIL)
    login = driver.find_element(By.ID, 'login-button')
    login.click()
    driver.implicitly_wait(5)
    password = driver.find_element(By.ID,'password')
    password.send_keys(PASSWORD)
    login = driver.find_element(By.ID, 'login-button')
    login.click()
    driver.implicitly_wait(5)

    # Prompt the user for two factor authentication code
    otp = driver.find_element(By.ID, 'otp')
    # if opt exists
    if otp:
        # prompt the user to enter the otp
        code = input('Enter the code: ')
        otp.send_keys(code)
        # press login
        # <button type="submit" id="login-button" name="_eventId_submit" class="btn btn-primary btn-lg btn-block" style="margin-bottom: 15px;">Login</button>
        login = driver.find_element(By.ID, 'login-button')
        login.click()
        # wait for the page to load
        driver.implicitly_wait(5)

    # fetch all closed pdfs
    with open(file, 'r') as file:
        urls = file.readlines()
        for url in urls:
            url = url.strip()
            driver.get(url)
            # <a href="https://www.zora.uzh.ch/id/eprint/132070/1/1-s2.0-S0093691X16000686-main.pdf" class="download_logged_in btn btn-uzh-prime" title="Download PDF ">Download
            # <span class="ep_document_citation">PDF</span>
            pdf_link = driver.find_element(By.CLASS_NAME, 'download_logged_in').get_attribute('href')
            if pdf_link:
                id = url.split('/')[-1]
                # download the pdf
                
                if not session.cookies:
                    # Add selenium to the session
                    cookies = {c['name']: c['value'] for c in driver.get_cookies()}
                    session.cookies.update(cookies)
                
                get_pdf(pdf_link, id, 'closed', session=session)
            else:
                print(f'No pdf found for {url}')

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
                with open('pdf/faulty_pdf_text_extract.txt', 'a') as file:
                    file.write(f'{pdf}: {e}\n')

    # write all text to a json file
    with open('pdf/pdf_text_raw.json', 'w', encoding='utf-8') as file:
        json.dump(pdf_text_dict, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # read in the data
    data = read_in_data('task1_train.jsonl')
    # get the pdfs
    for i, row in data.iterrows():
        id = row['ID'].lstrip('oai:www.zora.uzh.ch:')
        get_pdf_from_zora_url(row['URL'], id)

    # # get the closed pdfs
    closed_pdf = 'pdf/closed_pdfs.txt'
    get_closed_pdf_from_zora_url(closed_pdf)
    
    # text_mine_pdfs('pdf')
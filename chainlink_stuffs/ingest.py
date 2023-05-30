# Required libraries
import os
import re
import bs4
import time
import pickle
import requests
import html2text
from tqdm import tqdm
import pandas as pd
import concurrent.futures
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import urljoin
from requests.exceptions import RequestException

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

from langchain.docstore.document import Document

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#settings for requests
MAX_THREADS = 10
REQUEST_DELAY = 0.1
SESSION = requests.Session()

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Set up the webdriver
s=Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s, options=chrome_options)

def filter_urls_by_base_url(urls, base_url):
    """
    Filters a list of URLs and returns only those that include the base_url.

    :param urls: List of URLs to filter.
    :param base_url: Base URL to filter by.
    :return: List of URLs that include the base_url.
    """
    return [url for url in urls if base_url in url]

def normalize_url(url):
    """
    Normalize a URL by ensuring it ends with '/'.

    :param url: URL to normalize.
    :return: Normalized URL.
    """
    return url if url.endswith('/') else url + '/'

def fetch_url_request(url):
    """
    Fetches the content of a URL using requests library and returns the response.
    In case of any exception during fetching, logs the error and returns None.

    :param url: URL to fetch.
    :return: Response object on successful fetch, None otherwise.
    """
    try:
        response = SESSION.get(url)
        response.raise_for_status()
        return response
    except RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def fetch_url_selenium(url):
    """
    Fetches the content of a URL using Selenium and returns the source HTML of the page.
    In case of any exception during fetching, logs the error and returns None.

    :param url: URL to fetch.
    :return: HTML source as a string on successful fetch, None otherwise.
    """
    try:
        driver.get(url)
        driver.implicitly_wait(7)
        time.sleep(7)
        return driver.page_source
    
    except RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def process_url(response, visited, base_url):
    """
    Process a URL response. Extract all absolute URLs from the response that 
    haven't been visited yet and belong to the same base_url.

    :param response: Response object from a URL fetch.
    :param visited: Set of URLs already visited.
    :param base_url: Base URL to filter by.
    :return: Set of new URLs to visit.
    """
    urls = set()
    if response:
        soup = BeautifulSoup(response.content, 'html.parser')
        for link in soup.find_all('a'):
            href = link.get('href')
            if href is not None and '#' not in href:
                absolute_url = normalize_url(urljoin(response.url, href))
                if absolute_url not in visited and base_url in absolute_url:
                    visited.add(absolute_url)
                    urls.add(absolute_url)
    return urls

def get_all_suburls(url, visited=None):
    """
    Get all sub-URLs of a given URL that belong to the same domain.

    :param url: Base URL to start the search.
    :param visited: Set of URLs already visited.
    :return: Set of all sub-URLs.
    """
    if visited is None:
        visited = set()

    if not url.startswith("http"):
        url = "https://" + url

    base_url = url.split("//")[1].split("/")[0]
    urls = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_responses = [executor.submit(fetch_url_request, url)]
        while future_responses:
            for future in concurrent.futures.as_completed(future_responses):
                future_responses.remove(future)
                response = future.result()
                new_urls = process_url(response, visited, base_url)
                urls.update(new_urls)
                if len(future_responses) < MAX_THREADS:
                    for new_url in new_urls:
                        future_responses.append(executor.submit(fetch_url_request, new_url))

    urls = filter_urls_by_base_url(urls, base_url)
    return urls

def process_tag(tag):
    """
    Process an HTML tag. If the tag is a table, convert it to Markdown. 
    Otherwise, convert it to Markdown as-is.

    :param tag: HTML tag to process.
    :return: Markdown representation of the tag.
    """
    if tag.name == 'table':
        # Convert the table to a DataFrame
        df = pd.read_html(str(tag))[0]
        
        # Convert the DataFrame to Markdown
        return df.to_markdown(index=False) + '\n'
    else:
        # If it's not a table, convert it to Markdown as before
        html = str(tag)
        return html2text.html2text(html)

def fix_markdown_links(markdown_text):
    """
    Fix Markdown links by removing any spaces in the URL.

    :param markdown_text: Markdown text to process.
    :return: Fixed Markdown text.
    """
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\s+([^)]+)\)', r'[\1](\2\3)', markdown_text)

def process_nested_tags(tag):
    """
    Process nested HTML tags. Convert tags to Markdown recursively.

    :param tag: Root HTML tag to process.
    :return: Markdown text
    """
    if tag.name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'table', 'ol', 'ul'}:
        return process_tag(tag)
    else:
        markdown_parts = []
        for child in tag.children:
            if isinstance(child, bs4.element.Tag):
                markdown_parts.append(process_nested_tags(child))
        return ''.join(markdown_parts)

# def parse(url, use_selenium:list=[]):
#     if url in use_selenium:
#         response = fetch_url_selenium(url)  
#         if response:
#             soup = BeautifulSoup(response, 'html.parser')
#         else:
#             soup = None
#     else:
#         response = fetch_url_request(url)
#         if response:
#             soup = BeautifulSoup(response.content, 'html.parser')
#         else:
#             soup = None
#     if soup:
#         return parse_from_soup(soup)

def parse(url):
    """
    Fetches and parses a URL using Selenium and BeautifulSoup. 
    Extracts the useful information from the HTML and returns it.

    :param url: URL to fetch and parse.
    :return: Processed content from the URL if it exists, None otherwise.
    """
    # Fetch the page with Selenium
    html = fetch_url_selenium(url)
    
    if html:
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
    
    # Continue processing the page as before
    if soup:
        return parse_from_soup(soup)


def parse_from_soup(soup):
    """
    Parses the soup object from BeautifulSoup, removes unnecessary tags,
    and returns the content in markdown format.

    :param soup: BeautifulSoup object
    :return: Content from the soup in markdown format.
    """
    grid_main = soup.find('div', {'id': 'grid-main'})

    if grid_main:
        for img in grid_main.find_all('img'):
            img.decompose()

        for h2 in grid_main.find_all('h2', {'class': 'heading'}):
            h2.decompose()

        markdown_content = process_nested_tags(grid_main)
        fixed_markdown_content = fix_markdown_links(markdown_content)
        return fixed_markdown_content
    else:
        logger.error('Failed to find the "grid-main" div.')

def remove_duplicates(doc_list):
    """
    Removes duplicate documents from a list of Documents based on page_content.

    :param doc_list: List of Document objects.
    :return: List of unique Document objects.
    """
    content_to_doc = {}
    for doc in doc_list:
        if doc.page_content not in content_to_doc:
            content_to_doc[doc.page_content] = doc
    return list(content_to_doc.values())

def insert_full_url(text):
    """
    Inserts the full URL into Markdown links in the text.

    :param text: Text to process.
    :return: Text with full URLs in Markdown links.
    """
    base_url = 'https://docs.chain.link'
    def replacer(match):
        sub_url = match.group(2)
        # If the sub_url is an absolute URL, return it unchanged
        if sub_url.startswith('http://') or sub_url.startswith('https://'):
            return match.group(0)
        # If the sub_url starts with a slash, remove it to avoid double slashes in the final url
        if sub_url.startswith('/'):
            sub_url = sub_url[1:]
        return f'[{match.group(1)}]({base_url}/{sub_url})'

    return re.sub(r'\[(.*?)\]\((.*?)\)', replacer, text)

def refine_docs(docs:List[Document]):
    """
    Removes duplicates and inserts full URLs into the page_content of the Document objects.

    :param docs: List of Document objects.
    :return: Refined list of Document objects.
    """
    docs_filtered = remove_duplicates(docs)
    base_url = 'https://docs.chain.link'
    for doc in docs_filtered:
        doc.page_content = insert_full_url(base_url, doc.page_content)
    return docs_filtered

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    all_urls = get_all_suburls("https://docs.chain.link/")
    all_urls = sorted(list(set(all_urls)))

    documents = []
    for url in tqdm(all_urls):
        data = parse(url)
        if data:
            documents.append(Document(page_content=data, metadata={'source': url}))

    documents = remove_duplicates(documents)

    with open('chainlink_docs.pkl', 'wb') as f:
        pickle.dump(documents, f)
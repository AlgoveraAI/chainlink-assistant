import re
import time
import pickle
import requests
from tqdm import tqdm
from datetime import datetime
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

from langchain.docstore.document import Document

from config import get_logger

logger = get_logger(__name__)

# Settings for requests
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


def filter_links(soup, filter_str='/polygon/mainnet/'):
    # Get all links
    links = soup.find_all('a')
    
    # Filter links to only those that are for polygon mainnet
    hrefs = [link.get('href') for link in links]
    filtered_hrefs = [href for href in hrefs if href is not None and filter_str in href and href.count('/') == 4]

    return filtered_hrefs


def get_links(url):
    """
    Get all links from a given url
    """

    filter_sub_url = url.split("link")[1]

    all_links = []
    for i in range(10):
        if i == 0:
            driver.get(url)
            driver.implicitly_wait(7)
            time.sleep(7)

        else:
            driver.find_element(by="xpath", value="/html/body/div[1]/main/section[2]/div/div[2]/button[2]").click()
            driver.implicitly_wait(7)
            time.sleep(7)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        hrefs = filter_links(soup, filter_sub_url)
        all_links.extend(hrefs)
    
    # Add base url
    all_links = [f"https://data.chain.link{link}" for link in all_links]
    
    # Remove duplicates
    all_links = list(set(all_links))

    return all_links


def get_details(url):
    driver.get(url)
    driver.implicitly_wait(3)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    details = {}

    # Get the title
    details["pair"] = soup.find("h1").text

    # Get the details
    infos = soup.findAll("p")

    # Match pattern
    match = r'Minimum of (\d+)'

    prev_word = ""
    for info in infos:
        if prev_word == "Asset Name":
            details["asset_name"] = info.text
        elif prev_word == "Asset Class":
            details["asset_class"] = info.text
        elif prev_word == "Tier":
            details["tier"] = info.text
        elif prev_word == "Network":
            details["network"] = info.text
        elif prev_word == "Deviation threshold":
            details["deviation"] = info.text
        if re.search(match, prev_word):
            details["num_oracles"] = info.text
        prev_word = info.text

        try:
            for each in soup.find("div", class_="sc-d6e7e954-0 sc-e3a5e58-0 teTjm"):
                if each.name != "div":
                    details["contract_address"] = each.a.text
        except:
            pass

        try:
            for each in soup.find("div", class_="sc-d6e7e954-0 sc-3ba96657-0 sc-b8182c9f-1 hRpMsk iSLEhf"):
                    details["ens_address"] = each.div.next_sibling.text
        except:
            pass

    return details

BASE_URLS = [
    "https://data.chain.link/ethereum/mainnet",
    "https://data.chain.link/polygon/mainnet",
    "https://data.chain.link/optimism/mainnet",
    "https://data.chain.link/fantom/mainnet",
    "https://data.chain.link/moonriver/mainnet",
    "https://data.chain.link/metis/mainnet",
    "https://data.chain.link/bsc/mainnet",
    "https://data.chain.link/arbitrum/mainnet",
    "https://data.chain.link/avalanche/mainnet",
    "https://data.chain.link/harmony/mainnet",
    "https://data.chain.link/moonbeam/mainnet",
]


def make_sentence(details):
    """Make a sentence from the details"""

    first_sentence = """The following is the details for the pair {pair} which operates on the {network}."""
    second_sentence = """This asset is named "{asset_name}".""" 
    third_sentence = """and falls under the "{asset_class}" asset class."""
    fourth_sentence = """It has a tier status of "{tier}".""" 
    fifth_sentence = """The deviation threshold for this asset is set at {deviation}.""" 
    sixth_sentence = """{num_oracles} oracles carries and support this asset.""" 
    seventh_sentence = """You can find its contract at the address "{contract_address}"""
    eigth_sentence = """, and its ENS address is "{ens_address}"."""

    if "network" in details.keys() and "pair" in details.keys():
        sentence = first_sentence.format(pair=details["pair"], network=details["network"])

        if "asset_name" in details.keys():
            sentence += f" {second_sentence.format(asset_name=details['asset_name'])}"

        if "asset_class" in details.keys():
            sentence += f" {third_sentence.format(asset_class=details['asset_class'])}"

        if "tier" in details.keys():
            sentence += f" {fourth_sentence.format(tier=details['tier'])}"

        if "deviation" in details.keys():
            sentence += f" {fifth_sentence.format(deviation=details['deviation'])}"

        if "num_oracles" in details.keys():
            sentence += f" {sixth_sentence.format(num_oracles=details['num_oracles'])}"

        if "contract_address" in details.keys():
            sentence += f" {seventh_sentence.format(contract_address=details['contract_address'])}"

        if "ens_address" in details.keys():
            sentence += f" {eigth_sentence.format(ens_address=details['ens_address'])}"

    else:
        logger.error(f"Missing keys in details: {details.keys()}")

    return sentence

def scrap_data():
    """Scrap data from the website and put into a Document"""

    documents = []
    for base_url in tqdm(BASE_URLS, total=len(BASE_URLS)):
        logger.info(f"Scraping {base_url}")
        all_links  = get_links(base_url)
        logger.info(f"Total links: {len(all_links)}")

        all_details = []

        for u in tqdm(all_links, total=len(all_links)):
            try:
                all_details.append(get_details(u))
            except Exception as e:
                logger.error(f'Failed to get details for {u}')
                logger.error(e)

        # Make a sentence for each details
        all_sentences = [make_sentence(details) for details in all_details]

        # Maken sentences in a continuous string
        all_sentences = "\n\n".join(all_sentences)

        # put into a Document
        doc = Document(page_content=all_sentences, metadata={"source": base_url})
        documents.append(doc)

        logger.info(f"Scraping {base_url} done")

    # Save the documents
    with open(f"./data/datadocs_{datetime.now().strftime('%Y-%m-%d')}.pkl", 'wb') as f:
        pickle.dump(documents, f)
    
    return documents
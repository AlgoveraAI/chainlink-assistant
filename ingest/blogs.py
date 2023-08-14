import time
import pickle
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from langchain.docstore.document import Document

from config import get_logger

logger = get_logger(__name__)


# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Set up the webdriver
s=Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s, options=chrome_options)

def get_blog_urls():
    urls = set()
    try:
        for i in range(200):
            if i == 0:
                driver.get("https://blog.chain.link/?s=&categories=&services=&tags=&sortby=newest")
                driver.implicitly_wait(3)
                time.sleep(3)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                blogs = [post.a["href"] for post in soup.findAll('div', class_='post-card')]
                urls |= set(blogs)
            else:
                element = driver.find_element(By.XPATH, "/html/body/div[1]/div/section/div/div/div[3]/div[2]/a")
                element.click()
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                blogs = [post.a["href"] for post in soup.findAll('div', class_='post-card')]
                urls |= set(blogs)
            if i % 10 == 0:
                logger.info(f"Scraped {len(urls)} blog urls")
    except Exception as e:
        logger.error(f"Error scraping blog urls: {e}")
        pass

    return urls


def to_markdown(pair):
    url, soup = pair
    output = ""
    try:
        try:
            sub_soup = soup.find("h1", id="post-title")
            heading_level = int(sub_soup.name[1:])
            output += f"{'#' * heading_level} {sub_soup.get_text()}\n\n"
        except:
            sub_soup = soup.find("h1")
            heading_level = int(sub_soup.name[1:])
            output += f"{'#' * heading_level} {sub_soup.get_text()}\n\n"

        
        sub_soup_2 = soup.find("div", class_="post-header")
        if not sub_soup_2:
            sub_soup_2 = soup.find("article", class_="educational-content")
            
        for element in sub_soup_2.find_all([
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'li', 'blockquote', 'code', 'pre',
            'em', 'strong', 'ol', 'dl', 'dt', 'dd', 'hr', 'table', 'thead', 'tbody', 'tr',
            'th', 'td', 'sup', 'sub', 'abbr'
        ]):
            if element.name == 'p':
                output += f"{element.get_text()}\n\n"
            elif element.name.startswith('h'):
                try:
                    heading_level = int(element.name[1:])
                    output += f"{'#' * heading_level} {element.get_text()}\n\n"
                except:
                    pass
            elif element.name == 'ul':
                for li in element.find_all('li'):
                    output += f"- {li.get_text()}\n"
                output += '\n'
            elif element.name == 'li':
                output += f"- {element.get_text()}\n"
            elif element.name == 'blockquote':
                output += f"> {element.get_text()}\n\n"
            elif element.name == 'code':
                output += f"`{element.get_text()}`"
            elif element.name == 'pre':
                output += f"```\n{element.get_text()}\n```\n\n"
            elif element.name == 'em':
                output += f"*{element.get_text()}*"
            elif element.name == 'strong':
                output += f"**{element.get_text()}**"
            elif element.name == 'ol':
                for li in element.find_all('li'):
                    output += f"1. {li.get_text()}\n"
                output += '\n'
            elif element.name == 'dl':
                for dt, dd in zip(element.find_all('dt'), element.find_all('dd')):
                    output += f"{dt.get_text()}:\n{dd.get_text()}\n"
                output += '\n'
            elif element.name == 'hr':
                output += '---\n\n'
            elif element.name == 'table':
                table_text = element.get_text(separator='|', strip=True)
                output += f"{table_text}\n\n"
            elif element.name == 'thead':
                output += f"{element.get_text()}\n"
            elif element.name in ['tbody', 'tr', 'th', 'td']:
                pass  # Ignore these elements
            elif element.name == 'sup':
                output += f"<sup>{element.get_text()}</sup>"
            elif element.name == 'sub':
                output += f"<sub>{element.get_text()}</sub>"
            elif element.name == 'abbr':
                output += f"<abbr title='{element.get('title', '')}'>{element.get_text()}</abbr>"
                
        return (url, output)
    
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return (url, "")


def scrape_blogs():
    urls = get_blog_urls()

    soups = []
    unsuccessful_urls = []
    for url in tqdm(urls, total=len(urls)):
        try:
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            soups.append((url, soup))
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            unsuccessful_urls.append(url)

    blogs = []
    provessed_urls = []
    for (url, soup) in tqdm(soups, total=len(soups)):
        if url in provessed_urls:
            continue
        markdown = to_markdown((url, soup))
        blogs.append(markdown)
        provessed_urls.append(url)

    blogs_documents = []
    for url, markdown in blogs:
        blogs_documents.append(Document(page_content=markdown, metadata={"source": url, "type": "blog"}))

    # Make sure the output directory exists
    Path("./data").mkdir(parents=True, exist_ok=True)

    # Save the documents to a pickle file with date in the name
    with open(f"./data/blog_{datetime.now().strftime('%Y-%m-%d')}.pkl", 'wb') as f:
        pickle.dump(blogs_documents, f)
    
    logger.info(f"Scraped blog posts")

    return blogs_documents

        
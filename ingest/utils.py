import re
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def get_description_chain():
    system_template = """
    Please summarize the context below in one sentence (no more than 15 words). This will be used as the description of the article in the search results.

    Response should be NO MORE THAN 15 words.
    """

    human_template = """{context}"""

    PROMPT = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
    )

    llm = ChatOpenAI(temperature=0.) #, request_timeout=120)
    chain = LLMChain(llm=llm, prompt=PROMPT)

    return chain


def remove_prefix_text(markdown):
    # Split the content at the first title
    parts = re.split(r'^(#\s.+)$', markdown, maxsplit=1, flags=re.MULTILINE)

    # If a split occurred, then take the content from the first title onward
    new_text = parts[-2] + parts[-1] if len(parts) > 1 else markdown

    return new_text


def extract_first_n_paragraphs(content, num_para=2):

    # Split by two newline characters to denote paragraphs
    paragraphs = content.split('\n\n')
    
    # Return the first num_para paragraphs or whatever is available
    return '\n\n'.join(paragraphs[:num_para])


def get_driver():
    # Path to your chromedriver (change this to your path)
    CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver'

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Multiple potential paths for Chrome binary
    CHROME_PATHS = [
        "/opt/google/chrome/chrome-linux64/chrome",
        "/opt/google/chrome/chrome/chrome"
    ]

    # Set the binary location to the first existing path
    for path in CHROME_PATHS:
        if os.path.exists(path):
            chrome_options.binary_location = path
            break

    # Check if chromedriver exists at the specified path
    if not os.path.exists(CHROMEDRIVER_PATH):
        try:
            CHROMEDRIVER_PATH = ChromeDriverManager().install()
        except:
            CHROMEDRIVER_PATH = '/opt/homebrew/bin/chromedriver'

    # Set up the webdriver using the determined path
    s = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(options=chrome_options, service=s)

    return driver
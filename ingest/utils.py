import re
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

    llm = ChatOpenAI(temperature=0.)
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
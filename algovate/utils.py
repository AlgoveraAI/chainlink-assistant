import yaml
import logging
from itertools import product
from typing import List, Union
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.retrievers import SVMRetriever, TFIDFRetriever, KNNRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, CharacterTextSplitter, TokenTextSplitter

from .prompts import QA_CHAIN_PROMPT

SPLITTER_MAPPING = {"RecursiveCharacterTextSplitter":RecursiveCharacterTextSplitter, 
                    "MarkdownTextSplitter":MarkdownTextSplitter, 
                    "CharacterTextSplitter":CharacterTextSplitter, 
                    "TokenTextSplitter":TokenTextSplitter}

def get_logger(name, file_name, level=logging.INFO):
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level) # Set the threshold for this logger
    
    # Create handlers
    c_handler = logging.StreamHandler() # Console handler
    f_handler = logging.FileHandler(file_name) # File handler
    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def generate_experiment_configs_from_file(parameter_file):
    with open(parameter_file, 'r') as f:
        params = yaml.safe_load(f)

    base_config = {
        'paths': params['paths'][0],  # Use the first (and presumably only) paths dictionary
        'experiments': []
    }

    # Get all combinations of parameters excluding k and score_threshold
    combinations = product(
        params['splitter'], 
        params['chunk_size'], 
        params['chunk_overlap'], 
        params['search_type'], 
        params['model_name'], 
        params['chain_type'], 
        params['grading_type']
    )

    experiment_counter = 1  # counter for experiment names

    for combination in combinations:
        # If search_type is "similarity", add "k" and "score_threshold"
        if combination[3] == "similarity":
            k_and_score_combinations = product(params['k'], params['score_threshold'])
            for k, score in k_and_score_combinations:
                experiment = {
                    'experiment': {
                        'name': f'experiment_{experiment_counter}',
                        'splitter': combination[0],
                        'chunk_size': combination[1],
                        'chunk_overlap': combination[2],
                        'search_type': combination[3],
                        'model_name': combination[4],
                        'chain_type': combination[5],
                        'grading_type': combination[6],
                        'k': k,
                        'score_threshold': score,
                    }
                }
                base_config['experiments'].append(experiment)
                experiment_counter += 1
        else:
            experiment = {
                'experiment': {
                    'name': f'experiment_{experiment_counter}',
                    'splitter': combination[0],
                    'chunk_size': combination[1],
                    'chunk_overlap': combination[2],
                    'search_type': combination[3],
                    'model_name': combination[4],
                    'chain_type': combination[5],
                    'grading_type': combination[6],
                }
            }
            base_config['experiments'].append(experiment)
            experiment_counter += 1
    
    return base_config

def prepare_splitter(
    documents: List[Document],
    splitter:Union[RecursiveCharacterTextSplitter, MarkdownTextSplitter, CharacterTextSplitter, TokenTextSplitter],
    chunk_size:int,
    chunk_overlap:int
):
    if splitter == RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter == MarkdownTextSplitter:
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter == CharacterTextSplitter:
        splitter = CharacterTextSplitter(separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter == TokenTextSplitter:
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    splits = splitter.split_documents(documents)

    return splits

def prepare_faiss_vectorstore(
    chunked_docs, 
):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    return vectorstore

def prepare_retriever_ss(
    vectorstore, 
    search_type="similarity", # similarity, mmr
    k=4, 
    score_threshold=0.5, # only for similarity
):
    if search_type == "mmr":
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs ={"k":k})
    else:
        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                             search_kwargs ={"k":k, "score_threshold":score_threshold})

    return retriever

def prepare_retriever(
    documents: List[Document], 
    splitter:Union[RecursiveCharacterTextSplitter, MarkdownTextSplitter, CharacterTextSplitter, TokenTextSplitter],
    chunk_size:int,
    chunk_overlap:int,
    search_type="similarity", # similarity, mmr, svm, knn, tfidf 
    k=4, # only for similarity, mmr
    score_threshold=0.5,
):
    chunked_docs = prepare_splitter(documents, splitter=splitter, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    if search_type in ["similarity", "mmr"]:
        vectorstore = prepare_faiss_vectorstore(chunked_docs)
        retriever = prepare_retriever_ss(vectorstore, search_type=search_type, k=k, score_threshold=score_threshold)
    
    else:
        texts = [doc.page_content for doc in chunked_docs]
        if search_type == "tfidf":
            retriever = TFIDFRetriever.from_texts(texts)
        elif search_type == "svm":
            retriever = SVMRetriever.from_texts(texts, OpenAIEmbeddings())
        elif search_type == "knn":
            retriever = KNNRetriever.from_texts(texts, OpenAIEmbeddings())

    return retriever

def prepare_llm(
    model_name= "gpt-3.5-turbo", #"gpt-3.5-turbo", "gpt-4"
):
    return ChatOpenAI(model_name=model_name, temperature=0)

def prepare_chain(
    llm, 
    retriever,
    chain_type="stuff", # stuff, refine, map_reduce
):
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
    qa_chain = RetrievalQA.from_chain_type(llm,
                                            chain_type=chain_type,
                                            retriever=retriever,
                                            chain_type_kwargs=chain_type_kwargs,
                                            input_key="question")
    return qa_chain
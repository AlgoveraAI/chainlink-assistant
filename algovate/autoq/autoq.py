import pickle
import random
import logging
from typing import List, Union
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain

class AutoQ:
    def __init__(self, 
        document_store:Union[str, List[Document]]
    ):
        '''
        Args:
            document_store: Path to a pickled document store or a list of documents. Documents shouldn't be chunked
        '''
        if isinstance(document_store, str):
            # make sure it's a pickle file
            if not document_store.endswith(".pkl"):
                raise ValueError("document_store should be a path to a pickled document store")
            with open(document_store, "rb") as f:
                self.document_store = pickle.load(f)

        elif isinstance(document_store, List):
            self.document_store = document_store

        self.chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))

        # pairs
        self.pairs = []

        # Basic logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("AutoQ initialized")
        self.logger.info(f"Document store has {(self.document_store)}")


    def generate_single_pair(self, 
        chunk,
        metadata
    ):
        '''
        Args:
            chunk: A chunk of text from a document
        '''
        # generate question
        qa = self.chain.run(chunk)
        qa[0]["metadata"] = metadata
        self.pairs.append(qa)

    def generate_eval_pairs(self, 
        num_pairs:int=10, 
        chunk_size:int=3000
    ):
        self.logger.info(f"Generating {num_pairs} pairs")
        for i in range(num_pairs):
            # randomly sample a chunk
            sample_int = random.randint(0, len(self.document_store)-1)
            sample_chunk = self.document_store[sample_int].page_content
            metadata = self.document_store[sample_int].metadata

            num_of_chars = len(sample_chunk)
            
            if num_of_chars > chunk_size:
                starting_index = random.randint(0, num_of_chars-chunk_size)
                chunk = sample_chunk[starting_index:starting_index+chunk_size]
            else:
                chunk = sample_chunk

            self.generate_single_pair(chunk=chunk, metadata=metadata)

            # log progress
            self.logger.info(f"Generated {len(self.pairs)} of {num_pairs} pairs")

            



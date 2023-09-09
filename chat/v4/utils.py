import pickle
import re
import tiktoken
from pydantic import BaseModel
from typing import Any, Dict, List
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.schema import BaseRetriever
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter

from config import firebase
from chainlink.prompts_mem import FINAL_ANSWER_PROMPT
from utils import createLogHandler, StreamingLLMCallbackHandler

logger = createLogHandler(__name__, "logs.log")


class CustomeSplitter:
    def __init__(self, chunk_threshold=6000, chunk_size=6000, chunk_overlap=50):
        self.chunk_threshold = chunk_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def token_counter(self, document):
        tokens = self.enc.encode(document.page_content)
        return len(tokens)

    def split(self, documents):
        chunked_documents = []
        for i, doc in enumerate(documents):
            try:
                if self.token_counter(doc) > self.chunk_threshold:
                    chunks = self.splitter.split_documents([doc])
                    chunks = [
                        Document(
                            page_content=chunk.page_content,
                            metadata={
                                "source": f"{chunk.metadata['source']} chunk {i}"
                            },
                        )
                        for i, chunk in enumerate(chunks)
                    ]
                    chunked_documents.extend(chunks)
                else:
                    chunked_documents.append(doc)
            except Exception as e:
                chunked_documents.append(doc)
                print(f"Error on document {i}")
                print(e)
                print(doc.metadata["source"])

        return chunked_documents


class CustomRetriever(BaseRetriever, BaseModel):
    full_docs: List[Document]
    base_retriever: BaseRetriever = None
    k_final: int = 4

    logger: Any = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        full_docs: List[Document],
        search_kwargs: Dict[str, Any] = {},
        k_initial: int = 10,
        k_final: int = 4,
        logger: Any = None,
        **kwargs: Any,
    ):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
        split_docs = splitter.split_documents(full_docs)
        vector_store = FAISS.from_documents(split_docs, embedding=OpenAIEmbeddings())

        return cls(
            full_docs=full_docs,
            base_retriever=vector_store.as_retriever(search_kwargs={"k": k_initial}),
            logger=logger,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.base_retriever.get_relevant_documents(query=query)
        self.logger.info(f"Retrieved {len(results)} documents")
        doc_ids = [doc.metadata["source"] for doc in results]

        # make it a set but keep the order
        doc_ids = list(dict.fromkeys(doc_ids))[: self.k_final]

        # log to the logger
        self.logger.info(f"Retrieved {len(doc_ids)} unique documents")

        # get upto 4 documents
        full_retrieved_docs = [
            d for d in self.full_docs if d.metadata["source"] in doc_ids
        ]
        return full_retrieved_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError


def prepare_single_document(document):
    template = """Document: {document}\nSource: {source}
    """
    return template.format(
        document=document.page_content, source=document.metadata["source"]
    )


def prepare_multiple_documents(all_answers):
    # loop through and consider only if the document doesnt start with "no"
    answer_to_use = []
    for answer in all_answers:
        if not answer["answer"].lower().startswith("no"):
            answer_to_use.append(answer)

    if len(answer_to_use) == 0:
        return "no answer"

    return "\n\n".join(["Content: " + d["answer"] for d in answer_to_use])


def get_retriever_chain():
    # Get vector store
    assistant_info = firebase.get_assistant("org_chainlink")
    logger.info(f"assistant_info: {assistant_info}")
    folder = assistant_info["folder"]

    try:
        with open(f"{folder}/documents.pkl", "rb") as f:
            documents = pickle.load(f)
        logger.info(f"documents: {documents[0]}")

    except Exception as e:
        with open(
            "/home/marshath/play/chainlink/algovate/algovate/data/combined_documents.pkl",
            "rb",
        ) as f:
            documents = pickle.load(f)
        logger.info(f"Error: {e}")

    splitter = CustomeSplitter()
    chunked_documents = splitter.split(documents)

    retriever = CustomRetriever.from_documents(
        chunked_documents, k_initial=10, k_final=4, logger=logger
    )

    # Get chain
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0.0,
    )
    chain = LLMChain(llm=llm, prompt=FINAL_ANSWER_PROMPT)

    return retriever, chain


def get_streaming_chain(manager, chain):
    """Return a new streaming chain."""
    stream_handler = StreamingLLMCallbackHandler(manager)
    # stream_manager =  AsyncCallbackManager([stream_handler])

    llm_stream = ChatOpenAI(
        temperature=0.0,
        model="gpt-3.5-turbo-16k",
        streaming=True,
        callbacks=[stream_handler],
    )

    chain.llm = llm_stream

    return chain


retriever, chain = get_retriever_chain()

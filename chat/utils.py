import os
import faiss
import pickle
import tiktoken
from pydantic import BaseModel
from typing import Any, Dict, List
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.schema import BaseRetriever
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter

from chat.prompts_mem import FINAL_ANSWER_PROMPT
from utils import createLogHandler, StreamingLLMCallbackHandler
from search.search import SearchRetriever
from config import ROOT_DIR

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
    base_retriever_all: BaseRetriever = None
    base_retriever_data: BaseRetriever = None
    k_initial: int = 10
    k_final: int = 4

    logger: Any = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        full_docs: List[Document],
        vectorstore_all: FAISS,
        vectorstore_data: FAISS,
        search_kwargs: Dict[str, Any] = {},
        k_initial: int = 10,
        k_final: int = 4,
        logger: Any = None,
        **kwargs: Any,
    ):
        # splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
        # split_docs = splitter.split_documents(full_docs)
        # vector_store = FAISS.from_documents(split_docs, embedding=OpenAIEmbeddings())

        return cls(
            full_docs=full_docs,
            base_retriever_all=vectorstore_all.as_retriever(
                search_kwargs={"k": k_initial}
            ),
            base_retriever_data=vectorstore_data.as_retriever(
                search_kwargs={"k": k_initial}
            ),
            logger=logger,
        )

    def get_relevant_documents(self, query: str, workflow: int = 1) -> List[Document]:
        self.logger.info(f"Worflow: {workflow}")

        if workflow == 2:
            results = self.base_retriever_data.get_relevant_documents(query=query)
            self.logger.info(f"Retrieved {len(results)} documents")
            return results[: self.k_final]

        else:
            results = self.base_retriever_all.get_relevant_documents(query=query)
            self.logger.info(f"Retrieved {len(results)} documents")
            if workflow == 1:
                doc_ids = [doc.metadata["source"] for doc in results]

                # make it a set but keep the order
                doc_ids = list(dict.fromkeys(doc_ids))[: self.k_final]

                # log to the logger
                self.logger.info(f"Retrieved {len(doc_ids)} unique documents")

                # get upto 4 documents
                full_retrieved_docs = [
                    d for d in self.full_docs if d.metadata["source"] in doc_ids
                ]

                return self.prepare_source(full_retrieved_docs)

            full_retrieved_docs = results[: self.k_final]
            return self.prepare_source(full_retrieved_docs)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

    def prepare_source(self, documents: List[Document]) -> List[Document]:

        for doc in documents:
            source = doc.metadata["source"]
            if "chunk" in source:
                source = source.split("chunk")[0].strip()
                doc.metadata["source"] = source

        return documents


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

    folder = f"{ROOT_DIR}/data"

    # Open faiss index all
    index_all = faiss.read_index(f"{folder}/docs_all.index")

    # Open faiss vector store
    with open(f"{folder}/faiss_store_all.pkl", "rb") as f:
        vectorstore_all = pickle.load(f)

    # Put back index
    vectorstore_all.index = index_all

    # Open faiss index data
    index_data = faiss.read_index(f"{folder}/docs_data.index")

    # Open faiss vector store
    with open(f"{folder}/faiss_store_data.pkl", "rb") as f:
        vectorstore_data = pickle.load(f)

    # Put back index
    vectorstore_data.index = index_data

    # Open documents
    with open(f"{folder}/documents.pkl", "rb") as f:
        documents = pickle.load(f)

    splitter = CustomeSplitter()
    chunked_documents = splitter.split(documents)

    retriever = CustomRetriever.from_documents(
        chunked_documents,
        vectorstore_all=vectorstore_all,
        vectorstore_data=vectorstore_data,
        k_initial=10,
        k_final=4,
        logger=logger,
    )

    # Get chain
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
    )
    chain = LLMChain(llm=llm, prompt=FINAL_ANSWER_PROMPT)

    return retriever, chain


def get_streaming_chain(manager, chain, workflow):
    """Return a new streaming chain."""
    stream_handler = StreamingLLMCallbackHandler(manager)
    # stream_manager =  AsyncCallbackManager([stream_handler])

    if workflow == 1:
        llm_stream = ChatOpenAI(
            temperature=0.0,
            model="gpt-3.5-turbo-16k",
            streaming=True,
            callbacks=[stream_handler],
        )
        logger.info("Using long-form workflow")
        chain.llm = llm_stream
    else:
        llm_stream = ChatOpenAI(
            temperature=0.0,
            model="gpt-3.5-turbo",
            streaming=True,
            # max_tokens=256,
            callbacks=[stream_handler],
        )
        chain.llm = llm_stream
        logger.info("Using short-form workflow")

    return chain


def get_search_retriever():
    folder = f"{ROOT_DIR}/data"
    # Open blogs document
    with open(f"{folder}/blog_documents.pkl", "rb") as f:
        blog_documents = pickle.load(f)

    # Open technical documents
    with open(f"{folder}/tech_documents.pkl", "rb") as f:
        technical_documents = pickle.load(f)

    # data documents
    with open(f"{folder}/data_documents.pkl", "rb") as f:
        data_documents = pickle.load(f)

    # chain.link documents
    with open(f"{folder}/chain_link_main_documents.pkl", "rb") as f:
        chain_link_documents = pickle.load(f)

    # chainlink youtube documents
    with open(f"{folder}/chain_link_you_tube_documents.pkl", "rb") as f:
        chain_link_youtube_documents = pickle.load(f)

    chainlink_search_retrevier = SearchRetriever.from_documents(
        blog_docs=blog_documents,
        tech_docs=technical_documents,
        data_docs=data_documents,
        chain_link_docs=chain_link_documents,
        chain_link_youtube_docs=chain_link_youtube_documents,
        k_final=20,
        logger=logger,
    )

    return chainlink_search_retrevier

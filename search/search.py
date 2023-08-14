from pydantic import BaseModel
from typing import Any, List, Optional, Dict
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from langchain.retrievers import TFIDFRetriever

import logging

logger = logging.getLogger(__name__)

class SearchRetriever(BaseRetriever, BaseModel):
    # blog_docs: List[Document]
    # tech_docs: List[Document]
    blog_retriever: BaseRetriever
    tech_retriever: BaseRetriever
    k_final: int = 4
    logger: Optional[logging.Logger] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        blog_docs: List[Document],
        tech_docs: List[Document],
        k_final: int = 4,
        logger: Any = None,
        **kwargs: Any,
    ):
        blog_ret = TFIDFRetriever.from_documents(blog_docs, k=30)
        tech_ret = TFIDFRetriever.from_documents(tech_docs, k=30)

        return cls(
            blog_retriever=blog_ret,
            tech_retriever=tech_ret,
            k_final=k_final,
            logger=logger,
        )

    def get_relevant_documents(self, query: str, type_:str='all') -> List[Dict]:
        """
        Get relevant documents for a given query.

        param query: The query to search for.
        param type_: The type of documents to search for. Can be 'blog', 'tech', or 'all'.
        """

        if type_ == "blog":
            r_docs = self.blog_retriever.get_relevant_documents(query)

            # Get only the metadata from the original documents
            r_docs = [doc.metadata for doc in r_docs][:self.k_final]

            return r_docs

        if type_ == "technical_document":
            r_docs = self.tech_retriever.get_relevant_documents(query)

            # Get only the metadata from the original documents
            r_docs = [doc.metadata for doc in r_docs][:self.k_final]

            return r_docs

        if type_ == "all":
            r_docs_1 = self.blog_retriever.get_relevant_documents(query)
            r_docs_2 = self.tech_retriever.get_relevant_documents(query)

            # Merge the two lists; one object per document
            r_docs = []
            for doc1, doc2 in zip(r_docs_1, r_docs_2):
                r_docs.append(doc1.metadata)
                r_docs.append(doc2.metadata)

            return r_docs[:self.k_final]

        raise ValueError("type_ must be one of 'blog', 'technical_document', or 'all'")
    
    def aget_relevant_documents(self):
        raise NotImplementedError("This method is not implemented yet.")
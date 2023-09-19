import re
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from langchain.retrievers import TFIDFRetriever
from config import get_logger

logger = get_logger(__name__)


class SearchRetriever(BaseRetriever, BaseModel):
    blog_retriever: TFIDFRetriever
    tech_retriever: TFIDFRetriever
    data_retriever: TFIDFRetriever
    chain_link_retriever: TFIDFRetriever
    chain_link_youtube_retriever: TFIDFRetriever
    all_docs_retriever: TFIDFRetriever
    networks = [
        "ethereum",
        "polygon",
        "optimism",
        "fantom",
        "harmony",
        "moonriver",
        "metis",
        "bnb",
        "arbitrum",
        "avalanche",
        "gnosis",
        "base",
        "moonbeam",
    ]
    k_final: int = 20

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        blog_docs: List[Document],
        tech_docs: List[Document],
        data_docs: List[Document],
        chain_link_docs: List[Document],
        chain_link_youtube_docs: List[Document],
        k_final: int = 20,
        logger: Any = None,
        **kwargs: Any,
    ):
        # Remove duplicates from chainlink_docs
        unique_texts = {doc.page_content: doc for doc in chain_link_docs}
        filtered_chainlink_docs = list(unique_texts.values())

        blog_ret = TFIDFRetriever.from_documents(blog_docs, k=30)
        tech_ret = TFIDFRetriever.from_documents(tech_docs, k=30)
        data_ret = TFIDFRetriever.from_documents(data_docs, k=30)
        chain_link_ret = TFIDFRetriever.from_documents(filtered_chainlink_docs, k=30)
        chain_link_youtube_ret = TFIDFRetriever.from_documents(
            chain_link_youtube_docs, k=30
        )

        all_docs = (
            blog_docs + tech_docs + filtered_chainlink_docs + chain_link_youtube_docs
        )
        all_docs_ret = TFIDFRetriever.from_documents(all_docs, k=30)

        return cls(
            blog_retriever=blog_ret,
            tech_retriever=tech_ret,
            data_retriever=data_ret,
            chain_link_retriever=chain_link_ret,
            chain_link_youtube_retriever=chain_link_youtube_ret,
            all_docs_retriever=all_docs_ret,
            k_final=k_final,
            logger=logger,
        )

    def get_relevant_documents(self, query: str, type_: str = "all") -> List[Document]:
        logger.info(f"Query: {query}")
        r_docs = []

        if type_ == "all":
            matching_docs_for_pair = self.find_texts_for_pair(
                query, self.data_retriever.docs
            )

            if matching_docs_for_pair:
                ordered_texts = self.reorder_matched_texts_by_network(
                    query, matching_docs_for_pair
                )
                r_docs.extend([doc.metadata for doc in ordered_texts[:2]])

            # Add 5 from all docs if not already added to r_docs
            r_docs.extend(
                [
                    doc.metadata
                    for doc in self.all_docs_retriever.get_relevant_documents(query)[:5]
                ]
            )

            retrievers = [
                self.tech_retriever,
                self.blog_retriever,
                self.chain_link_retriever,
                self.chain_link_youtube_retriever,
            ]

            for ret in retrievers:
                for doc in ret.get_relevant_documents(query):
                    if doc.metadata not in r_docs:
                        r_docs.append(doc.metadata)
                    if len(r_docs) >= self.k_final:
                        break

        elif type_ in ["blog", "technical_document"]:
            retriever = self.blog_retriever if type_ == "blog" else self.tech_retriever
            r_docs.extend(
                [
                    doc.metadata
                    for doc in retriever.get_relevant_documents(query)[: self.k_final]
                ]
            )
        else:
            raise ValueError(
                "type_ must be one of 'blog', 'technical_document', or 'all'"
            )

        # Make sure no duplicates; use 'source' as unique identifier
        r_docs = list({doc["source"]: doc for doc in r_docs}.values())
        return r_docs

    def extract_pair(self, query):
        pattern = r"(?i)([a-z]{3,6})\s?/\s?([a-z]{3,6})"
        matches = re.findall(pattern, query)
        return matches[0] if matches else None

    def find_texts_for_pair(self, query, docs):
        pair = self.extract_pair(query)

        if not pair:
            return []

        normalized_pair = ("/".join(pair)).lower().replace(" ", "")
        matching_docs = [
            doc
            for doc in docs
            if normalized_pair in doc.page_content.lower().replace(" ", "")
        ]

        return matching_docs

    def reorder_matched_texts_by_network(self, query, matched_docs):
        matched_networks = [net for net in self.networks if net in query.lower()]

        network_docs = [
            doc
            for doc in matched_docs
            if any(net in doc.page_content.lower() for net in matched_networks)
        ]
        non_network_docs = [doc for doc in matched_docs if doc not in network_docs]

        return network_docs + non_network_docs

    def aget_relevant_documents(self):
        raise NotImplementedError

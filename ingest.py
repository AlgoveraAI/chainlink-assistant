import os
import pickle
import argparse
from datetime import datetime

import dotenv
from ingest.docs import scrap_docs
from ingest.blogs import scrape_blogs
from ingest.education import scrap_education_docs
from ingest.stackoverflow import scrap_stackoverflow
from config import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    # Accept parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', type=bool, help='Parse technical documentation', default=False)
    parser.add_argument('--blogs', type=bool, help='Parse blog posts', default=False)
    parser.add_argument('--education', type=bool, help='Parse chainlink.education', default=False)
    parser.add_argument('--stackoverflow', type=bool, help='Parse stackoverflow', default=False)
    parser.add_argument('--all', type=bool, help='Parse all', default=False)
    parser.add_argument('--so_token', type=str, help='Stackoverflow token', default=None)
    args = parser.parse_args()

    # Log the arguments
    logger.info(f"docs: {args.docs}; blogs: {args.blogs}; education: {args.education}; stackoverflow: {args.stackoverflow}; all: {args.all}; so_token: {args.so_token}")

    # If all is true, set all to true
    if args.all:
        args.docs = True
        args.blogs = True
        args.education = True
        args.stackoverflow = True

    # If all or stackoverflow is true, ensure token is set
    if args.all or args.stackoverflow:
        if not args.so_token:
            raise ValueError("Stackoverflow token must be set if parsing stackoverflow")

    # Parse technical documentation
    if args.docs:
        logger.info("Parsing technical documentation")
        docs_documents = scrap_docs()

    # Parse blog posts
    if args.blogs:
        logger.info("Parsing blog posts")
        blog_urls = scrape_blogs()

    # Parse chainlink.education
    if args.education:
        logger.info("Parsing chainlink.education")
        chainlink_education_documents = scrap_education_docs()

    # Parse stackoverflow
    if args.stackoverflow:
        logger.info("Parsing stackoverflow")
        stackoverflow_documents = scrap_stackoverflow(args.so_token)

    # Log the number of documents
    if args.docs:
        logger.info(f"Docs: {len(docs_documents)}")
    if args.blogs:
        logger.info(f"Blogs: {len(blog_urls)}")
    if args.education:
        logger.info(f"Education: {len(chainlink_education_documents)}")
    if args.stackoverflow:
        logger.info(f"Stackoverflow: {len(stackoverflow_documents)}")

    # Combine all documents into one list
    documents = []
    documents_count = 0
    if args.docs:
        documents_count += len(docs_documents)
        documents.extend(docs_documents)
    if args.blogs:
        documents_count += len(blog_urls)
        documents.extend(blog_urls)
    if args.education:
        documents_count += len(chainlink_education_documents)
        documents.extend(chainlink_education_documents)
    if args.stackoverflow:
        documents_count += len(stackoverflow_documents)
        documents.extend(stackoverflow_documents)
        
    logger.info(f"Total documents: {documents_count}")
    # Log the total number of documents
    logger.info(f"Total: {len(documents)}")

    # Save documents to disk
    with open(f"./data/documents_{datetime.now().strftime('%Y-%m-%d')}.pkl", 'wb') as f:
        pickle.dump(documents, f)

    logger.info("Done")
## What is `chainlink_assistant`?

`chainlink_assistant` is a custom-built solution for question-answering and basic search functionalities for Chainlink. It consists of three components:

1. Ingesting from Chainlink's various sources.
2. Q&A which employs retrieval-augmented generation (RAG) to respond to user questions.
3. A search feature that utilizes TF-IDF to find the relevant document based on queries.

## What are some of the key tools/libraries used?

1. **selenium/beautifulsoup**: Used for scraping documents. Selenium is especially beneficial for scraping dynamically loaded websites.
2. **fastapi/uvicorn**: Employed for constructing HTTP and WebSocket endpoints.
3. **langchain**: Utilized for the RAG system and Q&A.

### Ingest

#### What are the data sources used?

- Technical documents: [docs.chain.link](https://docs.chain.link)
- Main website (including YouTube videos): [chain.link](https://chain.link)
- Chainlink academy: [chainlink-education on GitHub](https://github.com/oceanByte/chainlink-education)
- Blogs: [blog.chain.link](https://blog.chain.link)
- Data feeds: [data.chain.link](https://data.chain.link)
- StackOverflow: Scraped using StackOverflow's HTTP API.

#### What's the endpoint?
`http://localhost:8000/ingest`

**Calling using the requests library:**

```python
res = requests.post("http://localhost:8000/ingest")
```

Stackoverflow
The process of obtaining the access_token is automated. The relevant function, get_access_token, can be found in ingest_script.py.

Initially, use the requests library to log into StackOverflow.
Acquire the code from the redirect URI.
Exchange the code for an access_token.

#### Where are the retrieved documents stored? In what format are they stored?
The retrieved documents are located in /chainlink-assistant/data. These documents are stored as pickle files, and each pickle file contains a list of Langchain `Documents`.

#### What are the main .py files for this endpoint
1. `ingest_script.py` on the project root dir
2. all files in `ingest` folder

#### At the end of `ingest`, what files should we have in the data folder?

- Individual doceemnts

    - `tech_documents.pkl`
    - `chain_link_main_documents.pkl`
    - `chain_link_you_tube_documents.pkl`
    - `education_documents.pkl`
    - `blog_documents.pkl`
    - `data_documents.pkl`
    - `stackoverflow_documents.pkl`

- Combined documents

    - documents.pkl
        - This file includes: tech_documents, blog_documents, education_documents, stackoverflow_documents, chain_link_main_documents, chain_link_you_tube_documents.
        - However, it excludes data_documents. Based on our findings, our workflow uses data_documents as a standalone retriever to yield better outcomes.

- Vectorstores

    - `docs_all.index`
    - `docs_data.index`
    - `faiss_store_all.pkl`
    - `faiss_store_data.pkl`
    - vectorstores are used in RAG system

### QandA
QandA utilizes Langchain, but we have mostly built our own workflow that includes a router to choose between different workflows. The three workflows are:
    1. Short-form workflow for answering simple questions.
    2. Long-form workflow for answering complex questions and questions requiring code.
    3. Specialized workflow for answering questions about data feeds.

We also stream the output as tokens are generated. To achieve this on the frontend, we currently use websockets.

#### Steps involved in qanda:
    1. WebSocket connection is established.
    2. The request arrives as JSON with the following format:
        ```json 
        {
            'username':'xxxxx', 
            'message':'user's question'
        }
        ```
    3. The username can be used to validate credentials. Currently, we have hardcoded this in this version. In our production environment, we used Firebase.
    4. This triggers the get_answer function:
        4.1 It loads the vectorstore and retrieval (in this version, we load vectorstores and retrieval with every call, but this doesn't seem to significantly impact speed). Files index_all.index, index_data.index, faiss_vectorstore_all.pkl, and faiss_vectorstore_data.pkl are utilized.
        4.2 Depending on the question, it selects the appropriate workflow.
        4.3 It retrieves the documents containing potential answers.
        4.4 It then makes an LLM call.
    5. streams the output every token

#### Using the simple frontend for QandA
We've constructed a basic HTML-based frontend for the chat functionality. It's available at http://localhost:8000/chainlink. If the application is hosted elsewhere, modify the base URL accordingly.


### Search
Search provides near real-time search.

#### Endpoint for search
`http://localhost:8000/search`

```python
res = requests.post(url, json={'query':'xxxx', 'type_':'xxxxx'})
```
`type_` can be `technical_document`, `blog`, `all`. Currently, we just use `all`

#### Steps involved in search

1. when the endpoint is called (http endpoint)
2. it will reload the retrievers if there are newly ingested docs else uses the existing retriever
3. makes a search
4. returns the results

### Files used in the retriever
- `blog_documents.pkl`
- `tech_documents.pkl`
- `chain_link_main_documents.pkl`
- `chain_link_you_tube_documents.pkl`
- `data_documents.pkl`

### Others
1. currently we have excluded user authentication
2. no function to track usage
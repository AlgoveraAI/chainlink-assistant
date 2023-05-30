### Run 1
- all defaults with `GPTVectorStoreIndex`
- code questions (7 total) = 3r, 3w, 1p (r=right, w=wrong, p=partial)
    - of the 3w, 2 had correct retrieval one had partial retrieval
- address questions (5 total) - 4r, 1w
    - 1w had right retieval
- other questions (15 total) - 14r, 1w
    - the wrong answer had wrong retieval
- issues:
    - chunking loses global context for codes very obvious

### Run 2
- same as Run 1 but has 4 retrieval docs
- code questions = 7p
    - 4 right retrievals, 3 partial retrievals
- address questions = 3r, 1w, 1p
    - both right retrievals
- other questions = 14r, 1r
    - wrong retrieval

### Run 3
- Run 2 with twice num_output
- code questions = 7p
    - 4r, 1w, 2p retrieval
- address questions = 3r, 1w, 1p
    - both right retrievals
- other questions = 1w
    - wrong retreival

### Run 4
- Run 2 with response_mode refine
- code questions = 7p
    - 4r, 1w, 2p retrieval
- address questions = 2w, 3p
    - all correct retrievals
- other questions = 2r, 1w, 12 p
- lots of hallucinations

### Run 5
- Run 2 with tree_summarize
- code questions = 2w, 5p
    - 4r, 3p retrieval
- address questions = 4r, 1w
    - right but hallucination
- other questions = 13r, 1w, 1p
    - 1w retrieval, 1w retrieval but hallucination

### Run 6
- Run 2 with accumulate
- didnt analyse - produces a response for each node

### Run 7
- Run 2 with compact_aaumulate
- not released in pypi

### Run 8
- LLM ReRanker
- similarity_search returns tok_k=10, reranker picks top 3 used in answering
- code questions - 6r, 1w
    - wrong retrieval
- address question - 3r, 2w
    - right retrievals but hallucination
- 13r, 2w 
    - wrong retrievals

### Run 9
- Run 8 with tree_summarizer
- code questions = 4r, 3w
    -  wrong retrieval
- address quesitons = 4w, 1p
    - all right retrieval but hallucination
- other questions = 10r, 2w, 3p
    - 1 wrong retrieval other hallucinations

### Run 10
- PrevNextNodePostprocessor node postprocessor
- picks the "next" chunk
- code questions = 1r, 1w, 5p
    - 3r, 1w, 3p
- address questions = 2r, 3w
- other questions = 15r 

### Run 11
- AutoPrevNextNodePostprocessor
- code questions = 2r, 5p
    - 2r, 3p
- address questions = 2r, 3w
    - all right retrieval 
- other questions = 15r

### Run 12
- GPTDocumentSummaryIndex
- takes very long to build the GPTDocumentSummaryIndex: almost 90mins for chainlink
- code questions = 1r, 6p
    - 3r, 3p
- address questions = 2r, 3w
    - right retrievals
- other questions = 15r


### Overall
- basic works ok but with k=2
- try basic with k=2 and tree_summarize
- LLMReranker is ok
- for code - LLM Reranker?
- for add - basic or LLM Reranker
- for others - AutoPrevNextNodePostprocessor, LLM ReRanker, basic

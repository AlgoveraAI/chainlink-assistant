# algovate - LLM Retrieval Q&A Evaluation Library
This library is built to automate the evaluation process of retrieval question and answering (RQA). It's fully automated for LC-based RQA. For LlamaIndex, it works by passing a `query_engine` object. Only works for a single experiment.

# Features
- fully automates evaluation of LangChain based QandA
- in LC, supports the following
    - splitter:
        - "RecursiveCharacterTextSplitter"
        - "MarkdownTextSplitter"
        - "CharacterTextSplitter"
        - "TokenTextSplitter"
        - different `chunk_size` and `chunk_overlap`
    - search_type:
        - "similarity"
        - "mmr"
        - "svm"
        - "knn"
        - "tfidf"
        - can set different `k`, `score_threshold` for similarity_search
    - LLM Model
        - LC's chat_models
    - chain_type:
        - "stuff"
        - "map_reduce"
        - "refine"

- LlamaIndex
    - not fully automated
    - you will have to pass any query_engine and it get a response to ground_truths and scores them


# Installation
1. clone the repo
    `git clone https://github.com/algoveraai/algovate.git`
2. Navigate to the cloned directory
    `cd llm-automation`
3. Install the required packages
    `pip install -r requirements.txt`
4. Pip install the library
    `pip install -e .`


# Usage
First, it expects 2 things in the `./data` folder. They are `documents.pkl` and `ground_truths.pkl`. `documents.pkl` is a list of LC's document object containing all the context to be used in RQA. `groun_truths.pkl` is a list of `dict` containing `question` and `answer` pair. Used as ground truths.

1. LangChain
There are two ways run the experiments.

By using the `experiment` mode:
    Use the experiment.template.yaml to set the different experiments and run as follows:

    ```
    from algovate.langchain. evaluate import run_experiments

    yaml_file = "./algovate/data/experiments.yaml"
    yaml_type = "experiment"

    run_experiments(yaml_file, yaml_type)
    ```

By using the `generate` mode in which experiments are auto-generated using the different parameters to be tested. 
    You can edit the `parameter.template.yaml` to set the different parameters to be tested.

    ```
    yaml_file = "./algovate/data/parameter.yaml"
    yaml_type = "generate"

    run_experiments(yaml_file, yaml_type)
    ```

2. LlamaIndex

    ```
    from algovate.llama.evaluate import run_experiment, grade_model_answer, grade_model_retrieval

    query_engine = index.as_query_engine()
    result_df = run_experiment(query_engine=query_engine, 
                            ground_truths_file_path="/path/to/ground_truths.pkl",
                            grading_type="fast",
                            logger=logger)
    ```


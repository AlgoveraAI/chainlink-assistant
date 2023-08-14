# Chainlink AI Assistant

This is a collection of LLM programs for a personalized AI assistant that is driven by Chainlink’s publicly-available developer resources: 

* [Chainlink Developer Docs](https://docs.chain.link/getting-started/conceptual-overview)
* [Chainlink Tags on Stack Overflow](https://stackoverflow.com/questions/tagged/chainlink)
* [Chainlink Academy](https://chainlink.education/)
* [Chainlink blogs](https://blog.chain.link/)

Our goal is to improve the productivity of developers that are building with Chainlink infrastructure. Many developers already use ChatGPT, but this is a general model that (i) often outputs instructions that are out of date, (ii) isn’t specialized towards developing on top of Chainlink. 

We use a recent approach to personalizing AI assistants, called in-context retrieval-augmented language models (see research overview here), which has the advantage of citing sources and reducing hallucination (making stuff up).

You can find further details of the project in this [doc](https://docs.google.com/document/d/1KledT3tFueBkgaTI19K_3I2OXIGf4CtR0Hc5gwXIgR0/edit?usp=sharing).

# What's in this Repo?

## 1. Data

We have scraped various data sources relevent to Chainlink development such as the [Chainlink Developer Docs](https://docs.chain.link/getting-started/conceptual-overview), [Chainlink Tags on Stack Overflow](https://stackoverflow.com/questions/tagged/chainlink) and [Chainlink Academy](https://chainlink.education/). We run this text through the OpenAI embedding model and store it in a vector db. This data can be found in the `algovate/data` folder e.g. `documents.pkl`.

## 2. LLM Assistants

We have experimented with a variety of LLM assistants using LLM programming approaches like LLM workflows/chains and agents, with frameworks such as LangChain and Llama Index. These LLM assistants use a variety of retrieval methods (e.g. vector-based retrieval), logic and models (e.g. the new 16k token context window model from OpenAI). These can be found in the `algovate/langchain` and `algovate/llama` dirs, and notebooks.

# How to ingest data

# How to run chat/qanda
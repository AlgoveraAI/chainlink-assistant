from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

final_answer_system_template = """
As an AI assistant helping answer a user's question about Chainlink, your task is to provide the answer to the user's question based on the collection of documents provided. Each document is demarcated by the 'Source:' tag. 

In most cases, the answer to the user's question can be found in one of the documents.

If the documents do not contain the required information to answer user's question, respond with 'I don't know'. In this case, you can provide a link to the Chainlink documentation.

Each point in your answer should be formatted with corresponding reference(s) using markdown. Conclude your response with a footnote that enumerates all the references involved. Please make sure to use only the references provided in the documents and not to use any external references. 

The footnote should be formatted as follows: 
```
References:
[^1^]: <reference 1> 
[^2^]: <reference 2> 
[^3^]: <reference 3>
```
Please avoid duplicating references. For example, if the same reference is used twice in the answer, please only include it once in the footnote.
"""

final_answer_human_template = """
User's question: {question}

Document: {document}

Answer:
"""

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(final_answer_system_template),
        HumanMessagePromptTemplate.from_template(final_answer_human_template),
    ]
)

final_answer_2_system_template = """
As an AI assistant helping answer a user's question about Chainlink, your task is to provide the answer to the user's question based on the potential answers derived from previous LLM call(s). 
If the document doesn't contain the required information, respond with 'I don't know'.
Each point in your answer should be formatted with corresponding reference(s) using markdown. Conclude your response with a footnote that enumerates all the references involved. 

The footnote should be formatted as follows: 
```
References:
[^1^]: <reference 1> 
[^2^]: <reference 2> 
[^3^]: <reference 3>
```
Please avoid duplicating references. For example, if the same reference is used twice in the answer, please only include it once in the footnote.
"""

final_answer_2_human_template = """
User's question: {question}

Document: {document}

Answer:
"""

FINAL_ANSWER_2_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(final_answer_2_system_template),
        HumanMessagePromptTemplate.from_template(final_answer_2_human_template),
    ]
)

router_system_prompt = """
As an AI assistant helping ansswer a user's question about Chainlink, your first task is to route the question to the proper workflow. 
There are two workflows:
    1. short-form which is suitable for simple questions. It is bad at answering questions requiring code output.
    2. long-form which is suitable for complex questions. It is good at answering questions requiring code output.

Given a question, you need to route it to the proper workflow.

Your output should be a number between 0 and 1. 0 means the question should be routed to the short-form workflow. 1 means the question should be routed to the long-form workflow.
"""

router_human_prompt = """
Question: {question}
"""

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(router_system_prompt),
        HumanMessagePromptTemplate.from_template(router_human_prompt),
    ]
)
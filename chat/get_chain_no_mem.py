import logging
import tiktoken
from chat.prompts_no_mem import (
    FINAL_ANSWER_PROMPT,
    FINAL_ANSWER_2_PROMPT,
    ROUTER_PROMPT,
)
from chat.utils import get_retriever_chain, get_streaming_chain

from schemas import ChatResponse, Sender, MessageType

try:
    logger = createLogHandler(__name__, "logs.log")
except:
    logger = logging.getLogger(__name__)
    # get formatter and stream handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.info("no mem chain.......")

model = "gpt-3.5-turbo"
try:
    encoding = tiktoken.encoding_for_model(model)
except KeyError:
    logger.info(f"Encoding for model {model} not found. Using default encoding.")
    encoding = tiktoken.get_encoding("cl100k_base")


def calculate_tokens(document, encoding):
    """Calculate the number of tokens in a list of documents."""
    return len(encoding.encode(document))


def concatenate_documents(documents, max_tokens):
    """Combine documents up to a certain token limit."""
    combined_docs = ""
    token_count = 0
    used_docs = []

    for doc in documents:
        doc_tokens = calculate_tokens(doc.page_content, encoding)
        if (token_count + doc_tokens) <= max_tokens:
            combined_docs += f"\n\n{doc.page_content}\nSource: {doc.metadata['source']}"
            token_count += doc_tokens
            used_docs.append(doc)

    return combined_docs, used_docs


def call_llm_final_answer(question, document, chain, stream=False):
    """Call LLM with a question and a single document."""
    chain.prompt = FINAL_ANSWER_PROMPT
    if stream:
        return chain.apredict(question=question, document=document)
    else:
        return chain.predict(question=question, document=document)


def call_llm_final_2_answer(question, document, chain):
    """Call LLM with a question and a single document."""
    chain.prompt = FINAL_ANSWER_2_PROMPT
    return chain.apredict(question=question, document=document)


def process_documents(question, chain, retriever, max_tokens=14_000):
    """Process a list of documents with LLM calls."""

    # Use router to decide which workflow to use
    chain.prompt = ROUTER_PROMPT
    try:
        workflow = int(chain.predict(question=question))
    except Exception as e:
        logger.error(f"Error in router: {e}")
        workflow = 0

    logger.info(f"Using workflow {workflow}")

    documents = retriever.get_relevant_documents(question, workflow=workflow)
    batches = []
    num_llm_calls = 0
    while documents:
        batch, used_docs = concatenate_documents(documents, max_tokens)
        batches.append(batch)
        # logger.info(f"Calling LLM with {batch}")
        documents = [doc for doc in documents if doc not in used_docs]
        num_llm_calls += 1
        logger.info(
            f"LLM call {num_llm_calls} complete. {len(documents)} documents remaining."
        )

    return batches, num_llm_calls, workflow


async def get_answer(question, manager, max_tokens=14_000):
    """Get an answer to a question."""

    # Get the retriever chain
    retriever, base_chain = get_retriever_chain()

    # Send a status message
    resp = ChatResponse(
        sender=Sender.BOT, message="Retrieving Documents", type=MessageType.STATUS
    )
    await manager.broadcast(resp)

    # Main code that calls process_documents
    batches, num_llm_calls, workflow = process_documents(
        question=question, max_tokens=max_tokens, chain=base_chain, retriever=retriever
    )

    # Get the streaming chain
    chain_stream = get_streaming_chain(
        manager=manager, chain=base_chain, workflow=workflow
    )

    # Send a status message
    resp = ChatResponse(
        sender=Sender.BOT, message=f"Generating Answer", type=MessageType.STATUS
    )
    await manager.broadcast(resp)

    if num_llm_calls == 1:
        result = await call_llm_final_answer(
            question=question, document=batches[0], chain=chain_stream, stream=True
        )
        return result

    else:
        # Handle the list of batches
        results = []
        for batch in batches:
            result = call_llm_final_answer(
                question=question, document=batch, chain=base_chain
            )
            results.append(result)

        combined_result = " ".join(results)

        logger.info(f"Final LLM call with {len(results)} results.")
        combined_result = await call_llm_final_2_answer(
            question=question, document=combined_result, chain=chain_stream
        )

        return combined_result

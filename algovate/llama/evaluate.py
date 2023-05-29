from numpy import log
import yaml
import time
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.docstore.document import Document
from llama_index.indices.query.base import BaseQueryEngine


from ..utils import prepare_retriever, prepare_llm, prepare_chain, SPLITTER_MAPPING, get_logger, generate_experiment_configs_from_file
from ..prompts import GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_DOCS_PROMPT


def grade_model_answer(
    ground_truths: List[Dict],
    predicted_answers: List[Dict],
    grading_type: str = "fast", # fast, full
):
    """
    Grade a model answer against a ground truth answer.
    """
    if grading_type == "fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    else:
        prompt = GRADE_ANSWER_PROMPT

    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                                      prompt=prompt)

    graded_outputs = eval_chain.evaluate(ground_truths,
                                         predicted_answers,
                                         question_key="question",
                                         prediction_key="result")

    return graded_outputs

def grade_model_retrieval(
    ground_truths: List[Dict],
    predicted_answers: List[Dict],
    grading_type: str = "fast", # fast, full
):
    """
    Grade a model answer against a ground truth answer.
    """
    if grading_type == "fast":
        prompt = GRADE_DOCS_PROMPT_FAST
    else:
        prompt = GRADE_DOCS_PROMPT

    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                                      prompt=prompt)

    graded_outputs = eval_chain.evaluate(ground_truths,
                                         predicted_answers,
                                         question_key="question",
                                         prediction_key="result")

    return graded_outputs

def run_eval(
    query_engine:BaseQueryEngine,
    ground_truth,
    grading_type,
    logger,
):
    try:
        predictions = []
        retrieved_docs = []
        gt_dataset = []
        latency = []

        start_time = time.time()
        question = ground_truth['question']
        response = query_engine.query(question)
        result = response.response

        retrieved_doc_text = ""
        for i, doc in enumerate(response.source_nodes):
            retrieved_doc_text += "Doc %s: " % str(i+1) + \
                doc.node.text + " "
        
        retrieved = {"question": ground_truth["question"],
                    "answer": ground_truth["answer"], "result": retrieved_doc_text}
        retrieved_docs.append(retrieved)
        
        predictions.append({ "question": question, 
                             "answer":ground_truth["answer"], 
                             "result": str(result)}
        )
        
        gt_dataset.append(ground_truth)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        latency.append(elapsed_time)
        
        d = pd.DataFrame(predictions)

        try:
            graded_answers = grade_model_answer(
                gt_dataset, predictions, grading_type=grading_type
            )
            d['answerScore'] = [g['text'] for g in graded_answers]
            d['answerScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                             'justification': text} for text in d['answerScore']]
        except:
            graded_answers = []
            d['answerScore'] = [{"score":"N/A", "justification":"N/A"}]
        try:
            graded_retrieval = grade_model_retrieval(
            gt_dataset, retrieved_docs, grading_type=grading_type)
            d['retrievalScore'] = [g['text'] for g in graded_retrieval]
            d['retrievalScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                                'justification': text} for text in d['retrievalScore']]
        except:
            graded_retrieval = []
            d['retrievalScore'] = [{"score":"N/A", "justification":"N/A"}]
        
        d['latency'] = latency
        d_dict = d.to_dict('records')
        logger.info(d_dict)
        return d_dict
    
    except Exception as e:
        logger.warning("######################")
        logger.warning(e)
        logger.info(ground_truth)
        logger.warning("######################")

def run_experiment(
    query_engine:BaseQueryEngine,
    ground_truths_file_path: str,
    grading_type: str,
    logger,
):
    with open(ground_truths_file_path, "rb") as f:
        ground_truths = pickle.load(f)


    logger.info(f"Number of ground truths: {len(ground_truths)}")
    
    result_df = pd.DataFrame()
    for i in range(len(ground_truths)):
        ground_truth = ground_truths[i]
        result = run_eval(query_engine=query_engine, 
                          ground_truth=ground_truth, 
                          grading_type=grading_type,
                          logger=logger)
        result_df = result_df.append(result, ignore_index=True)
        logger.info(f"Result:\n{result_df.to_dict('records')}")
        logger.info(f"Finished {i+1} out of {len(ground_truths)}")
        
    return result_df


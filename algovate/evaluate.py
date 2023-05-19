import yaml
import time
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.docstore.document import Document

from .utils import prepare_retriever, prepare_llm, prepare_chain, SPLITTER_MAPPING, get_logger, generate_experiment_configs_from_file
from .prompts import GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_DOCS_PROMPT


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
    chain,
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
    
        predictions.append(chain(ground_truth))
        gt_dataset.append(ground_truth)
        end_time = time.time()
        elapsed_time = end_time - start_time
        latency.append(elapsed_time)
        d = pd.DataFrame(predictions)
        try:
            graded_answers = grade_model_answer(
            gt_dataset, predictions, grading_type=grading_type)
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
    documents: List[Document],
    ground_truths: List[Dict],
    experiment_config,
    logger,
):
    experiment_config = experiment_config["experiment"]
    experiment_name = experiment_config["name"]
    logger.info(f"Running experiment: {experiment_name}")

    splitter = SPLITTER_MAPPING[experiment_config["splitter"]]
    chunk_size = experiment_config["chunk_size"]
    chunk_overlap = experiment_config["chunk_overlap"]
    search_type = experiment_config["search_type"]
    k = experiment_config["k"]
    score_threshold = experiment_config["score_threshold"]
    model_name = experiment_config["model_name"]
    chain_type = experiment_config["chain_type"]
    grading_type = experiment_config["grading_type"]
    
    logger.info("Preparing retriever...")
    retriever = prepare_retriever(documents=documents, 
                                    splitter=splitter,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    search_type=search_type,
                                    k=k,
                                    score_threshold=score_threshold)

    logger.info("Preparing LLM...")
    llm = prepare_llm(model_name=model_name)

    logger.info("Preparing chain...")
    chain = prepare_chain(chain_type=chain_type,
                            llm=llm,
                            retriever=retriever)
    
    logger.info(f"Number of ground truths: {len(ground_truths)}")
    result_df = pd.DataFrame()
    for i in range(len(ground_truths)):
        ground_truth = ground_truths[i]
        result = run_eval(chain=chain, 
                          ground_truth=ground_truth, 
                          grading_type=grading_type,
                          logger=logger)
        result_df = result_df.append(result, ignore_index=True)

    logger.info("Saving results...")

    # make results folder if it doesn't exist
    Path("./results").mkdir(parents=True, exist_ok=True)
    result_df.to_csv(f"./results/{experiment_name}.csv", index=False)

    answer_sum = result_df["answerScore"].apply(lambda x: 0 if str(x["score"]) == "N/A" else x["score"]).sum()
    retrieval_sum = result_df["retrievalScore"].apply(lambda x: 0 if str(x["score"]) == "N/A" else x["score"]).sum()

    exp_result = {
                "experiment_name": experiment_name, 
                "grading_type": grading_type,
                "splitter": splitter,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "search_type": search_type,
                "k": k,
                "score_threshold": score_threshold,
                "model_name": model_name,
                "chain_type": chain_type,
                "num_ground_truths": len(ground_truths),
                "num_correct": answer_sum,
                "num_retrieval_correct": retrieval_sum,
                "num_incorrect": len(ground_truths) - answer_sum,
                "num_retrieval_incorrect": len(ground_truths) - retrieval_sum,
                "accuracy": answer_sum / len(ground_truths),
                "retrieval_accuracy": retrieval_sum / len(ground_truths),
                }
    return exp_result

def run_experiments(
    yaml_file,
    yaml_type, # "generate" or "load" 
):
    if yaml_type == "generate":
        config = generate_experiment_configs_from_file(yaml_file)
        ground_truth_path = config["paths"]["ground_truths_path"]
        documents_path = config["paths"]["documents_path"]
        log_file_path = config["paths"]["log_file_path"]
    else:
        with open(yaml_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.FullLoader) 
        ground_truth_path = config["paths"][0]["ground_truths_path"]
        documents_path = config["paths"][0]["documents_path"]
        log_file_path = config["paths"][0]["log_file_path"]
    
    logger = get_logger(__name__, log_file_path)

    logger.info("Loading ground truths and documents...")
    with open(ground_truth_path, "rb") as f:
        ground_truths = pickle.load(f)
    
    with open(documents_path, "rb") as f:
        documents = pickle.load(f)

    logger.info("Running experiments...")
    logger.info(f"Number of experiments: {len(config['experiments'])}")
    num_experiments = len(config["experiments"])
    
    exp_df = pd.DataFrame()
    for i in range(num_experiments):
        logger.info("#############################################")
        logger.info(f"Running experiment {i+1}/{num_experiments}")
        experiment_config = config["experiments"][i]
        exp_result = run_experiment(documents=documents, 
                                    ground_truths=ground_truths, 
                                    experiment_config=experiment_config, 
                                    logger=logger)
        exp_df = exp_df.append(exp_result, ignore_index=True)
        logger.info(f"Finished experiment {i+1}/{num_experiments}")
        logger.info("#############################################")
    logger.info("Saving final results...")
    exp_df.to_csv("./results/final_results.csv", index=False)
    logger.info("Finished running experiments.")

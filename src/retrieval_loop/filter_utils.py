import pandas as pd
from datasets import Dataset
from transformers import pipeline
import torch
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from fast_bleu import SelfBLEU
from collections import defaultdict
import sys
from transformers import RobertaTokenizer
import datasets
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# filter the top docs of the retrieved docs by self-bleu score, to keep the self-bleu score of the top 5 docs =< 0.5


# sys.path.append('../retrieval_loop')
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
from evaluate_dpr_retrieval import has_answers, SimpleTokenizer, evaluate_retrieval


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Read and parse the configuration file
    config_file_path = args.config_file_path
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    print(f'Config Loaded: {config}')

    return config


def read_result_file(input_file):
    with open(input_file, 'r') as f:
        result = json.load(f)

    #result format:
    # {
    # "12": {
    # "question": "right to property according to the constitution of india is a?",
    # "answers": [
    #     "constitutional right"
    # ],
    # "contexts": [
    #     {
    #         "docid": "baichuan2-13b-chat_nq_from_llm-embedder_None_loop4_21_20240116132218",
    #         "score": 0.9580259323120117,
    #         "has_answer": false
    #     },
    #     {
    #         "docid": "baichuan2-13b-chat_nq_from_llm-embedder_None_loop4_21_20240116132218",
    #         "score": 0.9580259323120117,
    #         "has_answer": false
    #     }...
    # ]
    # }
    # }

    return result


def get_index(elasticsearch_url, index_name):
    index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name, verbose=False)
    return index


def get_added_documents(added_documents):
    # for all files under added_documents, read them into a huggingface dataset
    add_files = os.listdir(added_documents)
    # if jsonl file, read it into a dataset
    files = [f for f in add_files if f.endswith('.jsonl')]

    added_dataset = datasets.load_dataset('json', data_files=[added_documents + '/' + f for f in files])['train']
    input_doc_dict = {doc['id']: doc['response'] for doc in added_dataset}
    return input_doc_dict


def get_doc_text(docid, index, input_doc_dict, task='filter_source'):
    flag = False
    if docid in input_doc_dict:
        text = input_doc_dict[docid]
        flag = True
    if not flag:
        text = index.get_document_by_id(docid).page_content

    if task == 'filter_source':
        return text  # Split the text into words for BLEU computation
    else:
        return text.split()


def get_detector(model_dir, cuda='0'):
    # 设置 GPU 设备
    gpu_ids = cuda.split(',')
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')})")

    # 初始化 tokenizer（可选，如果需要任何预处理）
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)

    # 初始化文本分类 pipeline
    detector = pipeline(
        'text-classification',
        model=model_dir,
        tokenizer=model_dir,
        device=0 if torch.cuda.is_available() else -1,
        framework='pt',
        top_k=None,  # 获取所有标签
        batch_size=512  # 批处理大小
    )

    return detector


def compute_source(question, documents, detector):
    # paired = [dict(text=q, text_pair=a) for q, a in zip(batch['question'], batch['answer'])]
    # out = detector(paired , max_length=512, truncation=True)

    question = [question] * len(documents)
    paired = [dict(text=q, text_pair=a) for q, a in zip(question, documents)]
    out = detector(paired , max_length=512, truncation=True)
    return out


def find_docs_with_source(query_id, result, index, detector, added_dataset, num_docs=100, task='filter_source'):
    original_contexts = result['contexts'][:40]
    doc_ids = [context['docid'] for context in original_contexts]
    # compute label, if the doc_id is digit, it is human generated, if not, it is LLM generated

    origin_label = [0 if doc_id.isdigit() else 1 for doc_id in doc_ids]

    question = result['question']
    doc_texts = [get_doc_text(doc_id, index, added_dataset, task) for doc_id in doc_ids]

    candidate_contexts = original_contexts
    candidate_docs = doc_texts

    current_source = compute_source(question, candidate_docs, detector)
    # print(doc_ids)
    # print(current_source)
    # [[{'label': 'LABEL_1', 'score': 0.9999973773956299}, {'label': 'LABEL_0', 'score': 2.6422230803291313e-06}],
    # [{'label': 'LABEL_0', 'score': 0.9999982118606567}, {'label': 'LABEL_1', 'score': 1.7340204294669093e-06}]]
    # LABEL_0: Human, LABEL_1: chatgpt
    #get pred
    pred = []
    for i in range(len(current_source)):
        if current_source[i][0]['label'] == 'LABEL_0':
            pred.append(0)
        else:
            pred.append(1)
    new_contexts = []
    human_count = 0

    for i, context in enumerate(candidate_contexts):
        if human_count < num_docs:
            if pred[i] == 0:
                new_contexts.append(context)
                human_count += 1
                print(f'add human doc for query {query_id}: {context["docid"]}')
        else:
            break

    if len(new_contexts) < num_docs:
        # if there are not enough human docs, add from the first of the original_contexts
        for i, context in enumerate(original_contexts):
            if len(new_contexts) < num_docs and pred[i] == 1:
                new_contexts.append(context)
                print(f'add llm doc for query {query_id}: {context["docid"]}')


    result['contexts'] = new_contexts
    print(f'pred: {pred}')
    print(f'len of new_contexts: {len(new_contexts)}')
    return result, origin_label, pred


def compute_self_bleu(documents):
    # Set weights for trigram only
    weights = {'trigram': (1 / 3., 1 / 3., 1 / 3.)}
    self_bleu = SelfBLEU(documents, weights)
    scores = self_bleu.get_score()
    # Since we are only interested in the trigram score, we will return that directly
    average_score = np.mean(scores['trigram'])
    return average_score


def find_docs_with_self_bleu_constraint(query_id, result, index, add_data, max_self_bleu=0.4, num_docs=5, task='filter_bleu'):
    original_contexts = result['contexts'][:40]
    doc_ids = [context['docid'] for context in original_contexts]

    doc_texts = [get_doc_text(doc_id, index, add_data, task) for doc_id in doc_ids]

    candidate_contexts = original_contexts[:num_docs]
    candidate_docs = doc_texts[:num_docs]

    current_self_bleu = compute_self_bleu(candidate_docs)
    print(f'Initial Self-BLEU (trigram): {current_self_bleu:.4f}')

    next_docs_pool = doc_texts[num_docs:]
    next_contexts_pool = original_contexts[num_docs:]
    pop_contexts_pool = []

    while current_self_bleu > max_self_bleu and next_docs_pool:
        bleu_scores_excluding_each_doc = [compute_self_bleu(candidate_docs[:i] + candidate_docs[i + 1:]) for i in
                                          range(len(candidate_docs))]
        min_bleu_idx = np.argmin(bleu_scores_excluding_each_doc)
        removed_context = candidate_contexts.pop(min_bleu_idx)
        pop_contexts_pool.append(removed_context)
        candidate_docs.pop(min_bleu_idx)

        new_doc = next_docs_pool.pop(0)
        new_context = next_contexts_pool.pop(0)

        candidate_docs.append(new_doc)
        candidate_contexts.append(new_context)

        new_self_bleu = compute_self_bleu(candidate_docs)

        print(
            f'query_id: {query_id}, removed docid: {removed_context["docid"]}, new docid: {new_context["docid"]},  old self-bleu: {current_self_bleu:.4f}, new self-bleu: {new_self_bleu:.4f}')

        current_self_bleu = new_self_bleu

    # Append the remaining documents that were not filtered out
    candidate_contexts.extend(next_contexts_pool)
    if len(candidate_contexts) < num_docs:
        candidate_contexts.extend(pop_contexts_pool[:num_docs - len(candidate_contexts)])

    result['contexts'] = candidate_contexts
    return result


def run_filter_source(input_file, output_file, elasticsearch_url, index_name, max_self_bleu, num_docs,llm_detector_model, added_dataset):
    parent_dir = os.path.dirname(os.path.dirname(input_file))
    # llm_data = None
    result = read_result_file(input_file)
    index = get_index(elasticsearch_url, index_name)
    detector = get_detector(llm_detector_model, '0')
    label = []
    pred = []
    # parent_dir = os.path.dirname(os.path.dirname(input_file))
    # llm_data = gather_LLM_gen_text(parent_dir)
    for query_id, query_result in tqdm(result.items()):
        filtered_result, label_q, pred_q = find_docs_with_source(query_id, query_result, index, detector, added_dataset, num_docs, task='filter_source')
        result[query_id] = filtered_result
        label.extend(label_q)
        pred.extend(pred_q)

    with open(output_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Filtered results have been written to {output_file}")

    print(f'evaluating detection...')
    print(classification_report(label, pred, target_names=['human','chatgpt'], output_dict=True))

    print(f'evaluating input...')
    evaluate_retrieval(input_file, [5, 20, 100], False)
    print(f'evaluating output...')
    evaluate_retrieval(output_file, [5, 20, 100], False)

    refined_json_file = output_file
    input_trec_file = input_file+'.trec'
    output_trec_file = output_file+'.trec'
    refine_trec_file(input_trec_file, output_trec_file, refined_json_file)


def run_filter_bleu(input_file, output_file, elasticsearch_url, index_name, max_self_bleu, num_docs, added_dataset):
    result = read_result_file(input_file)
    index = get_index(elasticsearch_url, index_name)
    # llm_data = None
    for query_id, query_result in tqdm(result.items()):
        filtered_result = find_docs_with_self_bleu_constraint(query_id, query_result, index, added_dataset, max_self_bleu, num_docs, task='filter_bleu')
        result[query_id] = filtered_result

    with open(output_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Filtered results have been written to {output_file}")

    print(f'evaluating input...')
    evaluate_retrieval(input_file, [5, 20, 100], False)
    print(f'evaluating output...')
    evaluate_retrieval(output_file, [5, 20, 100], False)

    refined_json_file = output_file
    input_trec_file = input_file+'.trec'
    output_trec_file = output_file+'.trec'
    refine_trec_file(input_trec_file, output_trec_file, refined_json_file)


def refine_trec_file(input_trec_file, output_trec_file, refined_json_file):
    result = read_result_file(refined_json_file)
    q_d_dict = defaultdict(list)
    for query_id, query_result in result.items():
        for context in query_result['contexts']:
            q_d_dict[query_id].append(context['docid'])

    with open(input_trec_file, 'r') as f:
        lines = f.readlines()
    # if the docid is in the refined_json_file, keep it, otherwise remove it

    with open(output_trec_file, 'w') as f:
        for line in lines:
            query_id, _, doc_id, _, score, tag = line.strip().split()
            if doc_id in q_d_dict[query_id]:
                rank = q_d_dict[query_id].index(doc_id) + 1
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} {tag}\n")

    print(f"Refined TREC file has been written to {output_trec_file}")


if __name__ == '__main__':
    config = get_args()
    input_files = config.get("input_file")
    output_files = config.get("output_file")
    elasticsearch_url = config.get("elasticsearch_url")
    index_name = config.get("index_name", "bm25_psgs_index")
    max_self_bleu = config.get("max_self_bleu", 0.4)
    num_docs = config.get("num_docs", 5)
    task = config.get("task")
    llm_detector_model = config.get("llm_detector_model", "roberta-large-mnli")
    added_documents = config.get("added_documents")

    added_dataset = get_added_documents(added_documents)
    if task == 'filter_source':
        for input_file, output_file in zip(input_files, output_files):
            run_filter_source(input_file, output_file, elasticsearch_url, index_name, max_self_bleu, num_docs,
                              llm_detector_model, added_dataset)
    else:
        for input_file, output_file in zip(input_files, output_files):
            run_filter_bleu(input_file, output_file, elasticsearch_url, index_name, max_self_bleu, num_docs,
                            added_dataset)




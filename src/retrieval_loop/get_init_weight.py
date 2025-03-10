import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from typing import List, Tuple, Iterator
import itertools
import random
import json
import sys
import datasets
from retrieve_methods import Retrieval, load_retrieval_embeddings
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
sys.path.append('../llm_zero_generate')
from tqdm import tqdm
from fast_bleu import SelfBLEU
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from deap import base, creator, tools, algorithms
import math
import pickle
import os
import logging
from transformers import pipeline
import multiprocessing
from scipy.special import expit as sigmoid
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
import torch
import argparse
import re
from transformers import RobertaTokenizer
import string
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# 设置 Python 内置随机种子
random.seed(42)

# 设置 NumPy 随机种子
np.random.seed(42)

def get_text(input_file):
    """
    从文档中提取文本。
    jsonl文件，每行：{"id": "baichuan2-13b-chat_nq_from_bge-base_None_loop4_21_20231229122730", "question": "right to property according to the constitution of india is a?", "answers": ["constitutional right"], "response": "Right to Property According to Constitution of India", "exact_match": 0}

    """
    dataset = datasets.load_dataset('json', data_files=input_file)['train']
    return dataset


@retry(stop=stop_after_attempt(20), wait=wait_exponential(multiplier=1, max=10))
def get_response_llm(model_name, text, filter_words=None):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user",
             "content": text},
        ],
        max_tokens=128,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=filter_words,
        stream=False,
    )

    # print(text)
    # print(completion.choices[0].message.content)
    # print("\n")

    resp = completion.choices[0].message.content.strip().replace("\n", " ")
    # memory cleanup
    del completion
    return resp


def get_pseudo_query_dataset(input_file):

    def generate_pseudo_query(text: str, model_name="openai/gpt-4o-mini") -> str:
        """
        从文本生成伪查询。
        占位符实现：提取关键词或摘要。
        """
        # 简单的关键词提取作为占位符
        prompt = "Please generate a most possible question based on the following paragraph:\n\n" + text
        response = get_response_llm(model_name, prompt)
        filter_words = []

        while response.strip() == "" \
                or len(response.split(" ")) < 2:
            response = get_response_llm(model_name, prompt)
        # if response > 100 words, truncate to 100 words
        response = " ".join(response.split(" ")[:100])
        return response

    dataset = get_text(input_file)
    def process(data):
        text = data['response']
        pseudo_query = generate_pseudo_query(text)
        data['pseudo_query'] = pseudo_query
        return data
    dataset_with_pseudo_query = dataset.map(process, num_proc=10)
    return dataset_with_pseudo_query



def retrieve_documents(pseudo_query_dataset, retrieval_model, normalize_embeddings,
                       dense_index_name, dense_index_path, elasticsearch_url, with_alpha, retrieved_file, pq_file):
    # save pseudo query to file
    temp_file = pq_file
    pseudo_query_dataset.to_json(temp_file, orient='records', lines=True)
    # retrieve documents
    query_files = temp_file
    page_content_column = 'pseudo_query'
    Retrieval([query_files], page_content_column, retrieval_model, dense_index_name, dense_index_path, normalize_embeddings,
              [retrieved_file], elasticsearch_url, with_alpha)


def get_retrieved_documents(retreived_file):
    # analyze trec file
    with open(retreived_file, 'r') as f:
        lines = f.readlines()
    retrieved_docs = {}
    retrieved_scores = {}
    for line in lines:
        qid, _, pid, _, score, _ = line.strip().split(" ")
        qid = str(qid)
        if qid not in retrieved_docs:
            retrieved_docs[qid] = []
            retrieved_scores[qid] = []
        retrieved_docs[qid].append(pid)
        retrieved_scores[qid].append(float(score))
    return retrieved_docs, retrieved_scores


def get_embedding(retrieval_model, normalize_embeddings):
    # map retrieval model names: DPR, Contriever, RetroMAE, all-mpnet, BGE, LLM-Embedder
    query_instruction = ''
    doc_instruction = ''
    if 'DPR' in retrieval_model.upper():
        retrieval_model = '../../../../../Rob_LLM/ret_model/DPR/facebook-dpr-ctx_encoder-multiset-base'
        query_model = '../../../../../Rob_LLM/ret_model/DPR/facebook-dpr-question_encoder-multiset-base'
    elif 'CONTRIEVER' in retrieval_model.upper():
        retrieval_model = '../../../../../Rob_LLM/ret_model/contriever-base-msmarco'
    elif 'RETROMAE' in retrieval_model.upper():
        retrieval_model = '../../../../../Rob_LLM/ret_model/RetroMAE_BEIR'
    elif 'ALL-MPNET' in retrieval_model.upper():
        retrieval_model = '../../../../../Rob_LLM/ret_model/all-mpnet-base-v2'
    elif 'BGE-LARGE' in retrieval_model.upper():
        retrieval_model = '../../../../../Rob_LLM/ret_model/bge-large-en-v1.5'
        query_instruction = 'Represent this sentence for searching relevant passages: '
    elif 'BGE-BASE' in retrieval_model.upper():
        retrieval_model = '../../../../../Rob_LLM/ret_model/bge-base-en-v1.5'
        query_instruction = 'Represent this sentence for searching relevant passages: '
    elif 'LLM-EMBEDDER' in retrieval_model.upper():
        retrieval_model = '../../../../../Rob_LLM/ret_model/llm-embedder'
        query_instruction = 'Represent this query for retrieving relevant documents: '
        doc_instruction = "Represent this document for retrieval: "
    elif 'BM25' in retrieval_model.upper():
        retrieval_model = 'BM25'
    else:
        raise ValueError(f'unknown retrieval model: {retrieval_model}')

    # load the query embedder and index
    if "DPR" in retrieval_model.upper():
        embeddings = load_retrieval_embeddings(retrieval_model, normalize_embeddings=normalize_embeddings)
        query_embeddings = load_retrieval_embeddings(query_model, normalize_embeddings=normalize_embeddings)
        print(f'loaded ctx embedder: {retrieval_model}')
        print(f'loaded query embedder: {query_model}')

        return query_embeddings, embeddings

    else:
        embeddings = load_retrieval_embeddings(retrieval_model, normalize_embeddings=normalize_embeddings)
        print(f'loaded query embedder: {retrieval_model}')

        return embeddings, embeddings


def compute_similarity_score(input_file, query_embeddings, doc_embeddings, query_key='pseudo_query'):
    dataset = get_text(input_file)
    sim_scores = []
    qids = []
    dids = []
    queries = []
    docs = []
    pd_dict = {}
    for data in dataset:
        qids.append(str(data['id']))
        if 'docid' in data:
            dids.append(data['docid'])
        else:
            dids.append(data['id'])
        queries.append(data[query_key])
        docs.append(data['response'])
    batch_size = 128
    for i in range(0, len(queries), batch_size):
        query_batch = queries[i:i + batch_size]
        doc_batch = docs[i:i + batch_size]
        query_vecs = query_embeddings.embed_queries(query_batch)
        doc_vecs = doc_embeddings.embed_documents(doc_batch)
        sim_scores.extend(linear_kernel(query_vecs, doc_vecs).diagonal())
    for i in range(len(qids)):
        if qids[i] not in pd_dict:
            pd_dict[qids[i]] = {}
        pd_dict[qids[i]][dids[i]] = sim_scores[i]

    return pd_dict


def add_doc_to_list(pd_dict, retrieved_docs, retrieved_scores, weights):
    import copy

    # Create deep copies to avoid modifying the originals
    new_retrieved_docs = copy.deepcopy(retrieved_docs)
    new_retrieved_scores = copy.deepcopy(retrieved_scores)

    # Iterate over each query ID in pd_dict
    for qid in pd_dict:
        # Retrieve the current document and score lists, or initialize them if not present
        doc_list = new_retrieved_docs.get(qid, []).copy()
        score_list = new_retrieved_scores.get(qid, []).copy()

        # Define the new document ID and its weighted score
        new_docids = list(pd_dict[qid].keys())
        for new_docid in new_docids:
            new_score = pd_dict[qid][new_docid] * weights[qid][new_docid]

            # Append the new document and score
            doc_list.append(new_docid)
            score_list.append(new_score)

        # Sort the documents and scores based on the scores in descending order
        sorted_pairs = sorted(zip(doc_list, score_list), key=lambda x: x[1], reverse=True)
        sorted_docs, sorted_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # Update the copied dictionaries with the sorted lists
        new_retrieved_docs[qid] = list(sorted_docs)
        new_retrieved_scores[qid] = list(sorted_scores)

    return new_retrieved_docs, new_retrieved_scores


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
        batch_size=256  # 批处理大小
    )

    return detector


def compute_llm_probability(detector, queries, retrieved_docs, total_text_source, batch_size=2048):
    """
    计算文档由大型语言模型（LLM）生成的概率。

    参数:
    - detector: 已初始化的 HuggingFace pipeline 用于文本分类。
    - queries (dict): 查询ID到查询文本的映射。
    - retrieved_docs (dict): 查询ID到文档ID列表的映射。
    - total_text_source (dict): 查询ID到文档ID及其对应文本的嵌套字典。
    - batch_size (int): 每个批次处理的样本数量，默认32。

    返回:
    - llm_probs (dict): 查询ID到文档ID及其对应LLM概率的嵌套字典。
    """
    llm_probs = {qid: {} for qid in queries}

    # 准备所有的 (query, doc_text, qid, docid) 对
    paired_data = []
    for qid, query_text in queries.items():
        if qid not in retrieved_docs:
            continue
        doc_ids = retrieved_docs.get(qid, [])
        doc_texts = total_text_source.get(qid, {})
        assert len(doc_ids) == len(doc_texts), f"文档数量不匹配: id:{qid}, {len(doc_ids)} != {len(doc_texts)}, doc_ids: {doc_ids}, doc_texts: {doc_texts}"
        for docid in doc_ids:
            paired_data.append((qid, query_text, doc_texts[docid], docid))

    total_batches = math.ceil(len(paired_data) / batch_size)
    # 使用 tqdm 显示总进度
    with tqdm(total=total_batches, desc="Computing LLM probabilities", unit="batch") as pbar:
        for i in range(0, len(paired_data), batch_size):
            batch = paired_data[i:i + batch_size]
            qids, queries_batch, doc_texts_batch, doc_ids_batch = zip(*batch)

            # 创建输入格式
            inputs = [{"text": q, "text_pair": d} for q, d in zip(queries_batch, doc_texts_batch)]

            # 获取模型结果
            try:
                results = detector(inputs, max_length=512, truncation=True)
            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}")
                results = [None] * len(inputs)
            #[[{'label': 'LABEL_1', 'score': 0.9999973773956299}, {'label': 'LABEL_0', 'score': 2.6422230803291313e-06}],
            # [{'label': 'LABEL_0', 'score': 0.9999982118606567}, {'label': 'LABEL_1', 'score': 1.7340204294669093e-06}]]

            # 更新 llm_probs 字典
            for qid, docid, result in zip(qids, doc_ids_batch, results):
                llm_probs[qid][docid] = result[0]['score'] if result[0]['label'] == 'LABEL_1' else result[1]['score']

            # 更新进度条
            pbar.update(1)

    return llm_probs


def compute_source(question, documents, detector):
    # paired = [dict(text=q, text_pair=a) for q, a in zip(batch['question'], batch['answer'])]
    # out = detector(paired , max_length=512, truncation=True)

    question = [question] * len(documents)
    paired = [dict(text=q, text_pair=a) for q, a in zip(question, documents)]
    out = detector(paired, max_length=512, truncation=True)
    return out


def find_docs_with_source(query_id, result, index, detector, num_docs=100, llm_data=None, task='filter_source'):
    original_contexts = result['contexts']
    doc_ids = [context['docid'] for context in original_contexts]
    # compute label, if the doc_id is digit, it is human generated, if not, it is LLM generated

    origin_label = [0 if doc_id.isdigit() else 1 for doc_id in doc_ids]

    question = result['question']
    doc_texts = [get_doc_text(doc_id, index, task, llm_data) for doc_id in doc_ids]

    candidate_contexts = original_contexts
    candidate_docs = doc_texts

    current_source = compute_source(question, candidate_docs, detector)
    # print(doc_ids)
    # print(current_source)
    # [{'label': 'LABEL_1', 'score': 0.999941349029541}, {'label': 'LABEL_1', 'score': 0.9999754428863525}, {'label': 'LABEL_1', 'score': 0.9999765157699585}, {'label': 'LABEL_0', 'score': 0.8834363222122192}, {'label': 'LABEL_1', 'score': 0.999976634979248}]
    # LABEL_0: Human, LABEL_1: chatgpt
    # get pred
    pred = [int(item['label'][-1]) for item in current_source]
    new_contexts = []
    human_count = 0

    for i, context in enumerate(candidate_contexts):
        if human_count < num_docs:
            if current_source[i]['label'] == 'LABEL_0':
                new_contexts.append(context)
                human_count += 1
            else:
                print(f'del llm doc: {context["docid"]}')
        else:
            new_contexts.append(context)

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


def get_index(elasticsearch_url, index_name):
    index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name, verbose=False)
    return index


def get_doc_text(docid, index, input_doc_dict, task='filter_source'):
    # get the text of the document in input_dataset
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


def get_total_doc_text(retrieval_doc_dict, input_dataset, index, task='filter_source'):
    total_text = {qid: {} for qid in retrieval_doc_dict}

    # 收集input_dataset中所有的doc_id和文本
    input_doc_dict = {doc['id']: doc['response'] for doc in input_dataset}

    # 1. 收集所有唯一的 doc_ids
    unique_doc_ids = set()
    for doc_ids in retrieval_doc_dict.values():
        unique_doc_ids.update(doc_ids)

    # 定义获取单个 doc_text 的任务
    def fetch_doc_text(doc_id):
        return doc_id, get_doc_text(doc_id, index, input_doc_dict, task)

    # 使用 ThreadPoolExecutor 并行获取文档文本
    doc_text_dict = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_doc_text, doc_id): doc_id for doc_id in unique_doc_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching unique document texts"):
            doc_id, doc_text = future.result()
            doc_text_dict[doc_id] = doc_text

    # 分配文本到对应的查询
    for qid, doc_ids in tqdm(retrieval_doc_dict.items(), total=len(retrieval_doc_dict),
                             desc="Assigning texts to queries"):
        for doc_id in doc_ids:
            total_text[qid][doc_id] = doc_text_dict.get(doc_id, "")

    return total_text


def distinct_n(sentences, n=3):
    """
    计算 Distinct-N 分数。

    Args:
        sentences (list of str): 生成的句子列表。
        n (int): n-gram 的阶数（默认为1，即 unigrams）。

    Returns:
        float: Distinct-N 分数。
    """
    ngrams = []
    for sentence in sentences:
        tokens = sentence
        if len(tokens) < n:
            continue
        ngrams += [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))

    if total_ngrams == 0:
        return 0.0
    return -unique_ngrams / total_ngrams


def compute_distinctn_total(doc_text_dict, doc_ids):
    """
    计算文档列表的分数。
    较高的表明多样性较低。
    """
    doc_texts = []
    for docid in doc_ids:
        doc_texts.append(doc_text_dict[docid].split())
    logging.info("compute distinct_3 for single doc")
    # current_distinctn = compute_self_bleu(doc_texts)
    current_distinctn = distinct_n(doc_texts, 3)
    logging.info("finish compute distinct_3 for single doc")
    return current_distinctn


def compute_self_bleu(documents):
    # Set weights for trigram only
    weights = {'trigram': (1 / 3., 1 / 3., 1 / 3.)}
    self_bleu = SelfBLEU(documents, weights)
    scores = self_bleu.get_score()
    # Since we are only interested in the trigram score, we will return that directly
    average_score = np.mean(scores['trigram'])
    return average_score


def compute_self_bleu_total(doc_text_dict, doc_ids):
    """
    计算文档列表的自我 BLEU 分数。
    较高的 self-BLEU 表明多样性较低。
    """
    doc_texts = []
    for docid in doc_ids:
        doc_texts.append(doc_text_dict[docid].split())
    current_self_bleu = compute_self_bleu(doc_texts)
    return current_self_bleu


def mean_llm_probability(llm_prob_dict, doc_ids):
    """
    计算文档列表按照位置顺序加权的平均 LLM 概率。
    """
    total_llm_prob = 0.0
    for i, docid in enumerate(doc_ids):
        total_llm_prob += llm_prob_dict[docid] / (i + 1)
    return total_llm_prob


def compute_correctness_probability(fact_score, doc_ids, num_docs):
    """
    计算文档列表的正确性概率。
    """
    total_correct_prob = 0.0
    for i, docid in enumerate(doc_ids[:num_docs]):
        fact_score_now = fact_score.get(docid, 1.0)
        total_correct_prob += fact_score_now / (i + 1)
    return total_correct_prob


def evaluate_weights(alpha, beta, gamma, delta, retrieved_docs, retrieved_scores, sim_scores,
                     self_bleu, llm_probs, num_top, fact_score, qid, retrieved_text, llm_prob_dict):
    weight = {}
    weight[qid] = {}
    for did in sim_scores:
        weight[qid][did] = sigmoid(
            1 + alpha * sim_scores[did] - beta * self_bleu - gamma * llm_probs[qid][did] + delta * fact_score[did])
        if weight[qid][did] < 0:
            weight[qid][did] = 0
        if weight[qid][did] > 1:
            weight[qid][did] = 1
    new_retrieved_docs, new_retrieved_scores = add_doc_to_list({qid: sim_scores}, retrieved_docs, retrieved_scores,
                                                               weight)
    current_retrieved_docs = new_retrieved_docs
    current_retrieved_scores = new_retrieved_scores
    sum_sim_score = np.mean(current_retrieved_scores[qid][:num_top])
    sum_self_bleu = compute_distinctn_total(retrieved_text, current_retrieved_docs[qid][:num_top])
    sum_llm_prob = mean_llm_probability(llm_prob_dict, current_retrieved_docs[qid][:num_top])
    sum_correct_prob = compute_correctness_probability(fact_score, current_retrieved_docs[qid], num_top)

    return (sum_sim_score, sum_self_bleu, sum_llm_prob, sum_correct_prob)


def setup_deap():
    """
    设置 DEAP 的相关配置，包括个体、适应度函数和操作符。
    """
    # 定义适应度函数，四个目标
    if not hasattr(setup_deap, "creator_initialized"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        setup_deap.creator_initialized = True

    toolbox = base.Toolbox()

    # 定义个体属性，每个参数在 [0, 1] 之间
    toolbox.register("attr_alpha", random.uniform, 0, 1)
    toolbox.register("attr_beta", random.uniform, 0, 1)
    toolbox.register("attr_gamma", random.uniform, 0, 1)
    toolbox.register("attr_delta", random.uniform, 0, 1)

    # 个体由四个参数组成
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_alpha, toolbox.attr_beta, toolbox.attr_gamma, toolbox.attr_delta), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 交叉、变异和选择操作
    toolbox.register("mate", crossover_and_clip, alpha=0.3)
    # toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=-1.0, up=1.0, indpb=0.2)
    toolbox.register("mutate", mutate_and_ensure_real, eta=5.0, low=0, up=1.0, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    return toolbox


def crossover_and_clip(ind1, ind2, alpha=0.5, low=0, up=1.0):
    """
    自定义交叉函数：执行 Blend 交叉，然后裁剪基因值。
    """
    tools.cxBlend(ind1, ind2, alpha=alpha)

    for individual in [ind1, ind2]:
        for i in range(len(individual)):
            individual[i] = max(min(individual[i], up), low)

    return ind1, ind2


def select_best_solution(pareto_front, fit_weights=None):
    """
    从帕累托前沿中选择一个最佳解决方案。
    这里选取膝点，可以根据实际需求选择其他选择方法。
    """
    # 假设以最大化 (sim_score + correct_prob) 为主要目标，同时最小化 self_bleu 和 llm_prob
    # 可以使用简单的加权方法或其他优选方法
    # 这里使用简单的加和评分
    if fit_weights is None:
        fit_weights = [1, 1, 1, 1]
    best = None
    best_score = -np.inf
    a, b, c, d = fit_weights
    # print(f'pareto_front: {pareto_front}')
    for ind in pareto_front:
        score = a*ind.fitness.values[0] + d*ind.fitness.values[3] - b*ind.fitness.values[1] - c*ind.fitness.values[2]  # 实际需调整
        if score > best_score:
            best_score = score
            best = ind
    return best


def mutate_and_ensure_real(individual, eta=20.0, low=0, up=1.0, indpb=0.2):
    """
    自定义变异函数：执行多项式界限变异，然后确保所有基因都是实数。
    """
    # 使用 DEAP 的 mutPolynomialBounded 进行变异
    for i, gene in enumerate(individual):
        # 处理复数
        if isinstance(gene, complex):
            print(f"Gene {i} is complex: {gene}. Converting to real part.")
            individual[i] = gene.real

        # 处理非浮点数类型
        if not isinstance(individual[i], float):
            try:
                individual[i] = float(individual[i])
            except (ValueError, TypeError):
                print(f"Gene {i} cannot be converted to float: {individual[i]}. Setting to low bound.")
                individual[i] = low

        # 处理 NaN 和无穷大
        if math.isnan(individual[i]) or math.isinf(individual[i]):
            print(
                f"Gene {i} is NaN or Inf: {individual[i]}. Setting to {'low' if individual[i] < low else 'up'} bound.")
            individual[i] = low if individual[i] < low else up

        # 确保基因在指定范围内
        if individual[i] < low:
            print(f"Gene {i} below lower bound: {individual[i]}. Setting to low bound.")
            individual[i] = low
        elif individual[i] > up:
            print(f"Gene {i} above upper bound: {individual[i]}. Setting to up bound.")
            individual[i] = up

    mutated_individual, = tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)

    # 遍历个体中的每个基因，确保都是实数
    for i in range(len(mutated_individual)):
        gene = mutated_individual[i]
        if isinstance(gene, complex):
            print(f"Post-mutation Gene {i} is complex: {gene}. Converting to real part.")
            mutated_individual[i] = gene.real
        if math.isnan(gene) or math.isinf(gene):
            print(
                f"Post-mutation Gene {i} is NaN or Inf: {gene}. Setting to {'low' if gene < low else 'up'} bound.")
            mutated_individual[i] = low if gene < low else up
        if mutated_individual[i] < low:
            print(f"Post-mutation Gene {i} below lower bound: {mutated_individual[i]}. Setting to low bound.")
            mutated_individual[i] = low
        elif mutated_individual[i] > up:
            print(f"Post-mutation Gene {i} above upper bound: {mutated_individual[i]}. Setting to up bound.")
            mutated_individual[i] = up
    # print(f"Mutated individual: {mutated_individual}")
    return (mutated_individual,)


def get_fact_dict(new_add_fact_score_file):
    fact_score = {}
    # if new_add_fact_score_file is a directory, then we need to get the fact score from the files in the directory
    if os.path.isdir(new_add_fact_score_file):
        for file in os.listdir(new_add_fact_score_file):
            with open(os.path.join(new_add_fact_score_file, file), 'r') as f:
                # {"id": "baichuan2-13b-chat_nq_from_bge-base_None_loop4_21_20231229122730", "score": 0.7, "exact_match": 0}
                for line in f:
                    data_line = json.loads(line)
                    fact_score[data_line['id']] = data_line['score']
    else:
        with open(new_add_fact_score_file, 'r') as f:
            # {"id": "baichuan2-13b-chat_nq_from_bge-base_None_loop4_21_20231229122730", "score": 0.7, "exact_match": 0}
            for line in f:
                data_line = json.loads(line)
                fact_score[data_line['id']] = data_line['score']
    return fact_score


def mean_fact_probability(fact_score, doc_ids):
    """
    计算文档列表按照位置顺序加权的平均 fact score。
    """
    total_fact_prob = 0.0
    for i, docid in enumerate(doc_ids):
        fact_score_now = fact_score.get(docid, 1.0)
        total_fact_prob += fact_score_now / (i + 1)
    return total_fact_prob


def process_document(document, input_file, retrieved_file, retrieval_model, normalize_embeddings, llm_detector_model,
                     bm25_index, dense_index_name, dense_index_path, elasticsearch_url, with_alpha,
                     new_add_fact_score_file,
                     num_top_documents, output_file, added_documents, population_size=50, generations=100,
                     crossover_prob=0.5, mutation_prob=20.0, fit_weights=None):
    """
    处理单个文档：计算初始权重并添加到索引中。
    """
    # 步骤1：获取文本生成伪查询
    if fit_weights is None:
        fit_weights = [1, 1, 1, 1]
    print(f"Gathering pseudo queries for document")
    if not os.path.exists(input_file):
        pseudo_query_dataset = get_pseudo_query_dataset(document)
    else:
        pseudo_query_dataset = get_text(input_file)
    # 步骤2：检索相关文档
    print(f"Retrieving documents for pseudo queries")
    pq_file = input_file

    if not os.path.exists(retrieved_file + '.trec'):
        retrieve_documents(pseudo_query_dataset, retrieval_model, normalize_embeddings,
                           dense_index_name, dense_index_path, elasticsearch_url, with_alpha, retrieved_file, pq_file)

    # 步骤3：检索相关文档
    retrieved_docs_ori, retrieved_scores_ori = get_retrieved_documents(retrieved_file + '.trec')

    # 步骤4：计算相似度评分并合并
    logger.warning(f"Computing similarity scores for pseudo queries")
    q_embedding, d_embedding = get_embedding(retrieval_model, normalize_embeddings)
    sim_scores = compute_similarity_score(pq_file, q_embedding, d_embedding)

    weights = {qid: {did: 1 for did in sim_scores[qid]} for qid in sim_scores}
    retrieved_docs, retrieved_scores = add_doc_to_list(sim_scores, retrieved_docs_ori, retrieved_scores_ori, weights)
    logger.warning("getting total text source")

    total_text_path = retrieved_file + '_total_text.pkl'
    logger.warning(f"total text path: {total_text_path}, {os.path.exists(total_text_path)}")

    # for all files under added_documents, read them into a huggingface dataset
    add_files = os.listdir(added_documents)
    # if jsonl file, read it into a dataset
    files = [f for f in add_files if f.endswith('.jsonl')]

    added_dataset = datasets.load_dataset('json', data_files=[added_documents + '/' + f for f in files])['train']

    if os.path.exists(total_text_path):
        with open(total_text_path, 'rb') as f:
            total_text_source = pickle.load(f)
    else:
        total_text_source = get_total_doc_text(retrieved_docs, added_dataset, bm25_index, task='filter_source')
        with open(total_text_path, 'wb') as f:
            pickle.dump(total_text_source, f)

    # total_text_source = get_total_doc_text(retrieved_docs, pseudo_query_dataset, bm25_index, task='filter_source')
    # 步骤5：获取detecter，factor score
    print("getting detector")
    detector = get_detector(llm_detector_model)

    queries = {}
    for doc in pseudo_query_dataset:
        qid = doc['id']
        queries[qid] = doc['pseudo_query']

    total_llm_probs_path = retrieved_file + '_total_llm_probs.pkl'
    logger.warning(f"total llm probs path: {total_llm_probs_path}, {os.path.exists(total_llm_probs_path)}")
    if os.path.exists(total_llm_probs_path):
        print("loading total llm probs")
        with open(total_llm_probs_path, 'rb') as f:
            total_llm_probs = pickle.load(f)
    else:
        print("computing total llm probs")
        total_llm_probs = compute_llm_probability(detector, queries, retrieved_docs, total_text_source)
        with open(total_llm_probs_path, 'wb') as f:
            pickle.dump(total_llm_probs, f)
    # total_llm_probs = compute_llm_probability(detector, queries, retrieved_docs, total_text_source)

    fact_score = get_fact_dict(new_add_fact_score_file)

    # 对每个待添加文档，计算目标函数
    toolbox = setup_deap()
    data = pseudo_query_dataset


    new_data = []

    # for doc in data:
    def process(doc):
        qid = doc['id']
        retrieved_docs_q = {qid: retrieved_docs[qid]}
        retrieved_scores_q = {qid: retrieved_scores[qid]}
        retrieved_scores_q_ori = {qid: retrieved_scores_ori[qid]}
        retrieved_docs_q_ori = {qid: retrieved_docs_ori[qid]}
        # sim_score_q = {qid: sim_scores[qid]}
        # pseudo_query = doc['pseudo_query']
        retrieval_text = total_text_source[qid]
        self_bleu_scores = compute_distinctn_total(retrieval_text, retrieved_docs_q[qid][:num_top_documents])

        # 定义适应度函数
        def fitness_function(individual):
            alpha, beta, gamma, delta = individual
            return evaluate_weights(alpha, beta, gamma, delta, retrieved_docs_q_ori, retrieved_scores_q_ori, sim_scores[qid],
                                    self_bleu_scores, total_llm_probs, num_top_documents, fact_score, qid,
                                    retrieval_text,
                                    total_llm_probs[qid])

        local_toolbox = base.Toolbox()
        local_toolbox.register("evaluate", fitness_function)
        local_toolbox.register("mate", toolbox.mate)
        local_toolbox.register("mutate", toolbox.mutate)
        local_toolbox.register("select", toolbox.select)
        # cpu_count = multiprocessing.cpu_count()
        # print(f"CPU count: {cpu_count}")
        # pool = multiprocessing.Pool(cpu_count)
        # toolbox.register("map", pool.map)
        # 初始化种群
        print(f"Optimizing weights for document {qid}")
        pop = toolbox.population(n=population_size)
        # 检查初始种群
        for individual in pop:
            for gene in individual:
                if isinstance(gene, complex):
                    raise ValueError("初始种群中存在复数值。")
        # 使用 NSGA-II 进行多目标优化
        algorithms.eaMuPlusLambda(pop, local_toolbox, mu=population_size, lambda_=population_size,
                                  cxpb=crossover_prob, mutpb=mutation_prob, ngen=generations,
                                  stats=None, halloffame=None, verbose=False)

        # 提取帕累托前沿
        pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

        # 绘制帕累托前沿
        # plot_pareto_front(pareto_front)

        # 选择最佳解决方案
        best_solution = select_best_solution(pareto_front)
        alpha, beta, gamma, delta = best_solution

        print(f"优化得到的参数: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")

        # 步骤7：计算初始权重
        # 使用给定公式：w = 1 - alpha * sim_score - beta * self_bleu - gamma * llm_prob - delta * correct_prob
        # 这里需要根据 retrieved_docs 计算综合权重
        # print(f"相关性分数: {sim_scores[qid][qid]}, BLEU分数: {self_bleu_scores}, LLM概率: {total_llm_probs[qid][qid]}, Fact分数: {fact_score[qid]}")
        w = sigmoid(
            1 + fit_weights[0]*alpha * sim_scores[qid][qid] - fit_weights[1]*beta * self_bleu_scores -
            fit_weights[2]*gamma * total_llm_probs[qid][qid] + fit_weights[3]*delta *fact_score[qid])

        print(f"计算得到的初始权重: {w}")
        value = float(w)
        if math.isnan(value):
            print(f"Weight is not a decimal: {w}")
            w = 0.0
        doc['alpha'] = w

        # new_data.append(doc)
        # pool.close()
        return doc

    # 保存文档
    data = data.map(process, num_proc=20)
    data.to_json(output_file)


def classify_docs(retrieval_docs, retrieval_scores):
    """
    将检索到的文档分为人类生成和 LLM 生成。
    """
    human_docs = {}
    llm_docs = {}

    human_scores = {}
    llm_scores = {}

    for qid in retrieval_docs:
        human_docs[qid] = []
        llm_docs[qid] = []
        human_scores[qid] = []
        llm_scores[qid] = []
        for i, docid in enumerate(retrieval_docs[qid]):
            if docid.isdigit():
                human_docs[qid].append(docid)
                human_scores[qid].append(retrieval_scores[qid][i])
            else:
                llm_docs[qid].append(docid)
                llm_scores[qid].append(retrieval_scores[qid][i])

    return human_docs, llm_docs, human_scores, llm_scores


def get_llm_query_doc_file(queries, llm_docs, total_text_source, output_file):
    """
    生成 LLM 查询文档文件。
    """
    data = []
    for qid in tqdm(queries, desc="Generating LLM query-doc file", total=len(queries)):
        for i, docid in enumerate(llm_docs[qid]):
            data.append({
                "id": qid,
                "docid": docid,
                "question": queries[qid],
                "response": total_text_source[qid][docid]
            })
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def update_process_document(input_query, retrieved_file, retrieval_model, normalize_embeddings, llm_detector_model,
                            bm25_index, dense_index_name, dense_index_path, elasticsearch_url, with_alpha,
                            new_add_fact_score_file,
                            num_top_documents, output_file, added_documents, population_size=50, generations=100,
                            crossover_prob=0.5, mutation_prob=20.0, fit_weights=None):
    """
    处理单个文档：计算初始权重并添加到索引中。
    """
    if fit_weights is None:
        fit_weights = [1, 1, 1, 1]

    # 步骤1：获取待更新权重的文档
    queries = {}
    for file in tqdm(input_query, total=len(input_query), desc="Reading input query files"):
        data = get_text(file)
        for doc in data:
            qid = str(doc['id'])
            queries[qid] = doc['question']

    retrieved_docs = {}
    retrieved_scores = {}
    for retrieval_file in retrieved_file:
        retrieved_docs_ori, retrieved_scores_ori = get_retrieved_documents(retrieval_file + '.trec')
        for qid in retrieved_docs_ori:
            retrieved_docs[qid] = retrieved_docs_ori[qid]
            retrieved_scores[qid] = retrieved_scores_ori[qid]

    logger.warning(f"classifying docs.")
    human_docs, llm_docs, human_scores, llm_scores = classify_docs(retrieved_docs, retrieved_scores)
    llm_qids = llm_docs.keys()
    # logger.warning(f"human docs: {human_docs.keys()}")
    # logger.warning(f"llm docs: {llm_docs.keys()}")
    # 步骤2：计算相似度评分并计算权重
    logger.warning("getting total text source")

    total_text_path = retrieved_file[2] + '_total_text.pkl'
    logger.warning(f"total text path: {total_text_path}, {os.path.exists(total_text_path)}")

    # for all files under added_documents, read them into a huggingface dataset
    add_files = os.listdir(added_documents)
    # if jsonl file, read it into a dataset
    files = [f for f in add_files if f.endswith('.jsonl')]

    added_dataset = datasets.load_dataset('json', data_files=[added_documents + '/' + f for f in files])['train']

    if os.path.exists(total_text_path):
        with open(total_text_path, 'rb') as f:
            total_text_source = pickle.load(f)
    else:
        total_text_source = get_total_doc_text(retrieved_docs, added_dataset, bm25_index, task='filter_source')
        with open(total_text_path, 'wb') as f:
            pickle.dump(total_text_source, f)

    llm_pq_file = retrieved_file[2] + '_llm_pq.jsonl'
    get_llm_query_doc_file(queries, llm_docs, total_text_source, llm_pq_file)

    logger.warning(f"retrieval llm docs.")

    q_embedding, d_embedding = get_embedding(retrieval_model, normalize_embeddings)
    sim_scores = compute_similarity_score(llm_pq_file, q_embedding, d_embedding, query_key='question')
    # print(f"{sim_scores}")
    # weights = {qid: {did: llm_scores[qid][did]/sim_scores[qid][did] for did in sim_scores[qid]} for qid in sim_scores}

    # 步骤3：获取detecter，factor score
    print("getting detector")
    detector = get_detector(llm_detector_model)

    total_llm_probs_path = retrieved_file[2] + '_total_llm_probs.pkl'
    logger.warning(f"total llm probs path: {total_llm_probs_path}, {os.path.exists(total_llm_probs_path)}")
    if os.path.exists(total_llm_probs_path):
        print("loading total llm probs")
        with open(total_llm_probs_path, 'rb') as f:
            total_llm_probs = pickle.load(f)
    else:
        print("computing total llm probs")
        total_llm_probs = compute_llm_probability(detector, queries, retrieved_docs, total_text_source)
        with open(total_llm_probs_path, 'wb') as f:
            pickle.dump(total_llm_probs, f)
    # total_llm_probs = compute_llm_probability(detector, queries, llm_docs, total_text_source)

    fact_score = get_fact_dict(new_add_fact_score_file)

    # 对每个查询，计算目标函数
    toolbox = setup_deap()
    query_data = get_text(input_query)
    # filter out llm query
    query_data = query_data.filter(lambda x: str(x['id']) in sim_scores)
    logger.warning(f"query data length: {len(query_data)}")
    new_data = []

    def process(doc):
        qid = str(doc['id'])
        retrieved_docs_q = {qid: retrieved_docs[qid]}
        retrieved_scores_q = {qid: retrieved_scores[qid]}
        retrieved_scores_q_ori = {qid: human_scores[qid]}
        retrieved_docs_q_ori = {qid: human_docs[qid]}
        # sim_score_q = {qid: sim_scores[qid]}
        # pseudo_query = doc['pseudo_query']
        retrieval_text = total_text_source[qid]
        self_bleu_scores = compute_distinctn_total(retrieval_text, retrieved_docs_q[qid][:num_top_documents])

        # 定义适应度函数
        def fitness_function(individual):
            alpha, beta, gamma, delta = individual
            return evaluate_weights(alpha, beta, gamma, delta, retrieved_docs_q_ori, retrieved_scores_q_ori, sim_scores[qid],
                                    self_bleu_scores, total_llm_probs, num_top_documents, fact_score, qid,
                                    retrieval_text,
                                    total_llm_probs[qid])

        local_toolbox = base.Toolbox()
        local_toolbox.register("evaluate", fitness_function)
        local_toolbox.register("mate", toolbox.mate)
        local_toolbox.register("mutate", toolbox.mutate)
        local_toolbox.register("select", toolbox.select)
        # cpu_count = multiprocessing.cpu_count()
        # print(f"CPU count: {cpu_count}")
        # pool = multiprocessing.Pool(cpu_count)
        # toolbox.register("map", pool.map)
        # 初始化种群
        print(f"Optimizing weights for document {qid}")
        pop = toolbox.population(n=population_size)
        # 检查初始种群
        for individual in pop:
            for gene in individual:
                if isinstance(gene, complex):
                    raise ValueError("初始种群中存在复数值。")
        # 使用 NSGA-II 进行多目标优化
        algorithms.eaMuPlusLambda(pop, local_toolbox, mu=population_size, lambda_=population_size,
                                  cxpb=crossover_prob, mutpb=mutation_prob, ngen=generations,
                                  stats=None, halloffame=None, verbose=False)

        # 提取帕累托前沿
        pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

        # 绘制帕累托前沿
        # plot_pareto_front(pareto_front)

        # 选择最佳解决方案
        best_solution = select_best_solution(pareto_front)
        alpha, beta, gamma, delta = best_solution

        print(f"优化得到的参数: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")
        if not alpha:
            alpha = 0.0
        if not beta:
            beta = 0.0
        if not gamma:
            gamma = 0.0
        if not delta:
            delta = 0.0
        doc['alpha'] = alpha*fit_weights[0]
        doc['beta'] = beta*fit_weights[1]
        doc['gamma'] = gamma*fit_weights[2]
        doc['delta'] = delta*fit_weights[3]
        doc['self_bleu'] = self_bleu_scores
        # new_data.append(doc)
        # pool.close()
        return doc

    # 保存文档
    query_data = query_data.map(process, num_proc=20)
    # query_data.to_json(output_file)
    get_reweighted_file(llm_pq_file, query_data, output_file, sim_scores, total_llm_probs, fact_score)


def get_reweighted_file(pq_file, reweighted_dataset, output_path, sim_scores, total_llm_probs, fact_score):
    '''
    生成输出文件，格式为：id, question, answer, response, exactmatch, alpha(weight)
    :param pq_file: 获取 qid, did, question, response
    :param reweighted_dataset: 获取 qid, answer
    :param output_path: 输出文件路径
    :param sim_scores: 相似分数
    :param total_llm_probs: LLM 概率总和
    :param fact_score: 事实分数
    :return: 无
    '''
    # 加载 pq_file 数据集
    pq_data = datasets.load_dataset('json', data_files=pq_file)['train']

    # 初始化字典以存储各类数据
    qadict = {}
    a_dict = {}
    b_dict = {}
    g_dict = {}
    d_dict = {}
    bleu_dict = {}

    for data in reweighted_dataset:
        qadict[data['id']] = data['answer']
        a_dict[data['id']] = data.get('alpha', 1.0)  # 如果没有 alpha，默认值为 1.0
        b_dict[data['id']] = data.get('beta', 1.0)
        g_dict[data['id']] = data.get('gamma', 1.0)
        d_dict[data['id']] = data.get('delta', 1.0)
        bleu_dict[data['id']] = data.get('self_bleu', 1.0)

    def process(data):
        qid = str(data['id'])
        did = data['docid']
        question = data['question']
        response = data['response']
        answer = qadict.get(qid, "")

        # 计算权重
        weight_input = (
                1
                + a_dict.get(qid, 0) * sim_scores.get(qid, {}).get(did, 0)
                - b_dict.get(qid, 0) * bleu_dict.get(qid, 0)
                - g_dict.get(qid, 0) * total_llm_probs.get(qid, {}).get(did, 0)
                + d_dict.get(qid, 0) * fact_score.get(did, 0)
        )
        weight = sigmoid(weight_input)

        # 更新数据字典
        data['answers'] = answer
        data['alpha'] = weight
        return data

    # 对数据集中的每个条目应用处理函数
    new_data = pq_data.map(process, num_proc=20)

    new_data = evaluate(new_data)

    # 移除原始的 'id' 列并将 'did' 重命名为 'id'
    new_data = new_data.remove_columns(['id'])
    new_data = new_data.rename_column('docid', 'id')

    # 将 Hugging Face 的 Dataset 转换为 Pandas DataFrame 以便进行操作
    df = new_data.to_pandas()

    # 按 'id' 分组并聚合
    # 对于 'alpha' 计算平均值，其他字段假设相同，取第一个值
    aggregated_df = df.groupby('id').agg({
        'question': 'first',
        'response': 'first',
        'answers': 'first',
        'exact_match': 'first',
        'alpha': 'mean'
    }).reset_index()

    # 将聚合后的 DataFrame 转换回 Hugging Face 的 Dataset
    aggregated_dataset = datasets.Dataset.from_pandas(aggregated_df)

    # 可选：如果有验证集或测试集，可以相应处理
    # aggregated_dataset = aggregated_dataset.train_test_split(test_size=0.1)

    # 应用评估函数
    # final_data = evaluate(aggregated_dataset)

    # 将最终数据集保存为 JSON 格式
    aggregated_dataset.to_json(output_path)


def evaluate(predictions):
    # evaluate the predictions with exact match
    def _normalize_answer(s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def exact_match_score(example):
        ground_truths = example['answers']
        assert type(ground_truths) == list, f'ground_truths is not a list, id:{example["id"]}, ground_truth:{ground_truths}'
        prediction = example['response']
        example['exact_match'] = 0
        if not prediction:
            print(f'no prediction for qid {example["qid"]}, {example["query"]}')
            return example
        for ground_truth in ground_truths:
            if _normalize_answer(ground_truth) in _normalize_answer(prediction):
                example['exact_match'] = 1
                break
        return example

    predictions = predictions.map(exact_match_score)
    return predictions


def evaluation_retrieval(input_file, retrieved_file, retrieval_model, llm_detector_model,
                            bm25_index, dense_index_name, dense_index_path, new_add_fact_score_file,
                            num_top_documents, added_documents, query_key='question'):
    """
    评估检索结果的各项指标。
    计算检索结果的相关性得分、BLEU 分数、LLM 概率和正确性概率。
    """
    # 步骤1：获取查询文本
    queries = {}
    for file in tqdm(input_file, total=len(input_file), desc="Reading input query files"):
        data = get_text(file)
        for doc in data:
            qid = str(doc['id'])
            queries[qid] = doc[query_key]

    # for all files under added_documents, read them into a huggingface dataset
    add_files = os.listdir(added_documents)
    # if jsonl file, read it into a dataset
    files = [f for f in add_files if f.endswith('.jsonl')]
    added_dataset = datasets.load_dataset('json', data_files=[added_documents + '/' + f for f in files])['train']

    output_ret_metrics_file = retrieved_file[0] + '_metrics.jsonl'
    logger.warning(f"output retrieval metrics file: {output_ret_metrics_file}")

    metrics_list = []  # 用于存储所有的度量结果

    for retrieval_file in retrieved_file:
        # 步骤2：获取平均相关性得分
        retrieved_docs_ori, retrieved_scores_ori = get_retrieved_documents(retrieval_file + '.trec')
        total_text_path = retrieval_file + '_eva_total_text.pkl'
        logger.warning(f"total text path: {total_text_path}, {os.path.exists(total_text_path)}")
        if os.path.exists(total_text_path):
            with open(total_text_path, 'rb') as f:
                total_text_source = pickle.load(f)
        else:
            total_text_source = get_total_doc_text(retrieved_docs_ori, added_dataset, bm25_index, task='filter_source')
            with open(total_text_path, 'wb') as f:
                pickle.dump(total_text_source, f)

        #compute mean retrieval score of top_num_documents, and average over all queries
        mean_retrieval_score = 0
        for qid in retrieved_docs_ori:
            mean_retrieval_score += np.mean(retrieved_scores_ori[qid][:num_top_documents])
        mean_retrieval_score /= len(retrieved_docs_ori)

        # 步骤3：获取detecter，factor score
        print("getting detector")
        detector = get_detector(llm_detector_model)

        total_llm_probs_path = retrieval_file + '_eva_total_llm_probs.pkl'
        logger.warning(f"total llm probs path: {total_llm_probs_path}, {os.path.exists(total_llm_probs_path)}")
        if os.path.exists(total_llm_probs_path):
            print("loading total llm probs")
            with open(total_llm_probs_path, 'rb') as f:
                total_llm_probs = pickle.load(f)
        else:
            print("computing total llm probs")
            total_llm_probs = compute_llm_probability(detector, queries, retrieved_docs_ori, total_text_source)
            with open(total_llm_probs_path, 'wb') as f:
                pickle.dump(total_llm_probs, f)
        # total_llm_probs = compute_llm_probability(detector, queries, llm_docs, total_text_source)
        mean_llm_prob = 0
        for qid in retrieved_docs_ori:
            mean_llm_prob += mean_llm_probability(total_llm_probs[qid], retrieved_docs_ori[qid][:num_top_documents])
        mean_llm_prob /= len(retrieved_docs_ori)

        fact_score = get_fact_dict(new_add_fact_score_file)
        mean_fact_score = 0
        for qid in retrieved_docs_ori:
            mean_fact_score += mean_fact_probability(fact_score, retrieved_docs_ori[qid][:num_top_documents])
        mean_fact_score /= len(retrieved_docs_ori)

        mean_bleu_score = 0
        for qid in retrieved_docs_ori:
            mean_bleu_score += compute_self_bleu_total(total_text_source[qid], retrieved_docs_ori[qid][:num_top_documents])
        mean_bleu_score /= len(retrieved_docs_ori)

        metric_json = {
            "retrieval_file": retrieval_file,
            "retrieval_score": mean_retrieval_score,
            "bleu_score": mean_bleu_score,
            "llm_prob": mean_llm_prob,
            "fact_score": mean_fact_score
        }
        metrics_list.append(metric_json)
        logger.warning(f"Added metrics for {retrieval_file}: {metric_json}")

    with open(output_ret_metrics_file, 'w') as f:
        for item in metrics_list:
            f.write(json.dumps(item) + '\n')


def set_weight(input_file, output_file):
    data = get_text(input_file)
    def process(doc):
        doc['alpha'] = 1.0
        return doc
    data = data.map(process, num_proc=20)
    data.to_json(output_file)

def set_random_weight(input_file, output_file):
    data = get_text(input_file)
    def process(doc):
        doc['alpha'] = random.uniform(0, 1)
        return doc
    data = data.map(process, num_proc=20)
    data.to_json(output_file)


def set_llm_weight(document, input_file, llm_detector_model, output_file, batch_size=256):
    """
    处理单个文档：计算初始权重并添加到索引中。
    """
    # 步骤1：获取文本生成伪查询
    print(f"Gathering pseudo queries for document")
    if not os.path.exists(input_file):
        pseudo_query_dataset = get_pseudo_query_dataset(document)
    else:
        pseudo_query_dataset = get_text(input_file)

    print("getting detector")
    detector = get_detector(llm_detector_model)

    queries = {}
    docs = {}
    for doc in pseudo_query_dataset:
        qid = doc['id']
        queries[qid] = doc['pseudo_query']
        docs[qid] = doc['response']

    paired_data = [(qid, queries[qid], docs[qid]) for qid in queries]
    llm_probs = {}

    total_batches = math.ceil(len(paired_data) / batch_size)
    # 使用 tqdm 显示总进度
    with tqdm(total=total_batches, desc="Computing LLM probabilities", unit="batch") as pbar:
        for i in range(0, len(paired_data), batch_size):
            batch = paired_data[i:i + batch_size]
            qids, queries_batch, doc_texts_batch = zip(*batch)

            # 创建输入格式
            inputs = [{"text": q, "text_pair": d} for q, d in zip(queries_batch, doc_texts_batch)]

            # 获取模型结果
            try:
                results = detector(inputs, max_length=512, truncation=True)
            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}")
                results = [None] * len(inputs)
            # [[{'label': 'LABEL_1', 'score': 0.9999973773956299}, {'label': 'LABEL_0', 'score': 2.6422230803291313e-06}],
            # [{'label': 'LABEL_0', 'score': 0.9999982118606567}, {'label': 'LABEL_1', 'score': 1.7340204294669093e-06}]]

            # 更新 llm_probs 字典
            for qid, result in zip(qids, results):
                llm_probs[qid] = result[0]['score'] if result[0]['label'] == 'LABEL_1' else result[1]['score']

            # 更新进度条
            pbar.update(1)

    # for doc in data:
    def process(doc):
        qid = doc['id']

        doc['alpha'] = 1-llm_probs[qid]

        # new_data.append(doc)
        # pool.close()
        return doc

    # 保存文档
    data = pseudo_query_dataset.map(process, num_proc=20)
    data.to_json(output_file)


def set_fact_weight(document, input_file, fact_score_file, output_file):
    """
    处理单个文档：计算初始权重并添加到索引中。
    """
    # 步骤1：获取文本生成伪查询
    print(f"Gathering pseudo queries for document")
    if not os.path.exists(input_file):
        pseudo_query_dataset = get_pseudo_query_dataset(document)
    else:
        pseudo_query_dataset = get_text(input_file)

    fact_score = get_fact_dict(fact_score_file)

    # for doc in data:
    def process(doc):
        qid = doc['id']
        if math.isnan(fact_score[qid]):
            fact_score[qid] = 0.0
        doc['alpha'] = fact_score[qid]

        # new_data.append(doc)
        # pool.close()
        return doc

    # 保存文档
    data = pseudo_query_dataset.map(process, num_proc=20)
    data.to_json(output_file)


def get_openai_api(api_base, api_key):
    openai.api_key = api_key
    openai.api_base = api_base

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    # read config file
    config_file_path = args.config_file_path
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    print(f'config: {config}')

    return config


def main(config_path: str):
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 提取配置参数
    task = config['task']
    retrieval_model = config["retrieval_model"]
    normalize_embeddings = config["normalize_embeddings"]
    index_name = config["index_name"]
    elasticsearch_url = config["elasticsearch_url"]
    pseudo_query_output = config["pseudo_query_output"]
    retrieved_file = config["retrieved_file"]
    output_file = config["output_file"]
    new_add_fact_score_file = config["new_add_fact_score_file"]
    llm_detector_model = config["llm_detector_model"]
    dense_index_name = config["dense_index_name"]
    dense_index_path = config["dense_index_path"]
    openai_api_base = config["openai_api_base"]
    openai_api_key = config["openai_api_key"]
    added_documents = config["added_documents"]
    optimization_config = config["optimization"]
    population_size = optimization_config.get("population_size", 50)
    generations = optimization_config.get("generations", 100)
    crossover_probability = optimization_config.get("crossover_probability", 0.7)
    mutation_probability = optimization_config.get("mutation_probability", 0.2)

    num_top_documents = config.get("num_top_documents", 20)
    with_alpha = config.get("with_alpha", True)

    if "fit_weights" in config:
        fit = config["fit_weights"]
        if fit == "9":
            fit_weights = None
        else:
            fit_weights = [1,1,1,1]
            fit_index = fit
            fit_weights[int(fit_index)] = 0.5
            logger.warning(f"fit weights: {fit_weights}")
    else:
        fit_weights = None

    # 设置 OpenAI API
    get_openai_api(openai_api_base, openai_api_key)

    # 获取索引
    index = get_index(elasticsearch_url, index_name)

    if task == "optimization":
        # 处理文档
        input_file = config["input_file"]
        process_document(input_file, pseudo_query_output, retrieved_file, retrieval_model, normalize_embeddings, llm_detector_model,
                         index, dense_index_name, dense_index_path, elasticsearch_url, with_alpha,
                         new_add_fact_score_file, num_top_documents, output_file, added_documents, population_size,
                         generations, crossover_probability, mutation_probability, fit_weights)
        evaluation_retrieval([pseudo_query_output], [retrieved_file], retrieval_model, llm_detector_model,
                                index, dense_index_name, dense_index_path,
                                new_add_fact_score_file, num_top_documents, added_documents, query_key='pseudo_query')

    elif task == "weight_update":
        input_query = config["input_query"]
        # 更新文档
        update_process_document(input_query, retrieved_file, retrieval_model, normalize_embeddings, llm_detector_model,
                                index, dense_index_name, dense_index_path, elasticsearch_url, with_alpha,
                                new_add_fact_score_file, num_top_documents, output_file, added_documents, population_size,
                                generations, crossover_probability, mutation_probability, fit_weights)
    elif task == "evaluation":
        # 处理文档
        input_file = config["input_file"]
        evaluation_retrieval(input_file, retrieved_file, retrieval_model, llm_detector_model,
                             index, dense_index_name, dense_index_path,
                             new_add_fact_score_file, num_top_documents, added_documents)
    elif task == "set_weight":
        # 设置权重
        input_file = config["input_file"]
        set_weight(input_file,  output_file)
    elif task == "set_random_weight":
        # 设置随机权重
        input_file = config["input_file"]
        set_random_weight(input_file,  output_file)
    elif task == "set_llm_weight":
        # 设置 LLM 权重
        input_file = config["input_file"]
        set_llm_weight(input_file, pseudo_query_output, llm_detector_model, output_file)
    elif task == "set_fact_weight":
        # 设置事实权重
        input_file = config["input_file"]
        set_fact_weight(input_file, pseudo_query_output, new_add_fact_score_file, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    # read config file
    config_file_path = args.config_file_path
    main(config_file_path)


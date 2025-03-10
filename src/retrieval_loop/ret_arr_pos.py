# 为什么写这个代码，因为如果能先对所有查询和生成文档进行检索，然后记录他们的相关性分数，
# 然后再在包含纯人类文本的索引中对查询进行检索，可以获得当前的人类文档检索列表
# 这样当我们计算完成生成文本的新权重后，对于该文档所有的相关性分数乘这个权重，就是当前的相关性得分
# 此时我们可以根据这个相关性得分，将其插入到原来的检索列表中，获得新的检索列表
# 这样我们就可以省去索引的改变时间

import os
import sys
import json
import numpy as np
import pandas as pd
import time
import datasets
from retrieve_methods import Retrieval, load_retrieval_embeddings
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
sys.path.append('../llm_zero_generate')
from tqdm import tqdm
import logging
from evaluate_dpr_retrieval import has_answers, SimpleTokenizer, evaluate_retrieval
from get_init_weight import get_total_doc_text, get_index
import random
import argparse
import copy
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import re


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# 设置 Python 内置随机种子
random.seed(42)

# 设置 NumPy 随机种子
np.random.seed(42)

# 第一步，获取所有查询对当前人类索引的top200检索结果
def retrieve_documents(query_file, retrieval_model, normalize_embeddings,
                       dense_index_name, dense_index_path, elasticsearch_url, with_alpha, retrieved_file,
                       page_content_column='question'):
    Retrieval(query_file, page_content_column, retrieval_model, dense_index_name, dense_index_path, normalize_embeddings,
              retrieved_file, elasticsearch_url, with_alpha)


# 读取检索文件
def get_retrieved_documents(retreived_file):
    # analyze trec file
    with open(retreived_file, 'r') as f:
        lines = f.readlines()
    retrieved_docs = {}
    retrieved_scores = {}
    for line in lines:
        qid, _, pid, _, score, tag = line.strip().split(" ")
        qid = str(qid)
        if qid not in retrieved_docs:
            retrieved_docs[qid] = []
            retrieved_scores[qid] = []
        retrieved_docs[qid].append(pid)
        retrieved_scores[qid].append(float(score))
    return retrieved_docs, retrieved_scores, tag


def get_text(input_file):
    """
    从文档中提取文本。
    jsonl文件，每行：{"id": "baichuan2-13b-chat_nq_from_bge-base_None_loop4_21_20231229122730", "question": "right to property according to the constitution of india is a?", "answers": ["constitutional right"], "response": "Right to Property According to Constitution of India", "exact_match": 0}

    """
    dataset = datasets.load_dataset('json', data_files=input_file)['train']
    return dataset


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

        return query_embeddings, embeddings, query_instruction, doc_instruction

    else:
        embeddings = load_retrieval_embeddings(retrieval_model, normalize_embeddings=normalize_embeddings)
        print(f'loaded query embedder: {retrieval_model}')

        return embeddings, embeddings, query_instruction, doc_instruction


def compute_similarity_score(query_dataset, doc_dataset, query_embeddings, doc_embeddings, query_instruction, doc_instruction,  query_key='question'):
    sim_scores = []
    qids = []
    dids = []
    queries = []
    docs = []
    pd_dict = {}
    dp_dict = {}
    for query in query_dataset:
        qid = query['id']
        qids.append(qid)
        queries.append(query_instruction + query[query_key])
    for doc in doc_dataset:
        if 'id' in doc:
            did = doc['id']
        else:
            did = doc['docid']
        dids.append(did)
        docs.append(doc_instruction + doc['response'])

    # 设置批处理大小
    batch_size = 128
    num_queries = len(qids)
    num_docs = len(dids)

    # 初始化 pd_dict
    for qid in qids:
        pd_dict[qid] = {}

    # 生成文档嵌入（分批处理并显示进度条）
    print("计算文档嵌入向量...")
    doc_vecs = []
    for i in tqdm(range(0, num_docs, batch_size), desc="嵌入文档批次", unit="batch"):
        batch_docs = docs[i:i + batch_size]
        batch_vecs = doc_embeddings.embed_documents(batch_docs)

        # 确保嵌入向量是 NumPy 数组
        if not isinstance(batch_vecs, np.ndarray):
            batch_vecs = np.array(batch_vecs)

        doc_vecs.append(batch_vecs)

    # 将所有文档嵌入合并成一个数组
    doc_vecs = np.vstack(doc_vecs)
    assert doc_vecs.shape[0] == num_docs, f"文档嵌入数量 {doc_vecs.shape[0]} 与文档数量 {num_docs} 不匹配。"
    print("文档嵌入向量计算完成。")

    # 生成查询嵌入和计算相似度（分批处理并显示进度条）
    print("计算查询嵌入向量和余弦相似度...")
    for start_idx in tqdm(range(0, num_queries, batch_size), desc="处理查询批次", unit="batch"):
        end_idx = min(start_idx + batch_size, num_queries)
        query_batch = queries[start_idx:end_idx]
        qid_batch = qids[start_idx:end_idx]

        # 生成当前批次的查询嵌入
        query_vecs = query_embeddings.embed_queries(query_batch)
        if not isinstance(query_vecs, np.ndarray):
            query_vecs = np.array(query_vecs)

        # 计算当前批次与所有文档的余弦相似度
        sim_batch = linear_kernel(query_vecs, doc_vecs)  # 形状: (current_batch_size, num_docs)

        # 将相似度分数存储到 pd_dict
        for i, qid in enumerate(qid_batch):
            scores = sim_batch[i]
            pd_dict[qid] = dict(zip(dids, scores))


        # 可选：如果希望在处理每个查询批次时显示更详细的进度，可以在内部添加另一个 tqdm

    print("余弦相似度计算完成。")
    # 计算dp_dict
    for did in dids:
        dp_dict[did] = {}
    for qid in qids:
        for did in dids:
            dp_dict[did][qid] = pd_dict[qid][did]

    return pd_dict, dp_dict


def compute_q_llm_score(input_files, retrieval_model, normalize_embeddings, added_documents, output_file):
    q_embedding, d_embedding, query_instruction, doc_instruction = get_embedding(retrieval_model, normalize_embeddings)
    # for all files under added_documents, read them into a huggingface dataset
    add_files = os.listdir(added_documents)
    # if jsonl file, read it into a dataset
    files = [f for f in add_files if f.endswith('.jsonl')]
    doc_dataset = datasets.load_dataset('json', data_files=[added_documents + '/' + f for f in files])['train']

    query_dataset = get_text(input_files)

    sim_scores, dp_scores = compute_similarity_score(query_dataset, doc_dataset, q_embedding, d_embedding, query_instruction, doc_instruction)

    # save the sim_scores to a file
    output_sim_file = output_file + '_sim.json'
    output_dp_file = output_file + '_dp.json'
    with open(output_sim_file, 'w') as f:
        json.dump(sim_scores, f)
    with open(output_dp_file, 'w') as f:
        json.dump(dp_scores, f)


def add_doc_to_list(other_llm_retrieved_docs, other_llm_retrieved_scores, retrieved_docs_ori, retrieved_scores_ori,
                    weight_docs, weight_scores, qids, top_k=100):
    import copy

    # 创建深拷贝以避免修改原始数据
    new_retrieved_docs = copy.deepcopy(retrieved_docs_ori)
    new_retrieved_scores = copy.deepcopy(retrieved_scores_ori)

    # 遍历每个查询ID
    for qid in qids:
        # 获取原始文档和分数列表，如果不存在则初始化为空列表
        ori_doc_list = new_retrieved_docs.get(qid, []).copy()
        ori_score_list = new_retrieved_scores.get(qid, []).copy()

        # 获取其他LLM检索到的文档和分数
        other_docs = other_llm_retrieved_docs.get(qid, [])
        other_scores = other_llm_retrieved_scores.get(qid, [])

        # 获取加权文档和分数
        weight_docs_list = weight_docs.get(qid, [])
        weight_scores_list = weight_scores.get(qid, [])

        # 合并所有文档
        all_docs = ori_doc_list + other_docs + weight_docs_list
        all_scores = ori_score_list + other_scores + weight_scores_list

        # 对所有文档进行排序
        sorted_pairs = sorted(zip(all_docs, all_scores), key=lambda x: x[1], reverse=True)
        sorted_docs, sorted_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])

        new_retrieved_docs[qid] = sorted_docs[:top_k]
        new_retrieved_scores[qid] = sorted_scores[:top_k]

    return new_retrieved_docs, new_retrieved_scores


def get_updated_retrieved_docs(q_llm_file, retrieval_file_ori, retrieval_file_now, weights_file, output_file, query_file,
                               added_documents, bm25_index, top_k=100):
    # Load the query-LLM similarity scores
    with open(q_llm_file, 'r') as f:
        q_llm_scores = json.load(f)

    # get weighted scores
    weights = get_text(weights_file)
    query_dataset = get_text(query_file)
    q_ans = {}
    q_text = {}
    for query in query_dataset:
        qid = query['id']
        q_ans[qid] = query['answer']
        q_text[qid] = query['question']
    # Load the original retrieved documents and scores
    retrieved_docs_ori, retrieved_scores_ori, _ = get_retrieved_documents(retrieval_file_ori+'.trec')

    retrieved_docs_now, retrieved_scores_now, tag = get_retrieved_documents(retrieval_file_now+'.trec')
    qids = list(retrieved_docs_now.keys())
    weight_docs = {}
    weight_scores = {}
    for w in tqdm(weights, total=len(weights), desc="Processing weights"):
        did = w['id']
        weight = w['alpha']
        doc_scores = copy.deepcopy(q_llm_scores[did])
        for q in doc_scores:
            if q in qids:
                if q not in weight_docs:
                    weight_docs[q] = []
                    weight_scores[q] = []
                weight_docs[q].append(did)
                weight_scores[q].append(weight * doc_scores[q])

    # Get the documents that are not in the original retrieval list
    other_llm_retrieved_docs, other_llm_retrieved_scores = get_other_llm_score(retrieved_docs_now, retrieved_scores_now, weight_docs)

    final_retrieved_docs, final_retrieved_scores = add_doc_to_list(other_llm_retrieved_docs, other_llm_retrieved_scores,
                                                                   retrieved_docs_ori, retrieved_scores_ori,
                                                                     weight_docs, weight_scores, qids, top_k)

    # Save the updated retrieval list
    output_trec_file = output_file + '.trec'
    output_json_file = output_file + '.json'
    tag_now = tag + '_updated'
    with open(output_trec_file, 'w') as f:
        for qid in final_retrieved_docs:
            for i, doc in enumerate(final_retrieved_docs[qid]):
                f.write(f"{qid} Q0 {doc} {i+1} {final_retrieved_scores[qid][i]} {tag_now}\n")

    # 初始化检索结果字典
    # for all files under added_documents, read them into a huggingface dataset
    add_files = os.listdir(added_documents)
    # if jsonl file, read it into a dataset
    files = [f for f in add_files if f.endswith('.jsonl')]

    added_dataset = datasets.load_dataset('json', data_files=[added_documents + '/' + f for f in files])['train']
    total_text_source = get_total_doc_text(final_retrieved_docs, added_dataset, bm25_index, task='filter_source')
    retrieval = {}
    for qid in final_retrieved_docs:
        retrieval[qid] = {
            'question': q_text[qid],
            'answers': q_ans[qid],
            'contexts': []
        }
        for doc_id, score in zip(final_retrieved_docs[qid], final_retrieved_scores[qid]):
            # 假设有一个函数 get_document_content(doc_id) 获取文档内容
            # 需要根据具体存储方式实现
            text = total_text_source[qid][doc_id]  # 假设 get_document_content(doc_id) 返回文档内容

            # 判断答案是否存在于文档中
            tokenizer = SimpleTokenizer()  # 假设 SimpleTokenizer 已定义
            has_ans = has_answers(text, retrieval[qid]['answers'], tokenizer, False)

            retrieval[qid]['contexts'].append({
                'docid': doc_id,
                'score': float(score),
                'has_answer': has_ans
            })
    # 保存 JSON 文件
    with open(output_json_file, 'w', encoding='utf-8') as f_json:
        json.dump(retrieval, f_json, indent=4, ensure_ascii=False)

    print(f"更新后的 TREC 文件已保存至 {output_trec_file}")
    print(f"更新后的 JSON 文件已保存至 {output_json_file}")
    print(f'evaluating {query_file}')
    evaluate_retrieval(output_json_file, [5, 20, 100], False)

def get_other_llm_score(retreived_docs, retrieved_scores, weight_score_dict):
    new_retrieved_docs = {}
    new_retrieved_scores = {}
    for q in retreived_docs:
        for d in retreived_docs[q]:
            if d.isdigit():
                continue
            if d in weight_score_dict[q]:
                continue
            if q not in new_retrieved_docs:
                new_retrieved_docs[q] = []
                new_retrieved_scores[q] = []
            new_retrieved_docs[q].append(d)
            new_retrieved_scores[q].append(retrieved_scores[q][retreived_docs[q].index(d)])
    return new_retrieved_docs, new_retrieved_scores


def get_sorted_file_paths(weight_init_path, weight_update_path, loop):
    """
    获取排序后的文件路径列表，优先级为：
    init_output/loop0 < update_output/loop0 < init_output/loop1 < update_output/loop1 < ...
    """
    file_pattern = re.compile(r'loop(\d+)_merged_weight\.json')

    files_with_priority = []

    for i in range(loop):
        # 初始化路径文件
        init_file = os.path.join(weight_init_path, f'loop{i}_merged_weight.json')
        if os.path.isfile(init_file):
            # 优先级为 2*i
            files_with_priority.append((2 * i, init_file))

        # 更新路径文件
        update_file = os.path.join(weight_update_path, f'loop{i}_merged_weight.json')
        if os.path.isfile(update_file):
            # 优先级为 2*i + 1
            files_with_priority.append((2 * i + 1, update_file))

    # 按优先级排序，优先级低的先处理
    sorted_files = sorted(files_with_priority, key=lambda x: x[0])
    return [file_path for _, file_path in sorted_files]


def load_data(sorted_file_paths):
    """
    读取文件并根据优先级合并数据，处理重复的 id。
    """
    data_dict = {}

    for file_path in sorted_file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    record_id = record.get('id')
                    if record_id:
                        # 由于文件是按优先级排序的，后出现的会覆盖前面的
                        data_dict[record_id] = record
                except json.JSONDecodeError:
                    print(f"JSON decode error in file: {file_path}, line: {line}")

    return list(data_dict.values())


def compute_pq_llm_score(merged_data, retrieval_model, normalize_embeddings, input_file):
    # 获取查询和文档的嵌入
    query_embeddings, doc_embeddings, query_instruction, doc_instruction = get_embedding(retrieval_model, normalize_embeddings)

    # 读取查询和文档
    query_dataset = get_text(input_file)

    # 计算查询和文档之间的相似度
    sim_scores, dp_scores = compute_similarity_score(query_dataset, merged_data, query_embeddings, doc_embeddings, query_instruction, doc_instruction, query_key='pseudo_query')

    # sim_scores相关性分数乘以权重alpha
    # 创建一个字典映射文档ID到其alpha权重
    doc_alpha = {}
    for doc in merged_data:
        doc_id = doc.get('id') or doc.get('docid')  # 根据你的数据结构选择正确的键
        alpha = doc.get('alpha', 1.0)  # 默认为1.0，如果没有提供alpha
        doc_alpha[doc_id] = alpha

    # 计算加权相似度分数（sim_scores 为 pd_dict，映射 query_id -> {doc_id: score, ...}）
    weighted_docs = {}
    weighted_scores = {}
    for qid, doc_scores in sim_scores.items():
        weighted_scores[qid] = []
        weighted_docs[qid] = []
        for did, score in doc_scores.items():
            alpha = doc_alpha.get(did, 1.0)
            weighted_score = score * alpha
            weighted_docs[qid].append(did)
            weighted_scores[qid].append(weighted_score)

    return weighted_scores, weighted_docs


def retrieve_pq_llm_documents(input_file, retrieval_model, normalize_embeddings, dense_index_name, dense_index_path,
                                elasticsearch_url, with_alpha, retrieved_file, weight_init_path=None, weight_update_path=None,
                                added_documents=None, output_file=None,
                                loop=0, index=None, num_top_documents=100, page_content_column='pseudo_query'):

    if not os.path.exists(retrieved_file):
        logger.info(f"Retrieving documents for pseudo queries in {input_file}...")
        retrieve_documents([input_file], retrieval_model, normalize_embeddings, dense_index_name, dense_index_path,
                           elasticsearch_url, with_alpha, [retrieved_file], page_content_column='pseudo_query')
    else:
        logger.info(f"Retrieved documents for pseudo queries already exist in {retrieved_file}.")
    sorted_file_paths = get_sorted_file_paths(weight_init_path, weight_update_path, loop)
    print(f"排序后的文件列表: {sorted_file_paths}")

    if sorted_file_paths:
        merged_data = load_data(sorted_file_paths)
        print(f"合并后的数据条目数: {len(merged_data)}")

        weighted_llm_scores, weighted_llm_docs = compute_pq_llm_score(merged_data, retrieval_model, normalize_embeddings,
                                                                      input_file)
        retrieved_docs, retrieved_scores, tag = get_retrieved_documents(retrieved_file + '.trec')
        final_retrieved_docs, final_retrieved_scores = add_doc_to_list(weighted_llm_docs, weighted_llm_scores,
                                                                       retrieved_docs, retrieved_scores, {}, {},
                                                                       list(retrieved_docs.keys()), num_top_documents)
    else:
        final_retrieved_docs, final_retrieved_scores, tag = get_retrieved_documents(retrieved_file + '.trec')

    output_trec_file = output_file + '.trec'
    output_json_file = output_file + '.json'
    tag_now = tag + '_pq_llm_updated'
    with open(output_trec_file, 'w') as f:
        for qid in final_retrieved_docs:
            for i, doc in enumerate(final_retrieved_docs[qid]):
                f.write(f"{qid} Q0 {doc} {i + 1} {final_retrieved_scores[qid][i]} {tag_now}\n")

    # 初始化检索结果字典
    # for all files under added_documents, read them into a huggingface dataset

    add_files = os.listdir(added_documents)
    # if jsonl file, read it into a dataset
    files = [f for f in add_files if f.endswith('.jsonl')]
    added_dataset = datasets.load_dataset('json', data_files=[added_documents + '/' + f for f in files])['train']
    total_text_source = get_total_doc_text(final_retrieved_docs, added_dataset, index, task='filter_source')

    query_dataset = get_text(input_file)
    q_ans = {}
    q_text = {}
    for query in query_dataset:
        qid = query['id']
        # wether answer or answers
        q_ans[qid] = query.get('answers')
        q_text[qid] = query['question']

    retrieval = {}
    for qid in final_retrieved_docs:
        retrieval[qid] = {
            'question': q_text[qid],
            'answers': q_ans[qid],
            'contexts': []
        }
        for doc_id, score in zip(final_retrieved_docs[qid], final_retrieved_scores[qid]):
            # 假设有一个函数 get_document_content(doc_id) 获取文档内容
            # 需要根据具体存储方式实现
            text = total_text_source[qid][doc_id]  # 假设 get_document_content(doc_id) 返回文档内容

            # 判断答案是否存在于文档中
            tokenizer = SimpleTokenizer()  # 假设 SimpleTokenizer 已定义
            has_ans = has_answers(text, retrieval[qid]['answers'], tokenizer, False)

            retrieval[qid]['contexts'].append({
                'docid': doc_id,
                'score': float(score),
                'has_answer': has_ans
            })
    # 保存 JSON 文件
    with open(output_json_file, 'w', encoding='utf-8') as f_json:
        json.dump(retrieval, f_json, indent=4, ensure_ascii=False)

    print(f"更新后的 TREC 文件已保存至 {output_trec_file}")
    print(f"更新后的 JSON 文件已保存至 {output_json_file}")
    # evaluate
    print(f'evaluating {input_file}')
    evaluate_retrieval(output_json_file, [5, 20, 100], False)

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
    output_file = config["output_file"]
    dense_index_name = config["dense_index_name"]
    dense_index_path = config["dense_index_path"]
    added_documents = config["added_documents"]
    num_top_documents = config.get("num_top_documents", 20)
    with_alpha = config.get("with_alpha", True)



    # 获取索引
    index = get_index(elasticsearch_url, index_name)

    if task == "retrieve":
        # 处理文档 retrieved_file
        # 处理伪查询文档，输入：伪查询文件、索引、在此前已加入的文档，输出：检索文件
        # 对伪查询进行检索，然后计算伪查询与新加入文档的加权相关性分数，然后合并列表
        input_file = config["input_file"]
        retrieved_file = config["retrieved_file"]
        loop = int(config["loop"])
        weight_init_path = config["weight_init_path"]
        weight_update_path = config["weight_update_path"]
        retrieve_pq_llm_documents(input_file, retrieval_model, normalize_embeddings, dense_index_name, dense_index_path,
                                  elasticsearch_url, with_alpha, retrieved_file, weight_init_path, weight_update_path,
                                    added_documents, output_file, loop, index, num_top_documents)

    elif task == "compute_q_llm_score":
        # 计算查询和文档之间的相似度 output_file
        input_files = config["input_file"]
        compute_q_llm_score(input_files, retrieval_model, normalize_embeddings, added_documents, output_file)

    elif task == "get_weighted_results":
        # 获取加权结果 weights_file, retrieval_file_ori, retrieval_file_now, output_file
        weights_file = config["weights_file"]
        retrieval_file_oris = config["retrieval_file_ori"]
        retrieval_file_nows = config["retrieval_file_now"]
        output_files = config["output_file"]
        query_files = config["query_file"]
        q_llm_file = config["q_llm_file"]
        for retrieval_file_ori, retrieval_file_now, output_file, query_file in zip(retrieval_file_oris, retrieval_file_nows, output_files, query_files):
            get_updated_retrieved_docs(q_llm_file + '_dp.json', retrieval_file_ori, retrieval_file_now, weights_file, output_file,
                                       query_file, added_documents, index, num_top_documents)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    # read config file
    config_file_path = args.config_file_path
    main(config_file_path)
from langchain.document_loaders import HuggingFaceDatasetLoader
# from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
import os
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
import elasticsearch
from langchain.vectorstores import ElasticsearchStore
from langchain.vectorstores.utils import DistanceStrategy
from retrieve_methods import Retrieval
import argparse
import json
from tqdm import tqdm
import torch
import sys
from math import ceil
import time
import numpy as np
import pickle
import linecache
from faiss import IndexFlatIP, IndexIVFFlat
import faiss
# from langchain.vectorstores import InMemoryDocstore
# 测试路径：1. BM25索引建立、检索 2. BM25新增文档 3. BM25删除文档
# 4. dense索引建立、检索（有权重） 5. dense新增文档（有权重） 6. dense更新权重（有权重） 7. dense删除文档（有权重） 8. dense更新原索引权重（有权重）
# 9. dense索引建立、检索（无权重） 10. dense新增文档（无权重） 11. dense删除文档（无权重） 

sys.stdout.flush()
BATCH_SIZE = 512
# 1. 建索引和新增文档时扩展向量（建立docid->indexid，indexid->docid, docid->original_emb字典）
# 2. 对于已存在于索引的文档，检索回来后更新权重（先，先找到indexid然后删除，再插入相同id的修改emb的文档，记录原文档中被修改的indexid）
# 3. 一轮实验结束后，删除新增文档以及三个dict中对应数据，还原原索引中被修改的文档
# 需要新增的函数： 1. 新增向量：编码，根据权重计算新向量，插入索引，记录docid->indexid, indexid->docid, docid->original_emb
# 2. 更新权重：根据docid找到indexid，删除indexid，根据给定权重计算并插入新向量，记录被修改的indexid

def get_args():
    # get config_file_path, which is the path to the config file
    # config file formatted as a json file:
    # {
    #   "new_text_file": "../../data_v2/zero_gen_data/DPR/nq-test-gen-baichuan2-13b-chat.jsonl",
    #   "retrieval_method": "DPR", # BM25, DPR, Contriever, RetroMAE, all-mpnet, BGE, LLM-Embedder
    #   "index_name": "DPR_faiss_index",
    #   "index_path": "../../data_v2/indexes",
    #   "page_content_column": "question"
    # }
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    # read config file
    config_file_path = args.config_file_path
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    print(f'config: {config}')

    return config


def load_retrieval_embeddings(retrieval_model, normalize_embeddings=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings,
                     'batch_size': BATCH_SIZE,
                     'show_progress_bar': True}
    embeddings = HuggingFaceEmbeddings(model_name=retrieval_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    return embeddings


def load_new_text(new_text_file, page_content_column):
    loader = HuggingFaceDatasetLoader('json', data_files=new_text_file,
                                      page_content_column=page_content_column)
    new_text = loader.load()
    print(f'长度 of new text: {len(new_text)}')
    print(f'new text 格式: {new_text[0]}')
    return new_text


def map_retrieval_model(retrieval_model):
    model_mapping = {
        'DPR': '../../../../../Rob_LLM/ret_model/DPR/facebook-dpr-ctx_encoder-multiset-base',
        'CONTRIEVER': '../../../../../Rob_LLM/ret_model/contriever-base-msmarco',
        'RETROMAE': '../../../../../Rob_LLM/ret_model/RetroMAE_BEIR',
        'ALL-MPNET': '../../../../../Rob_LLM/ret_model/all-mpnet-base-v2',
        'BGE-LARGE': '../../../../../Rob_LLM/ret_model/bge-large-en-v1.5',
        'BGE-BASE': '../../../../../Rob_LLM/ret_model/bge-base-en-v1.5',
        'LLM-EMBEDDER': '../../../../../Rob_LLM/ret_model/llm-embedder',
        'BM25': 'BM25'
    }
    upper_model = retrieval_model.upper()
    instruction = ""
    if upper_model in model_mapping:
        mapped_model = model_mapping[upper_model]
        if upper_model == 'LLM-EMBEDDER':
            instruction = "Represent this document for retrieval: "
        return mapped_model, instruction
    else:
        raise ValueError(f'未知的检索模型: {retrieval_model}')


def load_embeddings(retrieval_model, normalize_embeddings):
    if retrieval_model != "BM25":
        embeddings = load_retrieval_embeddings(retrieval_model, normalize_embeddings=normalize_embeddings)
        print(f'已加载检索嵌入: {retrieval_model}')
        return embeddings
    else:
        print('请确保已启动 elasticsearch 服务器')
        return None


def process_texts(new_text, instruction):
    if instruction:
        for doc in tqdm(new_text, desc='添加指令到新文本'):
            doc.page_content = instruction + doc.page_content
    print(f'处理后的新文本格式: {new_text[0]}')
    return new_text


def validate_documents(new_text):
    print("验证文档中的 'alpha' 字段...")
    for idx, doc in enumerate(tqdm(new_text, desc='验证文档')):
        if 'alpha' not in doc.metadata:
            raise ValueError(f"文档索引 {idx} 缺少 'alpha' 字段。")
        if not (0.0 <= doc.metadata['alpha'] <= 1.0):
            raise ValueError(
                f"文档 ID {doc.metadata.get('id', '未知')} 的 'alpha' 值 {doc.metadata['alpha']} 不在 [0, 1] 范围内。")
    print("文档验证通过。")


def encode_embeddings(new_text, embeddings):
    modified_embeddings_list = []
    original_embeddings = []

    num_batches = ceil(len(new_text) / BATCH_SIZE)
    print(f'将文档编码为嵌入，分为 {num_batches} 个批次...')

    for batch_idx in tqdm(range(num_batches), desc='处理批次'):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min((batch_idx + 1) * BATCH_SIZE, len(new_text))
        batch_docs = new_text[batch_start:batch_end]

        # 编码文档，获取原始嵌入
        original_emb_batch = embeddings.embed_documents([doc.page_content for doc in batch_docs])
        original_emb_batch = np.array(original_emb_batch, dtype=np.float32)  # 根据需求调整数据类型

        # 获取 alpha 值
        alphas = np.array([doc.metadata['alpha'] for doc in batch_docs], dtype=np.float32)  # shape: (batch_size,)

        # 计算嵌入范数
        embedding_norms = np.linalg.norm(original_emb_batch, axis=1)  # shape: (batch_size,)

        # 计算新的维度值
        new_dimensions = np.sqrt(1 - np.square(alphas)) * embedding_norms  # shape: (batch_size,)

        # 修改嵌入：alpha * e
        modified_embeddings = original_emb_batch * alphas[:, np.newaxis]  # shape: (batch_size, emb_dim)

        # 追加新维度
        modified_embeddings = np.hstack([modified_embeddings, new_dimensions[:, np.newaxis]]).astype(
            np.float32)  # shape: (batch_size, emb_dim + 1)

        # 追加到列表
        modified_embeddings_list.append(modified_embeddings)
        original_embeddings.append(original_emb_batch)


    # 合并所有修改后的嵌入和原始嵌入
    all_modified_embeddings = np.vstack(modified_embeddings_list)
    all_original_embeddings = np.vstack(original_embeddings)
    print(f'所有修改后的嵌入形状: {all_modified_embeddings.shape}')  # (num_docs, emb_dim + 1)
    print(f'所有原始的嵌入形状: {all_original_embeddings.shape}')  # (num_docs, emb_dim)

    # 准备文本与嵌入对
    text_embedding_pairs = [
        (doc.page_content, all_modified_embeddings[idx].tolist(), doc.metadata)
        for idx, doc in enumerate(new_text)
    ]

    # 准备 docid 列表
    docid_list = [doc.metadata['id'] for doc in new_text]

    # 准备原始嵌入映射
    docid_original_embedding = {
        docid_list[idx]: all_original_embeddings[idx].tolist()
        for idx in range(len(docid_list))
    }

    return text_embedding_pairs, docid_original_embedding, docid_list


def save_original_embeddings(docid_original_embedding, embedding_storage_path, index_exist):
    """
    将 {docid: original_embedding} 保存到 docid_to_original_emb.pkl。
    """
    original_embeddings = load_pickle(embedding_storage_path, index_exist)
    original_embeddings.update(docid_original_embedding)
    save_pickle(embedding_storage_path, original_embeddings)


def save_docid_to_indexid(docid_to_indexid_mapping, mapping_storage_path, index_exist):
    """
    将 {docid: indexid} 保存到 docid_to_indexid.pkl。
    """
    docid_to_indexid = load_pickle(mapping_storage_path, index_exist)
    docid_to_indexid.update(docid_to_indexid_mapping)
    save_pickle(mapping_storage_path, docid_to_indexid)


def save_indexid_to_docid(indexid_to_docid_mapping, mapping_storage_path, index_exist):
    """
    将 {indexid: docid} 保存到 indexid_to_docid.pkl。
    """
    indexid_to_docid = load_pickle(mapping_storage_path, index_exist)
    indexid_to_docid.update(indexid_to_docid_mapping)
    save_pickle(mapping_storage_path, indexid_to_docid)


def update_faiss_index_without_alpha(index_path, index_name, retrieval_model, new_text, embeddings):
    index_p = os.path.join(index_path, index_name)
    index = FAISS.load_local(index_p, embeddings)
    print(f'已加载 {retrieval_model} 索引: {index_name}, 当前长度: {len(index.docstore._dict)}')
    print('正在添加新文本到索引...')

    # 添加嵌入到 FAISS 索引，获取新分配的 indexid
    new_ids = index.add_documents(new_text)
    
    index.save_local(index_p)
    print(f'已添加 {len(new_ids)} 条新文本到索引: {index_name}, 当前长度: {len(index.docstore._dict)}')

    # 获取 docid 列表
    docids = [doc.metadata['id'] for doc in new_text]

    # 确保 new_ids 和 docids 顺序一致
    if len(new_ids) != len(docids):
        raise ValueError("new_ids 和 docids 的长度不一致。")

    # 创建 {docid: indexid} 和 {indexid: docid} 映射
    docid_to_indexid_mapping = dict(zip(docids, new_ids))
    indexid_to_docid_mapping = dict(zip(new_ids, docids))

    return docid_to_indexid_mapping, indexid_to_docid_mapping, len(index.docstore._dict)


def update_faiss_index(index_path, index_name, retrieval_model, text_embedding_pairs, metadatas, embeddings):
    index_p = os.path.join(index_path, index_name)
    index = FAISS.load_local(index_p, embeddings)
    print(f'已加载 {retrieval_model} 索引: {index_name}, 当前长度: {len(index.docstore._dict)}')
    print('正在添加新文本到索引...')

    # 添加嵌入到 FAISS 索引，获取新分配的 indexid
    new_ids = index.add_embeddings(
        [(text, embedding) for text, embedding, _ in text_embedding_pairs],
        metadatas=[metadata for _, _, metadata in text_embedding_pairs]
    )
    index.save_local(index_p)
    print(f'已添加 {len(new_ids)} 条新文本到索引: {index_name}, 当前长度: {len(index.docstore._dict)}')

    # 获取 docid 列表
    docids = [metadata['id'] for _, _, metadata in text_embedding_pairs]

    # 确保 new_ids 和 docids 顺序一致
    if len(new_ids) != len(docids):
        raise ValueError("new_ids 和 docids 的长度不一致。")

    # 创建 {docid: indexid} 和 {indexid: docid} 映射
    docid_to_indexid_mapping = dict(zip(docids, new_ids))
    indexid_to_docid_mapping = dict(zip(new_ids, docids))

    return docid_to_indexid_mapping, indexid_to_docid_mapping, len(index.docstore._dict)


def update_bm25_index(elasticsearch_url, index_name, new_text):
    index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name)
    index_size = index.get_document_count()
    print(f'已加载 BM25 索引: {index_name}, 当前长度: {index_size}')
    print('正在添加新文本到索引...')

    new_ids = index.add_texts(new_text)
    index_size = index.get_document_count()
    print(f'已添加 {len(new_ids)} 条新文本到索引: {index_name}, 当前长度: {index_size}')

    # 假设 new_ids 与 new_text 顺序一致，创建 {docid: indexid} 和 {indexid: docid} 映射
    docids = [doc.metadata['id'] for doc in new_text]
    docid_to_indexid_mapping = dict(zip(docids, new_ids))
    indexid_to_docid_mapping = dict(zip(new_ids, docids))

    return docid_to_indexid_mapping, indexid_to_docid_mapping, index_size


def log_new_ids(index_log_path, index_name, new_ids):
    index_log_name = f'{index_name}_{time.strftime("%Y%m%d-%H%M%S")}.log'
    index_log_file = os.path.join(index_log_path, index_log_name)
    with open(index_log_file, 'w', encoding='utf-8') as f:
        for id in new_ids:
            f.write(f'{id}\n')
    print(f'已创建索引日志文件: {index_log_file}')


def create_or_update_index_without_alpha(index_exists, index_name, index_path, retrieval_model,
                            elasticsearch_url, embeddings, new_text, index_log_path):
    if index_exists:
        print(f'索引: {index_name} 已存在')
        if retrieval_model != "BM25":
            docid_to_indexid_mapping, indexid_to_docid_mapping, index_size = update_faiss_index_without_alpha(
                index_path, index_name, retrieval_model, new_text, embeddings)
        else:
            docid_to_indexid_mapping, indexid_to_docid_mapping, index_size = update_bm25_index(
                elasticsearch_url, index_name, new_text)
        # 记录日志
        log_new_ids(index_log_path, index_name, list(docid_to_indexid_mapping.values()))
    else:
        print(f'索引: {index_name} 不存在，正在创建 {retrieval_model} 索引')
        if retrieval_model != "BM25":
            index_p = os.path.join(index_path, index_name)
            index = FAISS.from_documents(
                new_text,
                embeddings,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
            index.save_local(index_p)
            index_size = len(index.docstore._dict)
            print(f'已创建 {retrieval_model} 索引: {index_name}, 当前长度: {index_size}')

            # 获取 docid 和 indexid 并创建映射
            docids = [doc.metadata['id'] for doc in new_text]
            index_ids = list(index.docstore._dict.keys())
            if len(docids) != len(index_ids):
                raise ValueError
            docid_to_indexid_mapping = dict(zip(docids, index_ids))
            indexid_to_docid_mapping = dict(zip(index_ids, docids))
        else:
            index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name, overwrite_existing_index=True)
            index_size = index.get_document_count()
            print(f'已创建 BM25 索引: {index_name}, 当前长度: {index_size}')
            print('正在添加新文本到索引...')
            new_ids = index.add_texts(new_text)
            index_size = index.get_document_count()
            print(f'已添加 {len(new_text)} 条新文本到索引: {index_name}, 当前长度: {index_size}')
            docids = [doc.metadata['id'] for doc in new_text]
            docid_to_indexid_mapping = dict(zip(docids, new_ids))
            indexid_to_docid_mapping = dict(zip(new_ids, docids))
    print(f'已创建或更新 {retrieval_model} 索引: {index_name}, 当前长度: {index_size}')
    return docid_to_indexid_mapping, indexid_to_docid_mapping, index_size


def create_or_update_index(index_exists, index_name, index_path, retrieval_model, text_embedding_pairs, metadatas,
                           elasticsearch_url, embeddings, new_text, index_log_path):

    if index_exists:
        print(f'索引: {index_name} 已存在')
        docid_to_indexid_mapping, indexid_to_docid_mapping, index_size = update_faiss_index(
                index_path, index_name, retrieval_model, text_embedding_pairs, metadatas, embeddings)
        # 记录日志
        log_new_ids(index_log_path, index_name, list(docid_to_indexid_mapping.values()))
    else:
        print(f'索引: {index_name} 不存在，正在创建 {retrieval_model} 索引')



        # 获取嵌入向量的维度
        if len(text_embedding_pairs) == 0:
            raise ValueError("text_embedding_pairs 为空，无法创建索引。")
        # sample_embedding = text_embedding_pairs[0][1]
        # dimension = len(sample_embedding)
        #
        # # 准备训练数据（仅使用嵌入向量）
        # training_embeddings = np.array([pair[1] for pair in text_embedding_pairs]).astype('float32')
        #
        # # 设置 IVFPQ 参数
        # nlist = 100 # 簇的数量，根据数据量和向量维度调整
        # quantizer = IndexFlatIP(dimension)  # 内积（Inner Product）作为度量
        # index = IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        #
        #
        # # 训练索引
        # print("Training IVF index...")
        # index.train(training_embeddings)
        # print("Training completed.")

        index_p = os.path.join(index_path, index_name)

        # index = FAISS(
        #     embedding_function=embeddings,
        #     index=index,
        #     docstore=InMemoryDocstore(),
        #     index_to_docstore_id={},
        #     distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        # )
        index = FAISS.from_embeddings(
            [(text, embedding) for text, embedding, _ in text_embedding_pairs],
            embeddings,
            metadatas=[metadata for _, _, metadata in text_embedding_pairs],
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        # index.add_embeddings(
        #     [(text, embedding) for text, embedding, _ in text_embedding_pairs],
        #     metadatas=[metadata for _, _, metadata in text_embedding_pairs],
        # )

        index.save_local(index_p)
        index_size = len(index.docstore._dict)
        print(f'已创建 {retrieval_model} 索引: {index_name}, 当前长度: {index_size}')

        # 获取 docid 和 indexid 并创建映射
        docids = [metadata['id'] for metadata in metadatas]
        index_ids = list(index.docstore._dict.keys())
        if len(docids) != len(index_ids):
            raise ValueError("docids 和 index_ids 的数量不匹配。")
        docid_to_indexid_mapping = dict(zip(docids, index_ids))
        indexid_to_docid_mapping = dict(zip(index_ids, docids))

    print(f'已创建或更新 {retrieval_model} 索引: {index_name}, 当前长度: {index_size}')
    return docid_to_indexid_mapping, indexid_to_docid_mapping


def add_vectors_without_alpha(index_exist, new_text_file, page_content_column, retrieval_model, index_name, index_path,
                normalize_embeddings, index_log_path, elasticsearch_url):

    # 1. 加载新文本
    new_text = load_new_text(new_text_file, page_content_column)

    # 2. 映射检索模型
    mapped_model, instruction = map_retrieval_model(retrieval_model)

    # 3. 加载嵌入
    embeddings = load_embeddings(mapped_model, normalize_embeddings)

    # 4. 处理文本
    new_text = process_texts(new_text, instruction)

    # 5. 更新索引并获取 {docid: indexid} 和 {indexid: docid} 映射
    docid_to_indexid_mapping, indexid_to_docid_mapping, index_size = create_or_update_index_without_alpha(
        index_exists=index_exist,
        index_name=index_name,
        index_path=index_path,
        retrieval_model=mapped_model,
        elasticsearch_url=elasticsearch_url,
        embeddings=embeddings,
        new_text=new_text,
        index_log_path=index_log_path
    )
    print(f'docid_to_indexid_mapping长度: {len(docid_to_indexid_mapping)}')
    print(f'indexid_to_docid_mapping长度: {len(indexid_to_docid_mapping)}')




def add_vectors(index_exist, new_text_file, page_content_column, retrieval_model, index_name, index_path,
                normalize_embeddings, index_log_path, elasticsearch_url, original_embeddings_path,
                mapping_storage_path_docid_to_indexid, mapping_storage_path_indexid_to_docid):
    """
    新增向量：编码，根据权重计算新向量，插入索引，记录docid->indexid, indexid->docid, docid->original_emb
    """
    # 1. 加载新文本
    new_text = load_new_text(new_text_file, page_content_column)

    # 2. 映射检索模型
    mapped_model, instruction = map_retrieval_model(retrieval_model)

    # 3. 加载嵌入
    embeddings = load_embeddings(mapped_model, normalize_embeddings)


    # 4. 处理文本
    new_text = process_texts(new_text, instruction)

    # 5. 验证文档
    validate_documents(new_text)

    # 6. 编码嵌入
    text_embedding_pairs, docid_original_embedding, docid_list = encode_embeddings(new_text, embeddings)

    # 7. 保存原始嵌入
    save_original_embeddings(docid_original_embedding, original_embeddings_path, index_exist)

    # 8. 更新索引并获取 {docid: indexid} 和 {indexid: docid} 映射
    metadatas = [metadata for _, _, metadata in text_embedding_pairs]
    docid_to_indexid_mapping, indexid_to_docid_mapping = create_or_update_index(
        index_exists=index_exist,
        index_name=index_name,
        index_path=index_path,
        retrieval_model=mapped_model,
        text_embedding_pairs=text_embedding_pairs,
        metadatas=metadatas,
        elasticsearch_url=elasticsearch_url,
        embeddings=embeddings,
        new_text=new_text,
        index_log_path=index_log_path
    )
    print(f'docid_to_indexid_mapping长度: {len(docid_to_indexid_mapping)}')
    print(f'indexid_to_docid_mapping长度: {len(indexid_to_docid_mapping)}')
    print(f'original_embeddings长度: {len(docid_original_embedding)}')

    # 9. 保存映射
    save_docid_to_indexid(docid_to_indexid_mapping, mapping_storage_path_docid_to_indexid, index_exist)
    save_indexid_to_docid(indexid_to_docid_mapping, mapping_storage_path_indexid_to_docid, index_exist)


def save_changed_indexid(changed_indexid, changed_indexid_path):
    with open(changed_indexid_path, 'w', encoding='utf-8') as f:
        for idx in changed_indexid:
            f.write(f'{idx}\n')


def update_weights_batch(update_doc_file, page_content_column, mapped_model, index_name, index_path,
                         normalize_embeddings, elasticsearch_url, embeddings, mapping_storage_path_docid_to_indexid,
                         mapping_storage_path_indexid_to_docid, original_embeddings_path, changed_indexid_path, instruction):
    """
    批量更新权重：根据更新文档文件，批量更新对应文档的权重。
    """
    # 1. 加载更新的文档
    update_texts = load_new_text(update_doc_file, page_content_column)
    new_texts = process_texts(update_texts, instruction)

    # 2. 加载映射字典和原始嵌入
    print('加载映射字典和原始嵌入...')
    docid_to_indexid = load_pickle(mapping_storage_path_docid_to_indexid, True)
    indexid_to_docid = load_pickle(mapping_storage_path_indexid_to_docid, True)
    original_embeddings = load_pickle(original_embeddings_path, True)

    # 3. 加载 FAISS/BM25 索引
    if mapped_model != "BM25":
        index_p = os.path.join(index_path, index_name)
        print(f'加载索引: {index_name}')
        faiss_index = FAISS.load_local(index_p, embeddings)
        index_size = len(faiss_index.docstore._dict)
        print(f'已加载索引: {index_name}, 当前长度: {index_size}')
    else:
        raise NotImplementedError("BM25 通常不支持向量更新。")

    modified_docids = []
    modified_alpha_values = []
    docs = []

    # 4. 批量获取更新每个文档id和权重
    for doc in tqdm(new_texts, desc='处理更新文档'):
        # check if the id exists in the docid_to_indexid
        if doc.metadata['id'] not in docid_to_indexid:
            print(f'文档 {doc.metadata["id"]} 不存在于映射中。')
            continue
        modified_docids.append(doc.metadata['id'])
        modified_alpha_values.append(doc.metadata['alpha'])
        docs.append(doc)

    # 5. 获取原始嵌入和 indexid
    original_embeddings_list = []
    indexid_list = []
    for docid in tqdm(modified_docids, desc='获取原始嵌入和 indexid'):
        original_emb = np.array(original_embeddings[docid], dtype=np.float32)
        original_embeddings_list.append(original_emb)
        indexid = docid_to_indexid[docid]
        indexid_list.append(indexid)

    # 6. 基于numpy数组计算新的嵌入
    original_embeddings_array = np.array(original_embeddings_list)
    alphas = np.array(modified_alpha_values, dtype=np.float32)
    embedding_norms = np.linalg.norm(original_embeddings_array, axis=1)
    new_dimensions = np.sqrt(1 - np.square(alphas)) * embedding_norms
    modified_embeddings = original_embeddings_array * alphas[:, np.newaxis]
    modified_embeddings = np.hstack([modified_embeddings, new_dimensions[:, np.newaxis]]).astype(np.float32)

    # 7. 准备文本与嵌入对
    print(f'准备文本与嵌入对...')
    print(f'文档数量: {len(docs)}, 嵌入数量: {len(modified_embeddings)}')
    text_embedding_pairs = [
        (doc.page_content, modified_embeddings[idx].tolist(), doc.metadata)
        for idx, doc in enumerate(docs)
    ]

    # 8. 记录修改的 indexid
    changed_indexid_file = os.path.join(changed_indexid_path, f'{index_name}_changed_indexid_{time.strftime("%Y%m%d-%H%M%S")}.log')
    print(f'已记录修改的 indexid 到文件: {changed_indexid_file}')
    save_changed_indexid(indexid_list, changed_indexid_file)

    # 9. 删除当前 indexid
    print(f'正在删除 {len(indexid_list)} 条文档的权重...')
    faiss_index.delete(indexid_list)

    # 10. 重新插入新的嵌入
    print('正在更新权重...')
    new_ids = faiss_index.add_embeddings(
        [(text, embedding) for text, embedding, _ in text_embedding_pairs],
        metadatas=[metadata for _, _, metadata in text_embedding_pairs],
        ids=indexid_list
    )

    assert new_ids == indexid_list, "新的 indexid 与原 indexid 不一致。"

    # 11. 保存更新后的索引
    print(f'当前索引长度: {len(faiss_index.docstore._dict)}')
    faiss_index.save_local(index_p)

    print(f'已更新 {len(modified_docids)} 条文档的权重。')


def main(
    task,
    new_text_file,
    page_content_column,
    retrieval_model,
    index_name,
    index_path,
    normalize_embeddings,
    index_log_path,
    elasticsearch_url,
    original_embeddings_path,
    mapping_storage_path_docid_to_indexid,
    mapping_storage_path_indexid_to_docid,
    changed_indexid_path,
    update_doc_file,
    index_exists
):
    """
    几个功能：
    1. 新建索引，给定新文本文件，检索模型，索引名称，索引路径，是否归一化嵌入，索引日志路径，elasticsearch_url，原始嵌入路径，
    docid->indexid映射路径，indexid->docid映射路径，创建索引。
    2. 向已存在的索引中添加新文档，给定新文本文件，检索模型，索引名称，索引路径，是否归一化嵌入，索引日志路径，elasticsearch_url，
    原始嵌入路径，docid->indexid映射路径，indexid->docid映射路径，添加新文档。
    3. 更新权重，给定更新文档文件，文档内容列名，检索模型，索引名称，索引路径，是否归一化嵌入，elasticsearch_url，
    嵌入，docid->indexid映射路径，indexid->docid映射路径，原始嵌入路径，更新权重。
    """
    if task.startswith('new_index'):
        print(f'counting lines in the new text file: {new_text_file}')
        line_count = count_lines(new_text_file)
        print(f'line count: {line_count}')
        # count the number of files
        file_count = ceil(line_count / 2000000)
        print(f'file count: {file_count}')
        # split the file and save them into a list
        new_text_file_list = []
        for i in range(file_count):
            new_text_file_list.append(f'{new_text_file}_{retrieval_model}_{i}.jsonl')
        # split the file
        with open(new_text_file, 'r', encoding='utf-8') as f:
            for i in tqdm(range(file_count), desc='splitting the new text file'):
                with open(new_text_file_list[i], 'w', encoding='utf-8') as f2:
                    for j in range(2000000):
                        line = f.readline()
                        f2.write(line)
        # load the new text file
        for i in tqdm(range(file_count), desc='processing the new text file'):
            if i != 0:
                index_exists = True
            new_text_file = new_text_file_list[i]
            print(f'processing file: {new_text_file}')
            if task == 'new_index':
                add_vectors(index_exists, new_text_file, page_content_column, retrieval_model, index_name, index_path,
                            normalize_embeddings, index_log_path, elasticsearch_url, original_embeddings_path,
                            mapping_storage_path_docid_to_indexid, mapping_storage_path_indexid_to_docid)
            elif task == 'new_index_without_alpha':
                add_vectors_without_alpha(index_exists, new_text_file, page_content_column, retrieval_model, index_name, index_path,
                            normalize_embeddings, index_log_path, elasticsearch_url)
            else:
                raise ValueError(f'未知的任务: {task}')
        # remove the split files
        for i in range(file_count):
            os.remove(new_text_file_list[i])
    elif task.startswith('add_vectors'):
        index_exists = True
        if task == 'add_vectors':
            add_vectors(index_exists, new_text_file, page_content_column, retrieval_model, index_name, index_path,
                        normalize_embeddings, index_log_path, elasticsearch_url, original_embeddings_path,
                        mapping_storage_path_docid_to_indexid, mapping_storage_path_indexid_to_docid)
        elif task == 'add_vectors_without_alpha':
            add_vectors_without_alpha(index_exists, new_text_file, page_content_column, retrieval_model, index_name, index_path,
                        normalize_embeddings, index_log_path, elasticsearch_url)
        else:
            raise ValueError(f'未知的任务: {task}')
    elif task == 'update_weights':
        mapped_model, instruction = map_retrieval_model(retrieval_model)
        embeddings = load_embeddings(mapped_model, normalize_embeddings)
        update_weights_batch(update_doc_file, page_content_column, mapped_model, index_name, index_path,
                             normalize_embeddings, elasticsearch_url, embeddings, mapping_storage_path_docid_to_indexid,
                             mapping_storage_path_indexid_to_docid, original_embeddings_path, changed_indexid_path, instruction)
    else:
        raise ValueError(f'未知的任务: {task}')


def load_pickle(file_path, index_exist):
    """
    加载 pickle 文件，如果不存在则返回空字典。
    """
    if os.path.exists(file_path) and index_exist:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"已加载 pickle 文件: {file_path}")
        print(f"数据长度: {len(data)}")
    else:
        data = {}
        print(f"初始化新的 pickle 文件: {file_path}")
    return data


def save_pickle(file_path, data):
    """
    保存字典到 pickle 文件。
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"已保存 pickle 文件: {file_path}")
    print(f"数据长度: {len(data)}")

import linecache

def count_lines(filename):
    count = len(linecache.getlines(filename))
    return count


if __name__ == '__main__':
    config = get_args()
    task = config["task"]
    new_text_file = config["new_text_file"]
    page_content_column = config["page_content_column"]
    retrieval_model = config["retrieval_model"]
    normalize_embeddings = config["normalize_embeddings"]
    index_name = config["index_name"]
    index_path = config["index_path"]
    index_exists = config["index_exists"]
    elasticsearch_url = config["elasticsearch_url"]
    index_log_path = os.path.join(config["index_add_path"], 'index_add_logs')
    pickle_path = os.path.join(config["index_pickle_path"], 'pickles')
    changed_indexid_path = os.path.join(config["index_add_path"], 'changed_indexes')
    if not os.path.exists(config["index_add_path"]):
        os.makedirs(config["index_add_path"])
    if not os.path.exists(changed_indexid_path):
        os.makedirs(changed_indexid_path)
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    if not os.path.exists(index_log_path):
        os.makedirs(index_log_path)
    # json:
    # {
    #   "task": "new_index",
    #   "new_text_file": "../../data_v2/zero_gen_data/DPR/nq-test-gen-baichuan2-13b-chat.jsonl",
    #   "page_content_column": "question",
    #   "retrieval_model": "DPR",
    #   "index_name": "DPR_faiss_index",
    #   "index_path": "../../data_v2/indexes",
    #   "normalize_embeddings": false,
    #   "index_exists": false,
    #   "elasticsearch_url": "http://localhost:9200",
    #   "index_add_path": "../../data_v2/index_add"
    # }



    main(task, new_text_file, page_content_column, retrieval_model, index_name, index_path, normalize_embeddings,
            index_log_path, elasticsearch_url, os.path.join(pickle_path, 'original_embeddings.pkl'),
            os.path.join(pickle_path, 'docid_to_indexid.pkl'), os.path.join(pickle_path, 'indexid_to_docid.pkl'),
            changed_indexid_path, new_text_file, index_exists)

    query_files = config["query_files"]
    page_content_column = config["query_page_content_column"]
    retrieval_model = config["retrieval_model"]
    index_name = config["index_name"]
    index_path = config["index_path"]
    normalize_embeddings = config["normalize_embeddings"]
    output_files = config["output_files"]

    # test the index
    # load the test query file
    if 'without_alpha' in task:
        Retrieval(query_files, page_content_column, retrieval_model, index_name, index_path, normalize_embeddings,
                  output_files, elasticsearch_url, False)
    else:
        Retrieval(query_files, page_content_column, retrieval_model, index_name, index_path, normalize_embeddings,
                  output_files, elasticsearch_url, True)




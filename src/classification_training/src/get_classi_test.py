import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Construct test data for the QA detection model")

    # Paths
    parser.add_argument('--query-dir', type=str, required=True,
                        help='Path to the sampled_query directory containing nq, webq, tqa, pop query files')
    parser.add_argument('--docs-dir', type=str, required=True,
                        help='Path to the test directory containing docs files (nq_top100_docs.json, etc.)')
    parser.add_argument('--llm-data', type=str, required=True,
                        help='Path to the llm_test_data.jsonl file')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to the output combined test data JSONL file')
    parser.add_argument('--multiple-responses', action='store_true', default=False,
                        help='If set, include all responses for each query; otherwise, include only the first response')
    return parser.parse_args()


def load_queries(query_file):
    """
    读取查询文件，并返回一个字典，键为qid，值为问题。
    """
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading queries from {Path(query_file).name}", unit="query"):
            item = json.loads(line.strip())
            qid = item['id']
            question = item['question']
            queries[qid] = question
    return queries


def load_docs(docs_file):
    """
    读取文档文件，并返回一个字典，键为qid，值为包含多个内容的列表。
    """
    docs = {}
    with open(docs_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading docs from {Path(docs_file).name}", unit="doc"):
            item = json.loads(line.strip())
            qid = item['qid']
            content = item['contents']
            if qid not in docs:
                docs[qid] = []
            docs[qid].append(content)
    return docs


def process_human_data(query_dir, docs_dir, datasets, multiple_responses=False):
    """
    处理人类生成的数据，返回一个样本列表，标签为0。
    """
    human_samples = []
    for dataset in datasets:
        query_file = Path(query_dir) / f"{dataset}-test-sample-200.jsonl"
        docs_file = Path(docs_dir) / f"{dataset}_top100_docs.json"

        if not query_file.exists():
            print(f"Warning: Query file {query_file} does not exist. Skipping dataset {dataset}.")
            continue
        if not docs_file.exists():
            print(f"Warning: Docs file {docs_file} does not exist. Skipping dataset {dataset}.")
            continue

        queries = load_queries(query_file)
        docs = load_docs(docs_file)
        for qid, question in tqdm(queries.items(), desc=f"Processing {dataset} queries", unit="query"):
            if str(qid) in docs:
                qid = str(qid)
                responses = docs[qid]
                if multiple_responses:
                    for response in responses:
                        sample = {
                            'query': question,
                            'response': response,
                            'label': 0
                        }
                        human_samples.append(sample)
                else:
                    # 仅使用第一个响应
                    response = responses[0]
                    sample = {
                        'query': question,
                        'response': response,
                        'label': 0
                    }
                    human_samples.append(sample)
            else:
                print(f"Warning: No docs found for qid {qid} in dataset {dataset}. Skipping this query.")
    return human_samples


def process_llm_data(llm_data_file):
    """
    处理LLM生成的数据，返回一个样本列表，标签为1。
    """
    llm_samples = []
    with open(llm_data_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading LLM data from {Path(llm_data_file).name}", unit="sample"):
            item = json.loads(line.strip())
            question = item['question']
            response = item['response']
            sample = {
                'query': question,
                'response': response,
                'label': 1
            }
            llm_samples.append(sample)
    return llm_samples


def save_combined_data(output_file, human_samples, llm_samples):
    """
    将人类和LLM生成的样本保存到一个JSONL文件中。
    """
    total = len(human_samples) + len(llm_samples)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(human_samples, desc="Saving human data", unit="sample"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        for sample in tqdm(llm_samples, desc="Saving LLM data", unit="sample"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Combined test data saved to {output_file} ({total} samples)")


def main():
    args = parse_args()

    query_dir = Path(args.query_dir)
    docs_dir = Path(args.docs_dir)
    llm_data_file = Path(args.llm_data)
    output_file = Path(args.output_file)

    datasets = ['nq', 'webq', 'tqa', 'pop']
    # datasets = ['pop']
    print("Processing human-generated data...")
    human_samples = process_human_data(query_dir, docs_dir, datasets, multiple_responses=args.multiple_responses)
    print(f"Human-generated samples: {len(human_samples)}")

    print("Processing LLM-generated data...")
    llm_samples = process_llm_data(llm_data_file)
    print(f"LLM-generated samples: {len(llm_samples)}")

    # 如果需要，可以在此处对样本进行随机打乱
    # from random import shuffle
    # combined_samples = human_samples + llm_samples
    # shuffle(combined_samples)

    print("Saving combined test data...")
    save_combined_data(output_file, human_samples, llm_samples)


if __name__ == "__main__":
    main()

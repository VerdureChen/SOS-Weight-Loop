import json
import random
import re
import string
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_custom_data(files):
    custom_data = {}
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                query = entry['query']
                response = entry['response']
                if query in custom_data:
                    custom_data[query].append(response)
                else:
                    custom_data[query] = [response]
    print(f'len:{len(custom_data)}')
    return custom_data


def enhance_with_custom_data(dataset_name='nq'):


    # 文件路径模板
    original_train_file = '/home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/sos_train/data/ori_train/{}-train.jsonl'
    right_file_template = '/home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/sos_train/data/llm_gen/right/{}_query.jsonl'
    wrong_file_template = '/home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/sos_train/data/llm_gen/wrong/{}_query.jsonl'
    output_file_template = '/home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/ada_journal/SOS-Retrieval-Loop/src/classification_training/data/LLM-detection/{}-train.jsonl'

    # 利用数据集名称插入模板路径
    ORIGINAL_TRAIN_FILE = original_train_file.format(dataset_name)

    RIGHT_FILES = [
        right_file_template.format(f'chatglm3-6b_{dataset_name}'),
        right_file_template.format(f'Baichuan2-13B-Chat_{dataset_name}'),
        right_file_template.format(f'llama-2-13b-chat-hf_{dataset_name}'),
        right_file_template.format(f'Qwen-14B-Chat_{dataset_name}')
    ]

    WRONG_FILES = [
        wrong_file_template.format(f'chatglm3-6b_{dataset_name}'),
        wrong_file_template.format(f'Baichuan2-13B-Chat_{dataset_name}'),
        wrong_file_template.format(f'llama-2-13b-chat-hf_{dataset_name}'),
        wrong_file_template.format(f'Qwen-14B-Chat_{dataset_name}')
    ]

    OUTPUT_FILE = output_file_template.format(dataset_name)


    # Load the original training data
    with open(ORIGINAL_TRAIN_FILE, 'r') as file:
        original_data = [json.loads(line) for line in file]

    # Load the right samples data
    right_data = load_custom_data(RIGHT_FILES)

    # Load the wrong samples data
    wrong_data = load_custom_data(WRONG_FILES)

    enhanced_data_llm = []
    enhanced_data_human = []

    for entry in original_data:
        query = entry['query']
        pos = entry['pos']
        neg = entry['neg']
        out_pos = []
        out_neg = []
        # Handle positive samples
        if query in right_data:
            for custom_pos in right_data[query]:
                out_pos.append(custom_pos)
        # Ensure pos length is twice the original length
        target_pos_length = max(len(out_pos) * 2, 8)
        pos_llm = out_pos
        pos_human = random.sample(pos, min(target_pos_length - len(out_pos), len(pos)))

        # Handle negative samples
        if query in wrong_data:
            for custom_neg in wrong_data[query]:
                out_neg.append(custom_neg)
        # Ensure neg length is twice the original length
        target_neg_length = max(len(out_neg) * 2, 8)
        neg_llm = out_neg
        neg_human = random.sample(neg, min(target_neg_length - len(out_neg), len(neg)))

        enhanced_data_llm.extend([{'query': query, 'response': response, 'label': 1} for response in pos_llm])
        enhanced_data_llm.extend([{'query': query, 'response': response, 'label': 1} for response in neg_llm])
        enhanced_data_human.extend([{'query': query, 'response': response, 'label': 0} for response in pos_human])
        enhanced_data_human.extend([{'query': query, 'response': response, 'label': 0} for response in neg_human])

    # Shuffle the data
    data = enhanced_data_llm + enhanced_data_human
    random.shuffle(data)

    # Write the enhanced data to the output file
    with open(OUTPUT_FILE, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

    print(f'Enhanced data written to {OUTPUT_FILE}')


if __name__ == '__main__':
    dataset_names = ['nq', 'webq', 'trivia']
    for dataset_name in dataset_names:
        enhance_with_custom_data(dataset_name)

    # merge all the data
    


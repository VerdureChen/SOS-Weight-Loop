import os
import json
import argparse
from pathlib import Path
import torch
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline,
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# 获取 Transformers 的 logger
logger = logging.getLogger("transformers")
# 设置日志级别为 ERROR，忽略 WARNING 和 INFO 级别的日志
logger.setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Roberta model for ChatGPT QA Detection")

    parser.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size per device (default: 16)')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='Number of training epochs (default: 2)')
    parser.add_argument('--cuda', '-c', type=str, default='0', help='GPU ids, like: "0,1,2" (default: "0")')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length (default: 512)')
    parser.add_argument("--pair", action="store_true", default=True, help='Use paired input (default: False)')
    parser.add_argument("--all-train", action="store_true", default=True,
                        help='Use all data for training without validation split (default: False)')

    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            item = json.loads(line.strip())
            data.append({
                'query': item['query'],
                'response': item['response'],
                'label': item['label']
            })
    return data


def create_dataset(data):
    return Dataset.from_list(data)


def preprocess_function(examples, tokenizer, max_length, pair):
    if pair:
        return tokenizer(
            examples['query'],
            examples['response'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=False
        )
    else:
        return tokenizer(
            examples['query'] + " " + examples['response'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=False
        )


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    args = parse_args()

    # 设置种子以确保结果可复现
    set_seed(args.seed)

    # 设置GPU设备
    gpu_ids = args.cuda.split(',')
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')})")

    # 配置参数
    MODEL_NAME = "../../../../../../Rob_LLM/ret_model/chatgpt-qa-detector-roberta"
    DATA_PATH = "../data/LLM-detection/merge_train.jsonl"
    OUTPUT_DIR = "../output/fine-tuned-chatgpt-qa-detector-roberta"
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = 1e-5  # 可根据需求进一步参数化
    MAX_SEQ_LENGTH = args.max_length
    SEED = args.seed
    PAIR_INPUT = args.pair
    ALL_TRAIN = args.all_train

    # 加载数据
    raw_data = load_data(DATA_PATH)

    # 划分训练集和验证集
    if not ALL_TRAIN:
        train_data, eval_data = train_test_split(raw_data, test_size=0.1, random_state=SEED)
        print(f"训练集大小: {len(train_data)}, 验证集大小: {len(eval_data)}")
    else:
        train_data = raw_data
        eval_data = raw_data  # 使用训练集作为验证集（或不进行验证）
        print(f"使用所有数据进行训练 (总数据量: {len(train_data)})")

    # 转换为 Hugging Face 的 Dataset 对象
    train_dataset = create_dataset(train_data)
    eval_dataset = create_dataset(eval_data)

    if not ALL_TRAIN:
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': eval_dataset
        })
    else:
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': eval_dataset  # 虽然与 train 相同，但可以根据需要修改
        })

    # 加载分词器
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # 定义数据预处理函数，处理输入
    def preprocess(examples):
        return preprocess_function(examples, tokenizer, MAX_SEQ_LENGTH, PAIR_INPUT)

    # 应用预处理
    tokenized_datasets = dataset.map(preprocess, batched=True)

    # 准备标签
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 加载模型
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)  # 将模型移动到 GPU

    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=SEED,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        fp16=torch.cuda.is_available(),  # 启用混合精度训练
        dataloader_num_workers=4,  # 根据 CPU 核数调整
        # 如果需要更多优化，可以添加以下参数
        # gradient_accumulation_steps=2,
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    trainer.train()

    # 评估模型
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")

    # 保存模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"模型已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

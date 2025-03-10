import os
import json
import argparse
from pathlib import Path
import torch
from transformers import pipeline, RobertaTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Configure logging
logger = logging.getLogger("transformers")
logger.setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Fine-tuned Roberta model for ChatGPT QA Detection using pipeline")

    parser.add_argument('--model-dir', type=str, required=True, help='Path to the trained model directory')
    parser.add_argument('--test-data', type=str, required=True, help='Path to the test data JSONL file')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length (default: 512)')
    parser.add_argument('--pair', action="store_true", default=True, help='Use paired input (default: True)')
    parser.add_argument('--cuda', '-c', type=str, default='0', help='GPU ids, like: "0,1,2" (default: "0")')
    parser.add_argument('--save-predictions', action='store_true', default=True, help='Flag to save predictions to a file')
    args = parser.parse_args()
    print(args)
    return args


def load_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading test data"):
            item = json.loads(line.strip())
            data.append({
                'query': item['query'],
                'response': item['response'],
                'label': item['label']
            })
    return data


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true, y_pred)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def get_batches(inputs, batch_size):
    """Divide inputs into batches of specified size."""
    for i in range(0, len(inputs), batch_size):
        yield inputs[i:i + batch_size]


def main():
    args = parse_args()

    # Set GPU devices
    gpu_ids = args.cuda.split(',')
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')})")

    # Load test data
    test_data = load_data(args.test_data)
    print(f"Test set size: {len(test_data)}")

    # Initialize tokenizer (optional, if needed for any preprocessing)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)

    # Initialize the text classification pipeline
    detector = pipeline(
        'text-classification',
        model=args.model_dir,
        tokenizer=args.model_dir,
        device=0 if torch.cuda.is_available() else -1,
        framework='pt',
        top_k=None,  # to get all labels
    )

    # Prepare inputs for the pipeline
    inputs = []
    for item in test_data:
        if args.pair:
            inputs.append({'text': item['query'], 'text_pair': item['response']})
        else:
            concatenated = f"{item['query']} {item['response']}"
            inputs.append({'text': concatenated})

    # Batch processing
    all_preds = []
    all_scores = []
    print("Starting batch processing...")
    for batch in tqdm(get_batches(inputs, args.batch_size),
                      total=(len(inputs) + args.batch_size - 1) // args.batch_size, desc="Processing Batches"):
        try:
            if args.pair:
                results = detector(batch, truncation=True, max_length=args.max_length)
            else:
                texts = [item['text'] for item in batch]
                results = detector(texts, truncation=True, max_length=args.max_length)

            for res in results:
                # Handle cases where res is a list (for paired inputs) or a single dict
                if isinstance(res, list):
                    # Assuming binary classification with labels 'LABEL_0' and 'LABEL_1'
                    if res[0]['score'] > res[1]['score']:
                        label = res[0]['label']
                        score = res[1]['score']
                    else:
                        label = res[1]['label']
                        score = res[1]['score']
                else:
                    # Single label classification
                    label = res['label']
                    score = res['score']

                pred = int(label.split('_')[-1])  # Extract numeric label
                all_scores.append(score)
                all_preds.append(pred)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Optionally, handle specific errors or skip the batch
            for _ in batch:
                all_preds.append(-1)

    # Extract true labels
    all_labels = [item['label'] for item in test_data]

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds)
    print(
        f"Evaluation Metrics:\nAccuracy: {metrics['accuracy']:.4f}\nPrecision: {metrics['precision']:.4f}\nRecall: {metrics['recall']:.4f}\nF1 Score: {metrics['f1']:.4f}")

    # Optionally save predictions
    if args.save_predictions:
        output_predictions = []
        for i in range(len(all_preds)):
            output_predictions.append({
                'query': test_data[i]['query'],
                'response': test_data[i]['response'],
                'true_label': all_labels[i],
                'predicted_label': all_preds[i],
                'score': all_scores[i]
            })

        output_path = os.path.join(args.model_dir, "test_predictions_pipeline.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in output_predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()

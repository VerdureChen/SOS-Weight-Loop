import argparse
import os
import json
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import pandas as pd
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import datasets
import re
import string

# Define the prompt for scoring factual correctness
SCORING_PROMPT_TEMPLATE = """
Question: {question}
Text: {response}
Please rate the factual correctness of the text on a scale from 0 to 1, allowing decimal values, where:
- 1.0 means the response is completely correct.
- 0.7 means the response is mostly correct but may have minor inaccuracies or omissions.
- 0.4 means the response contains some accurate information but has significant factual inaccuracies or omissions.
- 0.1 means the response is mostly incorrect with only a minimal portion being factual.
- 0.0 means the response is completely incorrect.
Provide only the numerical score (e.g., 0.85) without any additional text.
"""

def get_openai_api(model_config):
    """
    Configure the OpenAI API with the provided model configuration.
    """
    model_name = model_config["model_name"]
    api_base = model_config.get("api_base")
    api_key = model_config.get("api_key")

    if not api_base or not api_key:
        raise ValueError(f"api_base and api_key must be provided for model {model_name}")

    openai.api_key = api_key
    openai.api_base = api_base
    return model_name

def get_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="LLM Fact-Checking Scorer")
    parser.add_argument("--config_file_path", type=str, required=True, help="Path to the JSON config file.")
    args = parser.parse_args()
    return args

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_factual_score(model_config, question, response):
    """
    Send a prompt to the LLM to rate the factual correctness of the response.
    Returns a float score between 0 and 1.
    """
    prompt = SCORING_PROMPT_TEMPLATE.format(question=question, response=response)
    try:
        completion = openai.ChatCompletion.create(
            model=model_config["model_name"],
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0,  # Set to 0 for deterministic output
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        resp = completion.choices[0].message.content.strip()
        # Extract numerical score
        score = float(resp.split()[0])
        # Validate score range
        if score < 0 or score > 1:
            raise ValueError(f"Score {score} out of range [0,1] from model {model_config['model_name']}")
        return score
    except Exception as e:
        raise e

def process_model(model_config, df, output_dir, max_workers=20):
    """
    Process scoring for a single model using multi-threading.
    """
    model_name = get_openai_api(model_config)
    print(f"\nProcessing model: {model_name}")
    print(f'output file:{output_dir}')

    scores = [None] * len(df)  # Initialize list with None

    def score_row(idx, row):
        question = row["question"]
        response = row["response"]
        try:
            score = get_factual_score(model_config, question, response)
            return idx, score
        except Exception as e:
            print(f"Failed to get score for example ID {row.get('id', 'N/A')}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(score_row, idx, row): idx for idx, row in df.iterrows()}
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=f"Model: {model_name}"):
            idx, score = future.result()
            scores[idx] = score

    # Add the scores to the DataFrame
    df[f"{model_name}_score"] = scores

    # Save the scores to a JSONL file
    output_file = output_dir
    print(f"Saving scores to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for _, row in df.iterrows():
            output_dict = {
                "id": row.get("id"),
                "score": row.get(f"{model_name}_score"),
                "exact_match": row.get("exact_match"),
            }
            f_out.write(json.dumps(output_dict) + "\n")

    return model_name

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
            print(f'no prediction for qid {example["id"]}, {example["question"]}')
            return example
        for ground_truth in ground_truths:
            if _normalize_answer(ground_truth) in _normalize_answer(prediction):
                example['exact_match'] = 1
                break
        return example

    return exact_match_score(predictions)

def main():
    args = get_args()
    config_file_path = args.config_file_path

    # Load configuration
    with open(config_file_path, "r") as f:
        config = json.load(f)

    models = config.get("models", [])
    input_dir = config.get("input_file_path")
    output_dir = config.get("output_dir", "llm_scores")

    if not models:
        print("No models specified in the configuration.")
        sys.exit(1)

    jsonl_files = [file for file in os.listdir(input_dir) if file.endswith('.jsonl')]

    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(jsonl_files, desc='Evaluating'):
        input_file_path = os.path.join(input_dir, file)
        dataset = datasets.load_dataset('json', data_files=input_file_path)['train']
        predictions = dataset.map(evaluate)
        predictions.to_json(input_file_path)


    for file in tqdm(jsonl_files, desc='Generating'):
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)
        # Load the dataset
        print("Loading dataset...")
        data = []
        with open(input_file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)

        # Prepare a DataFrame to store exact_match scores
        if "exact_match" not in df.columns:
            print("The input data does not contain 'exact_match' field.")
            sys.exit(1)

        exact_match_scores = df["exact_match"].astype(float).values
        if len(exact_match_scores) != len(df):
            print("Mismatch in number of exact_match scores.")
            sys.exit(1)

        # Iterate over each model with parallel processing if desired
        # Here, we'll process models sequentially but handle scoring in parallel
        for model_config in models:
            process_model(model_config, df, output_file_path, max_workers=10)  # Adjust max_workers as needed

        # After scoring with all models, compute correlations
        print("\nComputing correlations with exact_match scores...")
        correlation_results = []
        for model_config in models:
            model_name = model_config["model_name"]
            score_col = f"{model_name}_score"

            # Drop entries where score is None
            valid_indices = df[score_col].notnull()
            if valid_indices.sum() == 0:
                print(f"No valid scores for model {model_name}, skipping correlation.")
                continue

            model_scores = df.loc[valid_indices, score_col].astype(float).values
            exact_matches = df.loc[valid_indices, "exact_match"].astype(float).values

            if len(model_scores) != len(exact_matches):
                print(f"Length mismatch for model {model_name}, skipping correlation.")
                continue

            # Compute Pearson and Spearman correlations
            pearson_corr = pd.Series(model_scores).corr(pd.Series(exact_matches), method='pearson')
            spearman_corr = pd.Series(model_scores).corr(pd.Series(exact_matches), method='spearman')

            correlation_results.append({
                "model": model_name,
                "pearson_correlation": pearson_corr,
                "spearman_correlation": spearman_corr
            })

        # Display the correlation results
        print("\nCorrelation Results:")
        for result in correlation_results:
            print(f"Model: {result['model']}")
            print(f"  Pearson Correlation: {result['pearson_correlation']:.4f}")
            print(f"  Spearman Correlation: {result['spearman_correlation']:.4f}\n")

if __name__ == '__main__':
    main()
python llm_test.py \
    --model-dir ../output/fine-tuned-chatgpt-qa-detector-roberta \
    --test-data ../data/LLM-detection/combined_test_data.jsonl \
    --batch-size 2048 \
    --max-length 512 \
    --cuda 0

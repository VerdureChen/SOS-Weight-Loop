python get_classi_test.py \
    --query-dir /home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/sos_train/data/test/sampled_query \
    --docs-dir /home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/sos_train/data/test \
    --llm-data /home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/sos_train/data/test/llm_test_data.jsonl \
    --output-file /home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/ada_journal/SOS-Retrieval-Loop/src/classification_training/data/LLM-detection/combined_test_data.jsonl \
    --multiple-responses  # 如果您希望每个query包含所有响应，否则可以省略此参数

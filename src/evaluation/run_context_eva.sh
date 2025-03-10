#!/usr/bin/env bash

# if you are using "QA_llm_mis" or "QA_llm_right" task, you need to set the API_KEY and API_BASE
API_KEY="none"
API_BASE="none"
QUERY_DATA_NAMES=(nq webq pop tqa)

RESULT_NAMES=(
    "fit_2_weight_nq_webq_tqa_pop_loop_output_llm-embedder_total_loop_10_20250222112331"
#    "filter_source_weight_nq_webq_tqa_pop_loop_output_llm-embedder_total_loop_10_20250214062309"
)


RETRIEVAL_MODEL_NAMES=llm-embedder
#RESULT_NAMES=(
#    "weight_nq_webq_tqa_pop_loop_output_bge-base_total_loop_10_20241224115737"
#)

#RESULT_NAMES=( "mis_passage_processed" )
RESULT_DIR="../../data_v2/loop_output/DPR"
WEIGHT_DIR="../../data_v2/function_test/llm_loop"
TASK=("retrieval" "llm_text_type_rate" "correctness_query" "bleu" )
for ((k=0;k<${#TASK[@]};k++))
    do
for ((i=0;i<${#QUERY_DATA_NAMES[@]};i++))
do
  for ((j=0;j<${#RESULT_NAMES[@]};j++))
  do

    QUERY_DATA_NAME=${QUERY_DATA_NAMES[i]}
    RESULT_NAME=${RESULT_NAMES[j]}
    RESULT_PATH="${RESULT_DIR}/${RESULT_NAME}/${QUERY_DATA_NAME}"
    echo "QUERY_DATA_NAME: ${QUERY_DATA_NAME}"
    echo "RESULT_NAME: ${RESULT_NAME}"
    echo "RESULT_PATH: ${RESULT_PATH}"
    python3 eva_pipe.py --config_file_path none --directory ${RESULT_PATH} --task ${TASK[k]} --api_key $API_KEY --api_base $API_BASE --weight_merge_path $WEIGHT_DIR --retrieval_model_names $RETRIEVAL_MODEL_NAMES
  done
  done
done

#!/usr/bin/env bash

RUN_DIR=$(pwd)
elasticsearch_url="http://124.16.138.150:9978"
#循环次数
TOTAL_LOOP_NUM=10
#方法设置
RETRIEVAL_MODEL_NAME=bge-base # dpr contriever retromae all-mpnet bge-base llm-embedder bm25
CORPUS_NAME=psgs_w100

QUERY_DATA_NAMES=(nq webq tqa pop)
QUERY_DATA_PATH="${RUN_DIR}/../data_v2/input_data/DPR/modified_sampled_query"
QUERY_NAME_FORMAT="-test-sample-200.jsonl"

ADD_FILE_PATH="${RUN_DIR}/../data_v2/function_test/llm_loop/${RETRIEVAL_MODEL_NAME}"

RET_LIST_NOW=""

#创建目录
TIMESTAMP=$(date +%Y%m%d%H%M%S)
#TIMESTAMP=20250126112633

# concanate all the query data names
QUERY_DATA_NAME=$(IFS=_; echo "${QUERY_DATA_NAMES[*]}")

LOOP_CONFIG_PATH_NAME="${RUN_DIR}/run_configs/weight_${QUERY_DATA_NAME}_merge_loop_config_${RETRIEVAL_MODEL_NAME}_total_loop_${TOTAL_LOOP_NUM}_${TIMESTAMP}"
mkdir -p $LOOP_CONFIG_PATH_NAME
TOTAL_LOG_DIR="${RUN_DIR}/run_logs/weight_${QUERY_DATA_NAME}_merge_loop_log_${RETRIEVAL_MODEL_NAME}_total_loop_${TOTAL_LOOP_NUM}_${TIMESTAMP}"
mkdir -p $TOTAL_LOG_DIR
OUTPUT_DIR="${RUN_DIR}/../data_v2/loop_output/DPR/weight_${QUERY_DATA_NAME}_loop_output_${RETRIEVAL_MODEL_NAME}_total_loop_${TOTAL_LOOP_NUM}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "create config dir: ${LOOP_CONFIG_PATH_NAME}"
echo "create log dir: ${TOTAL_LOG_DIR}"
echo "create output dir: ${OUTPUT_DIR}"



# 指定rerank GPU设备
export CUDA_VISIBLE_DEVICES=0

# search before loop

cd retrieval_loop
LOOP_NUM=0
echo "rewrite config file for ${RETRIEVAL_MODEL_NAME} on ${QUERY_DATA_NAME}..."
CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_0.json"
LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_00.log"
#RETRIEVAL_OUTPUT_PATH="${OUTPUT_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}.json"

TRET_CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_total_llm_retrieval_loop_0.json"
TRET_LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_total_llm_retrieval_loop_00.log"
TOTAL_RETRIEVAL_OUTPUT_PATH="${OUTPUT_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_total_llm_retrieval_loop_${LOOP_NUM}"

FACT_SCORE_PATH="${ADD_FILE_PATH}/fact_check/"

EVA_CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_eva_init_add_loop_0.json"
EVA_LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_eva_init_add_loop_0.log"

# Initialize empty strings for file lists
QUERY_FILE_LIST=""
ORI_OUTPUT_FILE_LIST=""

# Loop through QUERY_DATA_NAMES to build up the file lists
for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"; do
  QUERY_FILE_PATH="${QUERY_DATA_PATH}/${QUERY_DATA_NAME}${QUERY_NAME_FORMAT}"
  # Append the file path to the list, surrounded by quotes, separated by commas if not the first item
  if [ -z "${QUERY_FILE_LIST}" ]; then
    QUERY_FILE_LIST="\"${QUERY_FILE_PATH}\""
  else
    QUERY_FILE_LIST="${QUERY_FILE_LIST},\"${QUERY_FILE_PATH}\""
  fi
  OUTPUT_DIR_QUERY="${OUTPUT_DIR}/${QUERY_DATA_NAME}"
  mkdir -p $OUTPUT_DIR_QUERY
  OUTPUT_FILE_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}"
  if [ -z "${ORI_OUTPUT_FILE_LIST}" ]; then
    ORI_OUTPUT_FILE_LIST="\"${OUTPUT_FILE_PATH}\""
  else
    ORI_OUTPUT_FILE_LIST="${ORI_OUTPUT_FILE_LIST},\"${OUTPUT_FILE_PATH}\""
  fi
done

# 对查询进行初始化检索，确定未添加文档前系统的效果
python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "alpha_retrieval" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{"query_files": ['"${QUERY_FILE_LIST}"'], "output_files": ['"${ORI_OUTPUT_FILE_LIST}"'] , "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": true}'
wait
echo "Running retrieval for ${RETRIEVAL_MODEL_NAME}"


python retrieve_methods.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &

wait
RET_LIST_NOW=${ORI_OUTPUT_FILE_LIST}
python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "compute_total_llm" \
                        --output_dir "${TRET_CONFIG_PATH}" \
                        --overrides '{"input_file": ['"${QUERY_FILE_LIST}"'], "retrieved_file": ['"${ORI_OUTPUT_FILE_LIST}"'] , "elasticsearch_url": "'"${elasticsearch_url}"'",
                        "added_documents": "'"${ADD_FILE_PATH}"'",
                        "normalize_embeddings": true, "output_file": "'"${TOTAL_RETRIEVAL_OUTPUT_PATH}"'"}'
wait
echo "Running total sim for ${RETRIEVAL_MODEL_NAME}"


python ret_arr_pos.py --config_file_path "$TRET_CONFIG_PATH" > "$TRET_LOG_DIR" 2>&1 &

wait

python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "weight_init" \
                        --output_dir "${EVA_CONFIG_PATH}" \
                        --overrides '{"task":"evaluation", "input_file": ['"${QUERY_FILE_LIST}"'],
                        "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": true,
                        "retrieved_file": ['"${ORI_OUTPUT_FILE_LIST}"'],
                        "added_documents": "'"${ADD_FILE_PATH}"'",
                        "new_add_fact_score_file": "'"${FACT_SCORE_PATH}"'"}'

wait

echo "Running init eva for ${RETRIEVAL_MODEL_NAME}"

python get_init_weight.py --config_file_path "$EVA_CONFIG_PATH" > "$EVA_LOG_DIR" 2>&1 &

wait


# 对待添加文档进行权重初始化，然后对参数做一次评估，看直接优化的效果
for ((LOOP_NUM=0; LOOP_NUM<${TOTAL_LOOP_NUM}; LOOP_NUM++))
do
  INPUT_FILE_PATH="${ADD_FILE_PATH}/loop${LOOP_NUM}_merged.jsonl"
  OUTPUT_FILE_PATH="${OUTPUT_DIR}/weight_init_output/loop${LOOP_NUM}_merged_weight.json"
  WEIGHT_INIT_PATH="${OUTPUT_DIR}/weight_init_output"
  WEIGHT_UPDATE_PATH="${OUTPUT_DIR}/weight_update_output"
  mkdir -p $OUTPUT_DIR/weight_init_output
  mkdir -p $OUTPUT_DIR/weight_update_output
  mkdir -p $OUTPUT_DIR/retrieval_pq
  PSEUDO_QUERY_OUTPUT_FILE_PATH="${ADD_FILE_PATH}/pq/loop${LOOP_NUM}_pseudo_query.json"
  RETRIEVAL_PQ_OUTPUT_FILE_PATH="${OUTPUT_DIR}/retrieval_pq/pseudo_query_${RETRIEVAL_MODEL_NAME}_retrieval_loop_${LOOP_NUM}"
  ORI_RETRIEVAL_PQ_OUTPUT_FILE_PATH="${OUTPUT_DIR}/retrieval_pq/pseudo_query_${RETRIEVAL_MODEL_NAME}_retrieval_loop_${LOOP_NUM}_ori"
  FACT_SCORE_PATH="${ADD_FILE_PATH}/fact_check/loop${LOOP_NUM}_merged.jsonl"

  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_pq_llm_loop_${LOOP_NUM}.json"
  LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_pq_llm_loop_${LOOP_NUM}.log"


  python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "pq_llm" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{"task": "retrieve", "loop": "'"${LOOP_NUM}"'",
                        "weight_init_path": "'"${WEIGHT_INIT_PATH}"'", "weight_update_path": "'"${WEIGHT_UPDATE_PATH}"'",
                        "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": true,
                        "retrieved_file": "'"${ORI_RETRIEVAL_PQ_OUTPUT_FILE_PATH}"'",
                        "input_file": "'"${PSEUDO_QUERY_OUTPUT_FILE_PATH}"'",
                        "output_file": "'"${RETRIEVAL_PQ_OUTPUT_FILE_PATH}"'",
                        "added_documents": "'"${ADD_FILE_PATH}"'"}'

  wait

  echo "Running pq-llm eva for ${RETRIEVAL_MODEL_NAME}"

  # if loop_num is not 0, then we need to get the initial weight
#  if [ $LOOP_NUM -ne 0 ]; then
#    python get_init_weight.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
#  fi
  python ret_arr_pos.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &

  wait

  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_weight_loop_${LOOP_NUM}.json"
  LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_weight_loop_${LOOP_NUM}.log"


  python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "weight_init" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{"input_file": "'"${INPUT_FILE_PATH}"'", "output_file": "'"${OUTPUT_FILE_PATH}"'" ,
                        "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": true,
                        "pseudo_query_output": "'"${PSEUDO_QUERY_OUTPUT_FILE_PATH}"'",
                        "retrieved_file": "'"${RETRIEVAL_PQ_OUTPUT_FILE_PATH}"'",
                        "added_documents": "'"${ADD_FILE_PATH}"'",
                        "new_add_fact_score_file": "'"${FACT_SCORE_PATH}"'"}'

  wait

  echo "Running weight init for ${RETRIEVAL_MODEL_NAME}"

  # if loop_num is not 0, then we need to get the initial weight
#  if [ $LOOP_NUM -ne 0 ]; then
#    python get_init_weight.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
#  fi
  python get_init_weight.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &

  wait

  # 将权重初始化后的文档加入索引
  OUTPUT_FILE_LIST=""
  for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"; do
    OUTPUT_DIR_QUERY="${OUTPUT_DIR}/${QUERY_DATA_NAME}"
    mkdir -p $OUTPUT_DIR_QUERY
    OUTPUT_FILE_DIR="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_after_init_loop_${LOOP_NUM}"
    if [ -z "${OUTPUT_FILE_LIST}" ]; then
      OUTPUT_FILE_LIST="\"${OUTPUT_FILE_DIR}\""
    else
      OUTPUT_FILE_LIST="${OUTPUT_FILE_LIST},\"${OUTPUT_FILE_DIR}\""
    fi
  done
  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_merge_init_weight_loop_${LOOP_NUM}.json"
  python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "merge" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{ "task": "get_weighted_results", "weights_file": "'"${OUTPUT_FILE_PATH}"'", "output_file": ['"${OUTPUT_FILE_LIST}"'],
                        "retrieval_file_ori":['"${ORI_OUTPUT_FILE_LIST}"'], "query_file": ['"${QUERY_FILE_LIST}"'], "retrieval_file_now": ['"${RET_LIST_NOW}"'],
                        "q_llm_file": "'"${TOTAL_RETRIEVAL_OUTPUT_PATH}"'",
                        "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": true, "added_documents": "'"${ADD_FILE_PATH}"'"}'

  wait

  echo "Running index for ${RETRIEVAL_MODEL_NAME}"

  LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_merge_init_weight_loop_${LOOP_NUM}.log"
  python ret_arr_pos.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &


  wait



  #对检索后的文档进行评估
  RET_LIST_NOW=${OUTPUT_FILE_LIST}
  FACT_SCORE_PATH="${ADD_FILE_PATH}/fact_check/"

  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_eva_weight_after_add_loop_${LOOP_NUM}.json"
  LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_eva_weight_after_add_loop_${LOOP_NUM}.log"


  python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "weight_init" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{"task":"evaluation", "input_file": ['"${QUERY_FILE_LIST}"'],
                        "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": true,
                        "retrieved_file": ['"${OUTPUT_FILE_LIST}"'],
                        "added_documents": "'"${ADD_FILE_PATH}"'",
                        "new_add_fact_score_file": "'"${FACT_SCORE_PATH}"'"}'

  wait

  echo "Running weight init eva for ${RETRIEVAL_MODEL_NAME}"

  python get_init_weight.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &

  wait


  #基于用户查询进行批量化优化
  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_weight_updata_loop_${LOOP_NUM}.json"
  LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_weight_update_loop_${LOOP_NUM}.log"
  OUTPUT_FILE_PATH="${OUTPUT_DIR}/weight_update_output/loop${LOOP_NUM}_merged_weight.json"
  mkdir -p $OUTPUT_DIR/weight_update_output

  python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "weight_update" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{"task":"weight_update", "added_documents": "'"${ADD_FILE_PATH}"'",
                        "input_query": ['"${QUERY_FILE_LIST}"'], "output_file": "'"${OUTPUT_FILE_PATH}"'",
                        "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": true,
                        "retrieved_file": ['"${OUTPUT_FILE_LIST}"'],
                        "new_add_fact_score_file": "'"${FACT_SCORE_PATH}"'"}'

  wait

  echo "Running weight update for ${RETRIEVAL_MODEL_NAME}"

  python get_init_weight.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
  wait

  #更新优化后的索引权重
  OUTPUT_FILE_LIST=""
  for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"; do
    OUTPUT_DIR_QUERY="${OUTPUT_DIR}/${QUERY_DATA_NAME}"
    mkdir -p $OUTPUT_DIR_QUERY
    OUTPUT_FILE_DIR="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_after_update_loop_${LOOP_NUM}"
    if [ -z "${OUTPUT_FILE_LIST}" ]; then
      OUTPUT_FILE_LIST="\"${OUTPUT_FILE_DIR}\""
    else
      OUTPUT_FILE_LIST="${OUTPUT_FILE_LIST},\"${OUTPUT_FILE_DIR}\""
    fi
  done
  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_merge_update_weight_loop_${LOOP_NUM}.json"
  python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "merge" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{ "task": "get_weighted_results", "weights_file": "'"${OUTPUT_FILE_PATH}"'", "output_file": ['"${OUTPUT_FILE_LIST}"'],
                        "retrieval_file_ori":['"${ORI_OUTPUT_FILE_LIST}"'], "query_file": ['"${QUERY_FILE_LIST}"'], "retrieval_file_now": ['"${RET_LIST_NOW}"'],
                        "q_llm_file": "'"${TOTAL_RETRIEVAL_OUTPUT_PATH}"'",
                        "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": true, "added_documents": "'"${ADD_FILE_PATH}"'"}'
  wait

  echo "Running merge update for ${RETRIEVAL_MODEL_NAME}"

  LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_merge_update_weight_loop_${LOOP_NUM}.log"
  python ret_arr_pos.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &


  wait

  #对检索后的文档进行评估
  FACT_SCORE_PATH="${ADD_FILE_PATH}/fact_check/"
  RET_LIST_NOW=${OUTPUT_FILE_LIST}
  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_eva_weight_after_update_loop_${LOOP_NUM}.json"
  LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_eva_weight_after_update_loop_${LOOP_NUM}.log"

  python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "nq" \
                        --loop "${LOOP_NUM}" \
                        --stage "weight_init" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{"task":"evaluation", "input_file": ['"${QUERY_FILE_LIST}"'],
                        "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": true,
                        "retrieved_file": ['"${OUTPUT_FILE_LIST}"'],
                        "added_documents": "'"${ADD_FILE_PATH}"'",
                        "new_add_fact_score_file": "'"${FACT_SCORE_PATH}"'"}'

  wait

  echo "Running weight eva for ${RETRIEVAL_MODEL_NAME}"

  python get_init_weight.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
  wait


done
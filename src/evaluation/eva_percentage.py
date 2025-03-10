import json
import re
from collections import defaultdict
# 统计结果中每个查询的召回结果，计算LLM和人类生成文本的数量和比例。数值有可能大于总数，因为一个文档可能被多个查询召回。

# 使用正则表达式来匹配LLM名字
def extract_llm_name(docid):
    match = re.match(r'([a-zA-Z0-9.-]+)_', str(docid))
    return match.group(1) if match else None

def calculate_llm_human_proportion(retrieval_file, topks, llm_names_preset=None):
    with open(retrieval_file, 'r') as file:
        data = json.load(file)

    # 初始化结果字典，包括每个LLM和人类的计数
    results = defaultdict(lambda: defaultdict(int))
    if llm_names_preset:
        for topk in topks:
            for name in llm_names_preset:
                results[topk][name] = 0
            results[topk]['human'] = 0

    # 遍历每个查询
    for query_id, query_data in data.items():
        contexts = query_data['contexts']
        for topk in topks:
            # 获取当前topk的文档
            topk_contexts = contexts[:topk]
            llm_names = defaultdict(int)

            # 计算LLM和人类生成文本的数量
            for ctx in topk_contexts:
                llm_name = extract_llm_name(ctx['docid'])
                if llm_name:
                    llm_names[llm_name] += 1
                else:
                    # 假设数字id表示人类生成的文本
                    llm_names['human'] += 1

            # 更新结果
            for name, count in llm_names.items():
                results[topk][name] += count

    # 计算比例
    proportions = defaultdict(dict)
    # merge results[topk]['chatglm3-6b'] and results[topk]['chatglm3-6b-chat'] into results[topk]['chatglm3-6b']
    for topk in topks:
        if 'chatglm3-6b-chat' in results[topk]:
            results[topk]['chatglm3-6b'] += results[topk]['chatglm3-6b-chat']
            del results[topk]['chatglm3-6b-chat']

    for topk in topks:
        total = sum(results[topk].values())
        for name in results[topk]:
            proportions[topk][f'{name}_proportion'] = results[topk][name] / total
        print(f'Proportions for top-{topk}: {proportions[topk]}')
        print(f'sum: {sum(proportions[topk].values())}')
    # return dict like {5: {'human': 100, 'human_proportion': 0.5, 'gpt-3.5-turbo': 100, 'gpt-3.5-turbo_proportion': 0.5}}

    return proportions



def calculate_llm_type_rate(retrieval_file, topks, llm_names_preset=None):
    """
    Calculates the count and proportion of contexts with 'has_answer' as True and False
    for each LLM and human within the specified top-k documents.
    Additionally, calculates the total counts and proportions for all LLMs excluding human.

    Args:
        retrieval_file (str): Path to the JSON retrieval file.
        topks (list of int): List of top-k values to consider.
        llm_names_preset (list of str, optional): Preset list of LLM names to include. Defaults to None.

    Returns:
        dict: Nested dictionary containing counts and proportions for each top-k,
              including totals for all LLMs excluding human.
    """
    with open(retrieval_file, 'r') as file:
        data = json.load(file)

    # Initialize result dictionary:
    # results[topk][llm_name]['true'/'false'] = count
    results = defaultdict(lambda: defaultdict(lambda: {'true': 0, 'false': 0}))

    # If a preset list of LLM names is provided, initialize their counts to 0
    if llm_names_preset:
        for topk in topks:
            for name in llm_names_preset:
                results[topk][name]['true'] = 0
                results[topk][name]['false'] = 0
            results[topk]['human']['true'] = 0
            results[topk]['human']['false'] = 0

    # Iterate over each query in the data
    for query_id, query_data in data.items():
        contexts = query_data.get('contexts', [])
        for topk in topks:
            # Get the top-k contexts
            topk_contexts = contexts[:topk]
            # Temporary dictionary to count for this query and topk
            llm_names = defaultdict(lambda: {'true': 0, 'false': 0})

            for ctx in topk_contexts:
                llm_name = extract_llm_name(ctx.get('docid'))
                has_answer = ctx.get('has_answer', False)  # Default to False if missing

                if llm_name:
                    if has_answer:
                        llm_names[llm_name]['true'] += 1
                    else:
                        llm_names[llm_name]['false'] += 1
                else:
                    # Assume numeric or missing docid indicates human-generated text
                    if has_answer:
                        llm_names['human']['true'] += 1
                    else:
                        llm_names['human']['false'] += 1

            # Update the overall results with counts from this query
            for name, counts in llm_names.items():
                results[topk][name]['true'] += counts['true']
                results[topk][name]['false'] += counts['false']

    # Merge specific LLM variants if needed (e.g., 'chatglm3-6b-chat' into 'chatglm3-6b')
    for topk in topks:
        merged_name = 'chatglm3-6b'
        variant_name = 'chatglm3-6b-chat'
        if variant_name in results[topk]:
            results[topk][merged_name]['true'] += results[topk][variant_name]['true']
            results[topk][merged_name]['false'] += results[topk][variant_name]['false']
            del results[topk][variant_name]

    # Calculate proportions and add totals for all LLMs excluding human
    proportions = defaultdict(dict)
    for topk in topks:
        # Calculate proportions for each LLM and human
        for name, counts in results[topk].items():
            total = counts['true'] + counts['false']
            if total > 0:
                true_proportion = counts['true'] / total
                false_proportion = counts['false'] / total
            else:
                true_proportion = 0
                false_proportion = 0

            # Store counts and proportions in a flat structure
            proportions[topk][f'{name}_true'] = counts['true']
            proportions[topk][f'{name}_false'] = counts['false']
            proportions[topk][f'{name}_has_answer_true_proportion'] = round(true_proportion, 4)
            proportions[topk][f'{name}_has_answer_false_proportion'] = round(false_proportion, 4)

        # Calculate totals for all LLMs excluding human
        llm_total_true = sum(
            counts['true'] for name, counts in results[topk].items() if name != 'human'
        )
        llm_total_false = sum(
            counts['false'] for name, counts in results[topk].items() if name != 'human'
        )
        llm_total = llm_total_true + llm_total_false

        if llm_total > 0:
            llm_true_proportion = llm_total_true / llm_total
            llm_false_proportion = llm_total_false / llm_total
        else:
            llm_true_proportion = 0
            llm_false_proportion = 0

        # Store totals and their proportions
        proportions[topk]['llms_total_true'] = llm_total_true
        proportions[topk]['llms_total_false'] = llm_total_false
        proportions[topk]['llms_total_has_answer_true_proportion'] = round(llm_true_proportion, 4)
        proportions[topk]['llms_total_has_answer_false_proportion'] = round(llm_false_proportion, 4)

        # Optional: Print the proportions for debugging
        print(f'Proportions for top-{topk}:')
        for key, value in proportions[topk].items():
            print(f'  {key}: {value}')
        print(f'Total proportions sum to: {sum(v for k, v in proportions[topk].items() if "proportion" in k):.4f}\n')

    return proportions


def calculate_query_right_num(retrieval_file, topks, llm_names_preset=None):

    with open(retrieval_file, 'r') as file:
        data = json.load(file)

    # Initialize result dictionary:
    # results[topk][1] = count
    # results[topk][2] = count
    # results[topk][topk-1] = count
    results = defaultdict(lambda: defaultdict(int))


    # Iterate over each query in the data
    for query_id, query_data in data.items():

        contexts = query_data.get('contexts', [])
        for topk in topks:
            # Get the top-k contexts
            count = 0
            topk_contexts = contexts[:topk]
            for ctx in topk_contexts:
                has_answer = ctx.get('has_answer', False)  # Default to False if missing
                if has_answer:
                    count += 1
            results[topk][count] += 1

    return results


# 使用示例
# retrieval_file = 'path_to_your_file.json'  # 替换为实际文件路径
# topks = [5, 20, 50]
# proportions = calculate_llm_human_proportion(retrieval_file, topks)
# print(proportions)

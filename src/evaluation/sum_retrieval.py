#summarze percentage tsvs

import os
import sys
import json
import csv
from collections import defaultdict
import pandas as pd

# path_names = [
#     "mis_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240129064151",
#     "mis_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240124142811",
#     "mis_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240125140045",
#     "mis_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240123121401"
# ]
# path_names = [
#     "nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20231227041949",
#     "nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240113075935",
#     "nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20231229042900",
#     "nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240116050642",
#     "nq_webq_pop_tqa_loop_output_bm25_upr_total_loop_10_20231231125441",
#     "nq_webq_pop_tqa_loop_output_bm25_monot5_total_loop_10_20240101125941",
#     "nq_webq_pop_tqa_loop_output_bm25_bge_total_loop_10_20240103144945",
#     "nq_webq_pop_tqa_loop_output_bge-base_upr_total_loop_10_20240106093905",
#     "nq_webq_pop_tqa_loop_output_bge-base_monot5_total_loop_10_20240108014726",
#     "nq_webq_pop_tqa_loop_output_bge-base_bge_total_loop_10_20240109090024"
# ]

# path_names = [
#     "filter_bleu_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240130134307",
#     "filter_bleu_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240131140843",
#     "filter_bleu_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240131141029",
#     "filter_bleu_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240131141119"
# ]

path_names = [
    "no_update_weight_nq_webq_tqa_pop_loop_output_bge-base_total_loop_10_20250220113402"
]
# path_names = ["no_init_weight_nq_webq_tqa_pop_loop_output_llm-embedder_total_loop_10_20250219155330",
#                # "weight_nq_webq_tqa_pop_loop_output_llm-embedder_total_loop_10_20250131024055",
#                # "filter_bleu_weight_nq_webq_tqa_pop_loop_output_llm-embedder_total_loop_10_20250215044333",
#                # "no_init_weight_nq_webq_tqa_pop_loop_output_llm-embedder_total_loop_10_20250217060754"
#                ]

# path_names = [
#             # "fact_weight_nq_webq_tqa_pop_loop_output_contriever_total_loop_10_20250217160739",
#               "no_init_weight_nq_webq_tqa_pop_loop_output_contriever_total_loop_10_20250219155409",
#               ]

dataset_names = [
    "nq",
    "webq",
    "tqa",
    "pop",
]

method_name='bge-base'


# Function to parse a block of text from the TSV file
def parse_block(block_text, ref_num):
    lines = block_text.strip().split('\n')
    print(lines)

    loop_num = int(lines[0])  # Get the loop number from the first line
    header = lines[1].split('\t')[1:]  # Skip the method name entry
    data = lines[2].split('\t')
    if len(lines) > 3:
        print(lines)
        data2 = lines[3].split('\t')
        print(data2[0])
        if 'update' in data2[0] and '-' not in data2[1]:
            method = [data[0], data2[0]]
            values = [float(x) for x in data[1:]]
            values2 = [float(x) for x in data2[1:]]
            value_all = [values, values2]
            ditc_all = [dict(zip(header, values)), dict(zip(header, values2))]
        else:
            method = [data[0]]
            values = [float(x) for x in data[1:]]
            value_all = [values]
            ditc_all = [dict(zip(header, values))]
    else:
        method = [data[0]]
        values = [float(x) for x in data[1:]]
        value_all = [values]
        ditc_all = [dict(zip(header, values))]
    return method, loop_num, ditc_all, ref_num


# Base directory where the TSV files are located
base_dir = "/home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/ada_journal/SOS-Retrieval-Loop/data_v2/loop_output/DPR"

# Initialize a dictionary to keep all the results
results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# Iterate over each dataset in path_names
for dataset in dataset_names:
    for path_name in path_names:
        # Construct the path to the TSV file
        tsv_path = os.path.join(base_dir, path_name, dataset, "results", f"{dataset}_retrieval.tsv")
        # print('base_dir:', base_dir)
        # print('path_name:', path_name)
        # Check if the TSV file exists
        if os.path.isfile(tsv_path):
            # print(f"Processing {tsv_path}")
            # Open the TSV file and read the contents
            with open(tsv_path, 'r') as file:
                block_text = file.read()
                # print(block_text)
                ref_num = block_text.split('\n')[0].split(': ')[1]  # Get the Ref Num from the first line
                blocks = block_text.strip().split("Loop Num: ")[1:]  # Split the file into blocks
                # print(blo/cks)
                # Parse each block
                for block in blocks:
                    method, loop_num, acc_dict, ref_num = parse_block(block, ref_num)
                    print('method:', method)
                    for i, method_name in enumerate(method):
                        if loop_num not in results_dict[dataset][path_name][method_name]:
                            results_dict[dataset][path_name][method_name][loop_num] = acc_dict[i]
                            results_dict[dataset][path_name]['Ref Num']= ref_num  # Add Ref Num to the dictionary
#print the results_dict
print(json.dumps(results_dict, indent=4))
# Now convert the nested dictionary to a DataFrame for easier handling
dfs = {}

# Iterate over the results to create a DataFrame for each method and dataset
for dataset, paths in results_dict.items():
    df_list = []
    ref_num = None  # Initialize ref_num for each dataset
    for path_name, methods in paths.items():
        if 'Ref Num' in methods:
            ref_num = methods['Ref Num']
        for method, loops in methods.items():
            if method == 'Ref Num':
                continue
            # print('method:', method)
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame.from_dict(loops, orient='index')  # Keep loops as index
            df['Method'] = method
            df['Path Name'] = path_name  # Add path_name as a column
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Loop'}, inplace=True)
            df_list.append(df)

    # Combine all method dataframes for current dataset into one
    dataset_df = pd.concat(df_list, ignore_index=True)
    # Define the desired order for the methods, in lowercase
    method_order = [method_name+'+init', method_name+'+update']

    # # Update the 'Method' column to be categorical with the desired order
    # dataset_df['Method'] = pd.Categorical(dataset_df['Method'], categories=method_order, ordered=True)
    # print(dataset_df)

    # Sort the dataframe first by 'Method' according to the defined order, then by 'Loop'
    dataset_df.sort_values(by=['Path Name', 'Method', 'Loop'], inplace=True)

    # Insert the Ref Num at the beginning
    dataset_df.insert(0, 'Ref Num', ref_num)
    # Change the column order to include Path Name
    dataset_df = dataset_df[['Ref Num', 'Path Name', 'Loop', 'Method', 'acc@5', 'acc@20', 'acc@100']]
    # Add the dataset_df to the dictionary of dataframes
    dfs[dataset] = dataset_df

output_path = "sum_tsvs/low_retrieval_summary.tsv"

# Write the dictionary of dataframes to a TSV file
with open(output_path, 'w') as file:
    for dataset, df in dfs.items():
        file.write(f"Dataset: {dataset}\n")
        df.to_csv(file, sep='\t', index=False)
        file.write('\n')
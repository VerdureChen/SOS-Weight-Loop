import os

# 定义基础目录和文件名列表
basedir = "/home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/ada_journal/SOS-Retrieval-Loop/data_v2/loop_output/DPR" # 请将此路径替换为实际的路径
dataset_names = ["nq", "webq", "tqa", "pop"]
path_names = [
    "no_update_weight_nq_webq_tqa_pop_loop_output_bge-base_total_loop_10_20250220113402",
    # "no_init_weight_nq_webq_tqa_pop_loop_output_contriever_total_loop_10_20250219155409",
    # "filter_bleu_weight_nq_webq_tqa_pop_loop_output_bge-base_total_loop_10_20250215044855",
    # "llm_weight_nq_webq_tqa_pop_loop_output_bge-base_total_loop_10_20250217160412",
    # "filter_source_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240204103944"
]
file_name = "bleu_top5_ref5_trigram.tsv"

# 获取上一级目录名
parent_dir_name = os.path.basename(basedir)

# 定义输出文件
output_file = "sum_tsvs/bleu_output.tsv"

# 打开输出文件准备写入
with open(output_file, "w") as out_file:
    # 遍历每个目录
    for path_name in path_names:
        for dataset in dataset_names:
            # 拼接目录路径
            path_dir = os.path.join(basedir, path_name)
            dataset_dir = os.path.join(path_dir, dataset)+'/results'
            file_path = os.path.join(dataset_dir, file_name)

            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            with open(file_path, "r") as in_file:
                # 读取文件内容
                content = in_file.read()

            # 分割出所有的block
            # print(content)
            blocks = content.split("\n\n")
            print(blocks[0])

            # 处理每个block
            for block in blocks:
                # 检查block是否以llms_total结尾
                line_find = block.strip().split("\n")
                if len(line_find) < 2:
                    continue
                if 'init' in line_find[1] or 'update' in line_find[1]:
                    lines = block.strip().split("\n")


                    # 记录上一级目录名到输出文件
                    out_file.write(f"{dataset} {path_name}\n")

                    # 写入表头和数据
                    # for line in lines[1]:
                    out_file.write(lines[1] + "\n")

                    # 添加一个空行作为分隔
                    out_file.write("\n")

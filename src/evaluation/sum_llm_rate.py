import os

# 定义基础目录和文件名列表
basedir = "/home/xiaoyang2020/chenxiaoyang_11/SOS_benchmark/ada_journal/SOS-Retrieval-Loop/data_v2/loop_output/DPR" # 请将此路径替换为实际的路径
dataset_names = ["nq", "webq", "tqa", "pop"]
path_names = [
            "no_update_weight_nq_webq_tqa_pop_loop_output_bge-base_total_loop_10_20250220113402",
            # "no_init_weight_nq_webq_tqa_pop_loop_output_contriever_total_loop_10_20250219155409",
            # "filter_bleu_weight_nq_webq_tqa_pop_loop_output_llm-embedder_total_loop_10_20250215044333",
            # "filter_source_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240204103944"
]
file_name = "llm_type_rate_top20_ref5.tsv"

# 获取上一级目录名
parent_dir_name = os.path.basename(basedir)

# 定义输出文件
output_file = "sum_tsvs/consolidated_output.tsv"

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
                if line_find[0].strip().endswith("llms_total"):
                    lines = block.strip().split("\n")



                    # 计算并写入total行
                    total_true_line = lines[1].split("\t")[1:]
                    total_false_line = lines[2].split("\t")[1:]

                    # 检查total_true和total_false是否全部为0
                    total_true_values = list(map(int, total_true_line))
                    total_false_values = list(map(int, total_false_line))

                    if all(value == 0 for value in total_true_values) and all(value == 0 for value in total_false_values):
                        continue  # 如果所有值均为0，则跳过该block
                    # 记录上一级目录名到输出文件
                    out_file.write(f"{dataset} {path_name}\n")

                    # 写入表头和数据
                    for line in lines[:3]:
                        out_file.write(line + "\n")

                    # 计算并写入total行
                    total = [
                        str(true + false)
                        for true, false in zip(total_true_values, total_false_values)
                    ]
                    # 写入total行，保留第一列
                    out_file.write("total\t" + "\t".join(total) + "\n")

                    # 查找total_has_answer_true_proportion行
                    for line in lines:
                        if line.startswith("total_has_answer_true_proportion"):
                            out_file.write(line + "\n")

                    # 添加一个空行作为分隔
                    out_file.write("\n")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定字体路径
my_font = FontProperties(fname='../../../../../Rob_LLM/externals/popular-fonts/微软雅黑.ttf')
datasets = ['bge-base', 'llm-embedder', 'contriever']

for dataset in datasets:
    # 读取TSV文件
    file_path = f'png_tsvs/percentage/plot_{dataset}_tsv.tsv'  # 替换成你的TSV文件路径
    df = pd.read_csv(file_path, sep='\t')
    #converting columns names to string type
    df.columns = df.columns.astype(str)
    # df = df.rename(columns={'0': '1'})
    # for i in range(0,10):
    #     df = df.rename(columns={str(i): str(i+1)})

    # 重命名方法
    #bm25
    # contriever
    # bge-base
    # llm-embedder
    # bm25+upr
    # bm25+monot5
    # bm25+bge
    # bge-base+upr
    # bge-base+monot5
    # bge-base+bge
    df['Method'] = df['Method'].replace('vanilla', 'Vanilla')
    df['Method'] = df['Method'].replace('filter_BLEU', 'F_BLEU')
    df['Method'] = df['Method'].replace('filter_source', 'F_Source')
    df['Method'] = df['Method'].replace('llm_weight', 'LLM_W')
    df['Method'] = df['Method'].replace('fact_weight', 'Fact_W')
    # df['Method'] = df['Method'].replace('init_only', 'Init_Only')
    # df['Method'] = df['Method'].replace('update_only', 'Update_Only')
    df['Method'] = df['Method'].replace('init_weighted', 'Init_W')
    df['Method'] = df['Method'].replace('update_weighted', 'Update_W')
    # df['Method'] = df['Method'].replace('bge-base+bge', 'BGE-B+BR')

    # 转换DataFrame为长格式
    df_long = df.melt(id_vars=['Method'], var_name='loop', value_name='LLM_P')

    # 将loop列的数据类型转换为整数，以便在图表中正确排序
    df_long['loop'] = df_long['loop'].astype(str)
    #所有数值数据乘100
    df_long['LLM_P'] = df_long['LLM_P'] * 100
    sns.set_theme(style="ticks", rc={'axes.formatter.limits': (-4, 5)}, font_scale=2.6)
    # 画图
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.figure(figsize=(7.5, 7.5))  # 可以调整图片大小
    sns.lineplot(data=df_long, x='loop', y='LLM_P', hue='Method', palette="deep",linewidth=2.5)
    # plt.title('Self-BLEU Values per Method Over Iterations')  # 可以自定义标题
    sns.despine()
    plt.xlabel('Iteration') #, fontproperties=my_font, fontsize=30, fontweight='bold',)  # X轴标签
    plt.ylabel('Percentage')  # Y轴标签
    # if dataset == 'pop':# or dataset == 'tqa':
    if dataset == 'contriever1':  # or dataset == 'tqa':
        plt.legend(loc='upper right', bbox_to_anchor=(1.6, 0.9), fontsize=25)  # 图例
    else:
        # 不显示图例
        plt.legend([], [], frameon=False)

    # plt.tight_layout()  # 调整布局
    plt.show()  # 显示图表
    plt.savefig(f'png_tsvs/percentage/{dataset}_per.png', bbox_inches='tight')  # 保存图表到文件

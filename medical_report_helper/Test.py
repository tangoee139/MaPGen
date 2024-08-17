import pandas as pd
import os
from tqdm import tqdm
from report_parser import ReportParser
from util import post_label_srt

# 一些注意事项：
# 三个表的名称分别为：部位关键词表_简化，子部位，描述关键词表_简化，向上列名分别为：“是否向上”“是否保留”“是否向上”
#

if __name__ == '__main__':
    fixed_path = 'output/output_diagnosis/'
    keywords_path = "data/膝关节概念关键词表（新）.xlsx"
    keywords_path_old = os.path.join(fixed_path, "data/膝关节关键词表.xlsx")

    # 测试数据
    # file_path = "../data/膝关节MR_test.xls"
    # df_data = pd.read_excel(file_path)

    # 全部数据
    df_a = pd.read_excel('data/膝关节MR2020.xls')
    df_b = pd.read_excel('data/膝关节MR2021-22.xls')
    df_data = pd.concat([df_a,df_b])

    reportparser = ReportParser(keywords_path)
    df_data = df_data.fillna('')
    # 去除文件中的术后数据，仅保留术前数据
    df_data = df_data[~df_data['征象描述'].str.contains('术后')]
    df_data = df_data[~df_data['诊断结论'].str.contains('术后')]

    tqdm.pandas(desc="Processing_get_normal_label")
    df_data['parse_normal_label'] = df_data['诊断结论'].progress_apply(
        lambda x: post_label_srt("；".join([item.get_normal_label() for item in reportparser.parse_report(x)]))
)
    tqdm.pandas(desc="Processing_get_upword_label")
    df_data['parse_upword_label'] = df_data['诊断结论'].progress_apply(
        lambda x: post_label_srt("；".join([item.get_upword_label() for item in reportparser.parse_report(x)])))

    df_data[['诊断结论', 'parse_normal_label','parse_upword_label']].to_csv(os.path.join(fixed_path, "test.csv"), index=False)
    print(df_data)

    # 统计不同标签的数量
    # list_parse = reportparser.parse_report_list(df_data['诊断结论'].tolist())
    # print(list_parse)

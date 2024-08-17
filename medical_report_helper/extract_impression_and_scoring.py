# 读取data\train_14_finding_impression.csv
import pandas as pd
import pandas as pd
import os
from tqdm import tqdm
from report_parser import ReportParser
from util import post_label_srt
from scoer import calculate_f1_scores

def extract_impression(filepath):
    df = pd.read_csv(filepath, dtype=object)
    #去除original中的空格
    df['original'] = df['original'].str.replace(' ','')
    #去除generated中的空格
    df['generated'] = df['generated'].str.replace(' ','')

    # 将df中的original列，提取出 诊断结论之后的所有内容包括'\n'等特殊字符   
    # 将df中的generated列，提取出 诊断结论之后的所有内容包括'\n'等特殊字符

    df['original_impression'] = df['original'].str.extract('(?s).*诊断结论：(.*)', expand=False)
    df['generated_impression'] = df['generated'].str.extract('(?s).*诊断结论：(.*)', expand=False)

    # 仅提取出诊断结论之后，征象描述之前的内容
    # df['original_impression'] = df['original'].str.extract('(?s).*诊断结论：(.*)征象描述：', expand=False)
    # df['generated_impression'] = df['generated'].str.extract('(?s).*诊断结论：(.*)征象描述：', expand=False)

    # df.to_csv('train_findings_impressions_2024-01-12 11:17:27.csv', index=False)
    return df

def parse(df):
    fixed_path = 'output/output_diagnosis/'
    keywords_path = "data/膝关节概念关键词表（新）.xlsx"
    keywords_path_old = os.path.join(fixed_path, "data/膝关节关键词表.xlsx")

    df_data = df

    reportparser = ReportParser(keywords_path)

    df_data.dropna(inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    df_data['generated_impression'] = df_data['generated_impression'].apply(lambda x: x.replace(' ', ''))
    df_data['original_impression'] = df_data['original_impression'].apply(lambda x: x.replace(' ', ''))

    tqdm.pandas(desc="Processing")
    df_data['generated_parse'] = df_data['generated_impression'].progress_apply(
        lambda x: post_label_srt("；".join([item.get_upword_label() for item in reportparser.parse_report(x)])))

    df_data['original_parse'] = df_data['original_impression'].progress_apply(
        lambda x: post_label_srt("；".join([item.get_upword_label() for item in reportparser.parse_report(x)])))

    # df_data.to_csv("output/train_14_impression_Adam.csv", index=False)
    # print(df_data)
    return df_data
if __name__ == '__main__':
    filepath = 'results/train_findings_impressions_2024-01-12 11:18:40/test_output.csv'
    df = extract_impression(filepath)
    df_data = parse(df)
    print(calculate_f1_scores(df_data['generated_parse'],df_data['original_parse']))
import pandas as pd
import os
from tqdm import tqdm
from report_parser import ReportParser
from util import post_label_srt

if __name__ == '__main__':
    fixed_path = 'output/output_diagnosis/'
    keywords_path = "data/膝关节概念关键词表（新）.xlsx"
    keywords_path_old = os.path.join(fixed_path, "data/膝关节关键词表.xlsx")

    df_data = pd.read_csv('data/train_14_impression_Adam.csv', dtype=str)

    reportparser = ReportParser(keywords_path)

    df_data.dropna(inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    df_data['generated'] = df_data['generated'].apply(lambda x: x.replace(' ', ''))
    df_data['original'] = df_data['original'].apply(lambda x: x.replace(' ', ''))

    tqdm.pandas(desc="Processing")
    df_data['generated_parse'] = df_data['generated'].progress_apply(
        lambda x: post_label_srt("；".join([item.get_upword_label() for item in reportparser.parse_report(x)])))

    df_data['original_parse'] = df_data['original'].progress_apply(
        lambda x: post_label_srt("；".join([item.get_upword_label() for item in reportparser.parse_report(x)])))

    df_data.to_csv("output/train_14_impression_Adam.csv", index=False)
    print(df_data)

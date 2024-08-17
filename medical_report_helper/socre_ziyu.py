import os
import pandas as pd
from .report_parser_new import ReportParser
from tqdm import tqdm
from .util import post_label_str, encode_labels, part_supplement, decode_labels, score

def parse_label(df_data, reportparser, column_name, label_set_list):
    # 去掉空格
    df_data[column_name] = df_data[column_name].str.replace(' ', '')

    column_label = f'label_{column_name}'
    tqdm.pandas(desc="Processing_get_normal_label_multi_label")
    df_data[column_label] = df_data[column_name].progress_apply(
        lambda x: post_label_str(r"".join([item.get_normal_label() for item in reportparser.parse_report(x)])))
    df_data[column_label] = df_data[column_label].progress_apply(part_supplement)
    # 转为列表
    df_data[column_label] = df_data[column_label].apply(lambda x: x.split('\n'))

    # encode
    list_data_label = df_data[column_label].tolist()
    encode_label = encode_labels(label_set_list, list_data_label)
    df_data[f'encode_{column_label}'] = encode_label

    # decode
    list_data_encode = df_data[f'encode_{column_label}'].tolist()
    decode_label = decode_labels(label_set_list, list_data_encode)
    df_data[f'decode_{column_label}'] = decode_label

    return df_data


def socre(df_data, column_origin, column_generate, label_set_list, label_set_list_simple):
    # 解析标签
    keywords_path = "/mnt/nfs_share/liuzy/R2GenCMN-main/data/膝关节概念关键词表（新）.xlsx"
    reportparser = ReportParser(keywords_path)
    df_data = parse_label(df_data, reportparser, column_origin, label_set_list)
    df_data = parse_label(df_data, reportparser, column_generate, label_set_list)

    df_data[f'simple_decode_label_{column_origin}'] = df_data[f'decode_label_{column_origin}'].apply(label_simple, label_set_list_simple = label_set_list_simple, multi=True)
    df_data[f'simple_decode_label_{column_generate}'] = df_data[f'decode_label_{column_generate}'].apply(label_simple, label_set_list_simple = label_set_list_simple, multi=True)

    # 将小列表用；连接
    original_list = []
    for i in df_data[f'simple_decode_label_{column_origin}']:
        original_list.append('；'.join(i))
    generated_list = []
    for i in df_data[f'simple_decode_label_{column_generate}']:
        generated_list.append('；'.join(i))

    score_dict = score(generated_list, original_list)

    return score_dict

def label_simple(label, label_set_list_simple, multi = False):
    def simple_iter(label):
        parts = label.split("；")
        result_parts = []
        for part in parts:
            if part and "：" in part:
                key, value = part.split("：", 1)
                if key == "子部位" or key == "程度":
                    continue
                result_parts.append(value)
        # print(result_parts)
        return "".join(result_parts)

    if multi:
        result_list = []
        for item in label:
            result_list.append(simple_iter(item))
        result_list = remove_key(result_list, label_set_list_simple)
        return result_list

    else:
        return simple_iter(label)

def remove_key(label, label_list):
    new_label = []
    for item in label:
        if item in label_list:
            new_label.append(item)
    return new_label


if __name__ == '__main__':
    num_label = 300

    label_file = '../data/label_set_largerData.csv'
    label_set_list = pd.read_csv(label_file)['label'].tolist()

    label_simple_file = '../data/label_set_largerData_removeKey.csv'
    label_set_list_simple = pd.read_csv(label_simple_file)['label_simple'][:num_label].tolist()

    # 列名
    column_origin = 'original'
    column_generate = 'generated'

    df_data = pd.read_csv('test_output.csv')
    score = socre(df_data, column_origin, column_generate, label_set_list, label_set_list_simple)
    print(score)
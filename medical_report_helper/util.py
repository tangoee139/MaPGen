import os
import re
import pandas as pd


# 生成标签后的后处理，例如重复值去掉，并且去掉空的元素
def post_label_str(label_str: str):
    # list_label = re.split(r'[；;]', label_str)
    list_label = re.split(r'\n', label_str)
    unique_labels = []

    # 去掉重复的元素
    for item in list_label:
        if item not in unique_labels and item != '':
            unique_labels.append(item)

    # 去掉空的元素
    if '' in unique_labels:
        unique_labels.remove('')

    # return unique_labels
    return unique_labels

# 标签的编码和解码
def encode_labels(labels, samples, multi = True):
    """
    将标签列表编码为二进制数组。

    :param labels: 可能的标签列表。
    :param samples: 含有标签的样本列表。
    :return: 编码后的二进制数组列表。

    # 示例
    labels = ["红", "绿", "蓝"]
    sample_labels = [["红", "蓝"], ["绿"], ["红", "绿", "蓝"]]

    # 编码
    encoded_samples = encode_labels(labels, sample_labels)
    print("编码后:", encoded_samples)

    # 解码
    decoded_samples = decode_labels(labels, encoded_samples)
    print("解码后:", decoded_samples)

    编码后: [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
    解码后: [['红', '蓝'], ['绿'], ['红', '绿', '蓝']]
    """
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    encoded_samples = []
    # print(samples)
    if multi:
        for sample in samples:
            encoding = [0] * len(label_to_index)
            for label in sample:
                index = label_to_index.get(label)
                if index is not None:
                    encoding[index] = 1
            encoded_samples.append(encoding)
        return encoded_samples
    else:
        encoding = [0] * len(label_to_index)
        for label in samples:
            index = label_to_index.get(label)
            if index is not None:
                encoding[index] = 1
        return encoding

def decode_labels(labels, encoded_samples):
    """
    从二进制数组解码为原始标签列表。

    :param labels: 可能的标签列表。
    :param encoded_samples: 编码后的二进制数组列表。
    :return: 解码后的原始标签列表。
    """
    index_to_label = {idx: label for idx, label in enumerate(labels)}

    decoded_samples = []
    for encoded in encoded_samples:
        sample_labels = [index_to_label[idx] for idx, value in enumerate(encoded) if value == 1]
        decoded_samples.append(sample_labels)

    return decoded_samples

def get_score(csv_file, output_label, list_set_label, save_result = False,
              save_path = '../output/output_diagnosis/label_result_test.csv',
              threshold=0.5):
    '''
    计算F1分数 ,参数是encode的标签
    :param output_label: 生成的标签列表
    :param original_label: 原始的标签列表
    :return: F1分数
    '''

    df_test = pd.read_csv(csv_file)
    original_label = df_test['encode_label'].tolist()
    # 每一个元素转化为list
    original_label = [eval(item) for item in original_label]
    # 取和output一样的长度
    # original_label = original_label[:len(output_label)]

    # 对标签解码
    decoded_output = decode_labels(list_set_label, output_label)
    decoded_original = decode_labels(list_set_label, original_label)

    # 将小列表用；连接
    decoded_output_str = []
    for item in decoded_output:
        decoded_output_str.append("；".join(item))
    decoded_original_str = []
    for item in decoded_original:
        decoded_original_str.append("；".join(item))

    # 合并dataframe
    if save_result:
        df = pd.DataFrame()
        df['image'] = df_test['image']
        df['parse_label_supplement'] = df_test['parse_label_supplement']
        df['generated_parse'] = decoded_output
        df['original_parse'] = decoded_original

        # save
        # 如何前面的路径不存在，创建路径
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        # save_path = save_path.replace('.csv', f'threshold{threshold}.csv')
        df.to_csv(save_path, index=False)
        print(f'save result to {save_path}')

    score_dict = score(decoded_output_str, decoded_original_str)

    return score_dict


def score(generated_list, original_list):
    df = pd.DataFrame({'generated_parse': generated_list, 'original_parse': original_list})

    # 计算微平均准确率和召回率
    true_positives = 0
    total_generated = 0
    total_original = 0

    for _, row in df.iterrows():
        if pd.isna(row['generated_parse']) or row['generated_parse'] == '':
            generated_set = set()
        else:
            generated_set = set(str(row['generated_parse']).split('；'))        
        if pd.isna(row['original_parse']) or row['original_parse'] == '':
            original_set = set()
        else:
            original_set = set(str(row['original_parse']).split('；'))
        total_generated += len(generated_set)
        total_original += len(original_set)

        true_positives += len(generated_set & original_set)

    mic_precision = true_positives / total_generated if total_generated else 0
    mic_recall = true_positives / total_original if total_original else 0
    mic_f1 = 2 * (mic_precision * mic_recall) / (mic_precision + mic_recall) if (mic_precision + mic_recall) else 0

    # 计算宏平均准确率和召回率
    mac_precision = df.apply(lambda row: len(set(str(row['generated_parse']).split('；')) & set(str(row['original_parse']).split('；'))) / len(set(str(row['generated_parse']).split('；'))) if len(set(str(row['generated_parse']).split('；'))) else 0, axis=1).mean()
    mac_recall = df.apply(lambda row: len(set(str(row['generated_parse']).split('；')) & set(str(row['original_parse']).split('；'))) / len(set(str(row['original_parse']).split('；'))) if len(set(str(row['original_parse']).split('；'))) else 0, axis=1).mean()
    mac_f1 = 2 * (mac_precision * mac_recall) / (mac_precision + mac_recall) if (mac_precision + mac_recall) else 0

    # 保留4位小数
    # 保留4位小数
    return {'mic_precision': round(mic_precision, 4),
            'mic_recall': round(mic_recall, 4),
            'mic_f1': round(mic_f1, 4),
            'mac_precision': round(mac_precision, 4),
            'mac_recall': round(mac_recall, 4),
            'mac_f1': round(mac_f1, 4),
            'TP':true_positives,
            'TotalGenerated':total_generated,
            'TotalOriginal':total_original}



def part_supplement(labels:str):
    def parse_string(s):
        # 定义正则表达式
        pattern = r"部位：(.*?)；子部位：(.*?)；描述：(.*?)；程度：(.*?)$"

        # 使用正则表达式匹配
        match = re.match(pattern, s)
        if match:
            # 将匹配到的值转换成字典
            return {
                "部位": match.group(1),
                "子部位": match.group(2),
                "描述": match.group(3),
                "程度": match.group(4)
            }
        else:
            return None

    # list_labels = re.split(r'\n', labels)
    list_labels = labels
    list_labels_dict = []
    for label_item in list_labels:
        list_labels_dict.append(parse_string(label_item))
    for index, label_dict in enumerate(list_labels_dict):
        if label_dict:
            if label_dict['部位'] == '' and label_dict['子部位'] == '':
                list_labels_dict[index]['部位'] = list_labels_dict[index - 1]['部位']
                list_labels_dict[index]['子部位'] = list_labels_dict[index - 1]['子部位']
            if label_dict['部位'] == '':
                list_labels_dict[index]['部位'] = list_labels_dict[index - 1]['部位']

    # 再将字典转换成字符串
    list_labels_str = []
    for label_dict in list_labels_dict:
        if label_dict:
            list_labels_str.append("部位：" + label_dict['部位'] + "；子部位：" + label_dict['子部位']
                                   + "；描述：" + label_dict['描述'] + "；程度：" + label_dict['程度'])
    return "\n".join(list_labels_str)

def label_simple(label, multi = False, is_remove_key = False):
    def simple_iter(label):
        parts = label.split("；")
        result_parts = []
        for part in parts:
            if part and "：" in part:
                key, value = part.split("：", 1)
                if is_remove_key:
                    if key == "子部位" or key == "程度":
                        continue
                result_parts.append(value)
        return "".join(result_parts)

    if multi:
        result_list = []
        for item in label:
            result_list.append(simple_iter(item))
        return result_list

    else:
        return simple_iter(label)


def merge_similar_label(label, multi = False):
    '''
    合并相似的标签
    :param label:
    :return:
    '''

    replace_dict = {
        "部位：膝关节；子部位：；描述：积液；": "部位：膝关节腔；子部位：；描述：积液；",
        "部位：软组织；子部位：；描述：水肿；": "部位：软组织；子部位：；描述：肿胀；",
    }

    def replace_iter(label):
        for key, value in replace_dict.items():
            label = label.replace(key, value)
        return label

    if multi:
        result_list = []
        for item in label:
            result_list.append(replace_iter(item))
        return result_list
    else:
        return replace_iter(label)

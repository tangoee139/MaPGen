import pandas as pd

def calculate_f1_scores(generated_list, original_list):
    df = pd.DataFrame({'generated_parse': generated_list, 'original_parse': original_list})

    # 计算微平均准确率和召回率
    true_positives = 0
    total_generated = 0
    total_original = 0

    for _, row in df.iterrows():
        generated_set = set(str(row['generated_parse']).split('；'))
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
    return {'mic_precision': round(mic_precision, 4),
            'mic_recall': round(mic_recall, 4),
            'mic_f1': round(mic_f1, 4),
            'mac_precision': round(mac_precision, 4),
            'mac_recall': round(mac_recall, 4),
            'mac_f1': round(mac_f1, 4),
            'TP':true_positives,
            'TotalGenerated':total_generated,
            'TotalOriginal':total_original}

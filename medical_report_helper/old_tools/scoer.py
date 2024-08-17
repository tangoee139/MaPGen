import pandas as pd
# 计算微平均准确率和召回率
# 假设 df 是您的 DataFrame
# df['generated'] 和 df['original'] 是需要处理的列
df = pd.read_csv('output/train_14_impression_Adam.csv',dtype=str)
# 初始化变量
true_positives = 0
total_generated = 0
total_original = 0

# 对每一行进行迭代
for _, row in df.iterrows():
    generated_set = set(str(row['generated_parse']).split('；'))
    original_set = set(str(row['original_parse']).split('；'))

    total_generated += len(generated_set)
    total_original += len(original_set)

    true_positives += len(generated_set & original_set)

# 计算微平均准确率和召回率
mic_precision = true_positives / total_generated if total_generated else 0
mic_recall = true_positives / total_original if total_original else 0
print(f"Mic_Precision: {mic_precision}")
print(f"Mic_Recall: {mic_recall}")

# 计算宏平均准确率和召回率
mac_precision = df.apply(lambda row: len(set(str(row['generated_parse']).split('；')) & set(str(row['original_parse']).split('；'))) / len(set(str(row['generated_parse']).split('；'))), axis=1).mean()
mac_recall = df.apply(lambda row: len(set(str(row['generated_parse']).split('；')) & set(str(row['original_parse']).split('；'))) / len(set(str(row['original_parse']).split('；'))), axis=1).mean()
print(f"Mac_Precision: {mac_precision}")
print(f"Mac_Recall: {mac_recall}")

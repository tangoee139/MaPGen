# 读取data\train_14_finding_impression.csv
import pandas as pd
import pandas as pd
import os
from tqdm import tqdm
from report_parser import ReportParser
from util import post_label_srt

df = pd.read_csv('results/train_findings_impressions_2024-01-12 11:17:27/test_output.csv', dtype=object)
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

df.to_csv('train_findings_impressions_2024-01-12 11:17:27.csv', index=False)
print(df)
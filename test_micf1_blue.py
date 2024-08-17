import os
from abc import abstractmethod

import time
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from numpy import inf
import copy
import re
import jieba
from medical_report_helper.report_parser import ReportParser
from medical_report_helper.util import post_label_str, encode_labels, part_supplement, decode_labels, score
from medical_report_helper.socre_ziyu import socre, parse_label
from modules.metrics import compute_scores

label_file = '/mnt/nvme_share/liuzy/R2GenCMN-main/data/label_set_largerData.csv'
label_set_list = pd.read_csv(label_file)['label'][:300].tolist()
label_simple_file = '/mnt/nvme_share/liuzy/R2GenCMN-main/data/label_set_largerData_removeKey.csv'
label_set_list_simple = pd.read_csv(label_simple_file)['label_simple'][:300].tolist()

rows, val_gts, val_res =[], [], []
file_name = 'results/test_output.csv'

#prepare data
reports = []
ground_truths = []
to_be_tested_file = pd.read_csv('/mnt/nvme_share/huy/PromptMRG_qwen2/results/mask_prompt/test_output_val10.csv')
for index, item in to_be_tested_file.iterrows():
	reports.append(item['generated'])
	ground_truths.append(item['original'])

#reports = ['关节少量积液\n右膝关节退行性变','关节少量积液']
#ground_truths = ['右 膝关节 轻度 退行性 变 \n髌 股关节 多发 骨 软骨 损伤 ， 股骨 外侧 髁 为 著 \n关节 少许 积液 ， 髌下 脂肪 垫 水肿','腘窝囊肿']

val_res.extend([' '.join(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]', '', gt))) for i, gt  in enumerate(reports)])
val_gts.extend([' '.join(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]', '', gt))) for i, gt  in enumerate(ground_truths)])

for tes, gts in zip(reports, ground_truths):
    new_row = {'generated': tes.replace(';', '\n'), 'original': gts.replace(';', '\n')}
    rows.append(new_row)
        
df = pd.DataFrame(rows)
df.to_csv(file_name)

val_met = compute_scores({i: [gt] for i, gt  in enumerate(val_gts)},
                           {i: [res] for i, res in enumerate(val_res)})

val_met.update(socre(df, 'original', 'generated', label_set_list,label_set_list_simple))
print(val_met)
#print(**{'val_' + k: v for k, v in val_met.items()})
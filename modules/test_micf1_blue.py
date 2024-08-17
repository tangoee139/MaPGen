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

rows, val_gts, val_res =[], [], []
file_name = 'results/promptmrgknee_logitcls1/test_output.csv'

reports = ['膝 关 节 积 液 内 侧 半 月 板 撕 裂 膝 关 节 退 行 性 骨 关 节 病 膝 关 节 损 伤 右 膝 关 节 退 行 性 骨 关 节 病 ， 多 发 骨 软 骨 损 伤 ; 内 侧 半 月 板 撕 裂 ; 关 节 积 液', '膝 关 节 积 液 右 膝 关 节 少 量 积 液']
ground_truths = ['右膝关节积液;内侧半月 板 撕 裂;骨关节病;aabc;轻轻的你好,kakak;瓜瓜瓜','膝 关 节 积 液;轻轻的你好']
val_res.extend([' '.join(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]', '', gt))) for i, gt  in enumerate(reports)])
val_gts.extend([' '.join(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]', '', gt))) for i, gt  in enumerate(ground_truths)])

for tes, gts in zip(reports, ground_truths):
    new_row = {'generated': tes.replace(';', '\n'), 'original': gts.replace(';', '\n')}
    rows.append(new_row)
        
df = pd.DataFrame(rows)
df.to_csv(file_name)

val_met = compute_scores({i: [gt] for i, gt  in enumerate(val_gts)},
                           {i: [res] for i, res in enumerate(val_res)})

val_met.update(socre(df, self.column_origin, self.column_generate, self.label_set_list,self.label_set_list_simple))
log.update(**{'val_' + k: v for k, v in val_met.items()})
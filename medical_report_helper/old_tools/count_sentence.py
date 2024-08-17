import pandas as pd
from collections import Counter

# 读取CSV文件
df = pd.read_csv('output/output_diagnosis/test.csv',dtype=str)
df.dropna(inplace=True)

# 确保parse列存在
if 'parse' in df.columns:
    # 初始化一个计数器
    count = Counter()

    # 遍历parse列
    for item in df['parse']:
        # 分割字符串，去除空格，然后更新计数器
        count.update([x.strip() for x in item.split('；') if x.strip()])

    # 移除空字符串计数（如果有的话）
    count.pop('', None)

    # 打印总字段数量和每个字段的出现次数
    print(f"总字段数量: {len(count)}")
    print("每个字段的出现次数:")
    for key, value in count.items():
        print(f"{key}: {value}")
    # 将结果保存到CSV文件
    df = pd.DataFrame.from_dict(count, orient='index', columns=['count'])
    df.to_csv('output/output_diagnosis/count.csv')
else:
    print("文件中没有名为 'parse' 的列")

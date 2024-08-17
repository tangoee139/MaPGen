import re
import pandas as pd
import os
from fuzzywuzzy import process,fuzz
from tqdm import tqdm
import logging

# 配置日志记录器
logging.basicConfig(filename='./error.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"*********新的记录*********")

class BuWeiItem():
    def __init__(self, norm_name: str, alias: list = None, parent_name: str = None) -> None:
        self.norm_name = norm_name
        self.alias = alias
        self.parent_name = parent_name

    def __str__(self):
        return f"norm_name: {self.norm_name}, alias: {self.alias},parent_name:{self.parent_name}"


class ZiBuWeiItem():
    def __init__(self, norm_name: str, alias: list = None) -> None:
        self.norm_name = norm_name
        self.alias = alias

    def __str__(self):
        return f"norm_name: {self.norm_name}, alias: {self.alias}"

class final_ResultItem:
    def __init__(self, origin_label:str = '', norm_label:str = '', sentence:str = '', confidence:str = ''):
        self.origin_label = origin_label
        self.norm_label = norm_label
        self.sentence = sentence

        self.confidence  =confidence

    def __str__(self):
        return f"origin_label: {self.origin_label}, norm_label: {self.norm_label}, sentence: {self.sentence}"


class ResultItem:
    def __init__(self, buwei_notchange = '',zibuwei_notchange = '', norm_buwei='', origin_buwei='', norm_zibuwei='',
                 origin_zibuwei='', norm_desc='', origin_desc='', confidence=0, origin_sentence='', upword_buwei = '',
                 upword_zibuwei = '', upword_desc = '',normal_label = '',upword_label = ''):
        self.norm_buwei = norm_buwei # 归一化的部位
        self.origin_buwei = origin_buwei # 原始的部位

        self.norm_zibuwei = norm_zibuwei # 归一化的子部位
        self.origin_zibuwei = origin_zibuwei # 原始的子部位

        self.norm_desc = norm_desc # 归一化的部位
        self.origin_desc = origin_desc # 归一化的描述
        self.confidence = confidence # 评分
        self.origin_sentence = origin_sentence # 预处理之后，匹配之前的句子

        self.buwei_notchange = buwei_notchange # 为了计算分数保留的没有修改的部位
        self.zibuwei_notchange = zibuwei_notchange # 为了计算分数保留的没有修改的子部位

        # 向上之后的部位、子部位和描述
        self.upword_buwei = upword_buwei
        self.upword_zibuwei = upword_zibuwei
        self.upword_desc = upword_desc

        # 归一化的label
        self.normal_label = normal_label
        # 向上之后的label
        self.upword_label = upword_label

    def get_upword_label(self):
        return self.upword_label

    def get_normal_label(self):
        return self.normal_label


    def __str__(self) -> str:
        return (f"norm_buwei: {self.norm_buwei}, origin_buwei: {self.origin_buwei}, norm_zibuwei: {self.norm_zibuwei}, "
                f"origin_zibuwei: {self.origin_zibuwei}, norm_desc: {self.norm_desc}, origin_desc: {self.origin_desc}, "
                f"confidence: {self.confidence}, origin_sentence: {self.origin_sentence}, upword_buwei: {self.upword_buwei}, "
                f"upword_zibuwei: {self.upword_zibuwei}, "
                f"upword_desc: {self.upword_desc}, normal_label: {self.normal_label}, upword_label: {self.upword_label}")


class ReportParser():
    ###
    ### 读取报告并解析，主要针对诊断结论字段
    ###

    ### 资源初始化相关函数
    def __init__(self, keywords_xlsx: str):
        # keywords_xlsx: 关键词表
        # 其中有三个子表: 部位、子部位、描述
        self.keywords_xlsx = keywords_xlsx

        self.df_part_key = pd.read_excel(self.keywords_xlsx, sheet_name="部位关键词表_简化")
        self.df_part_key = self.df_part_key.dropna(subset=['部位'])

        self.df_subPart_key = pd.read_excel(self.keywords_xlsx, sheet_name="子部位")

        self.df_diagnosis_desc = pd.read_excel(self.keywords_xlsx, sheet_name="描述关键词表_简化")
        self.df_diagnosis_desc = self.df_diagnosis_desc.dropna(subset=['描述词'])

        # 待实现函数 self.read_buwei_keywords()
        # self.buwei_dict = {}    # key: 部位名称, value: BuWeiItem
        # self.buwei_alias_map = {}     # key: 别名, value: 部位名称
        # 部位、子部位、描述的读取
        self.read_buwei_keywords()
        self.read_zibuwei_keywords()
        self.read_describe_keywords()

        self.list_knee_keys = list(self.buwei_alias_map.keys())
        # 在每个部位词前添加“膝关节”
        list_knee_keys_add = []
        for item in self.list_knee_keys:
            list_knee_keys_add.append("膝关节" + item)
        self.list_knee_keys.extend(list_knee_keys_add)
        self.list_subPart_key = list(self.zibuwei_alias_map.keys())
        # 描述关键词
        self.list_describe_key = list(self.describe_alias_map.keys())

        # 初始化顿号统一写法字典
        # 要替换为顿号的字
        self.chars_to_check = ["伴", "及", "并", "合并"]

        # 简写替换的内容
        self.replacements = {
            "内、外半月板": "内侧半月板、外侧半月板",
            "内、外侧半月板": "内侧半月板、外侧半月板",
            "外、内侧半月板": "内侧半月板、外侧半月板",
            "外侧、内侧半月板": "内侧半月板、外侧半月板",
            "髌上、髌下脂肪垫": "髌上脂肪垫、髌下脂肪垫",
            "内侧、外侧半月板": "内侧半月板、外侧半月板",
            "'外、内侧半月板": "'内侧半月板、外侧半月板",
            "髌上、下脂肪垫": "髌上脂肪垫、髌下脂肪垫",
            "前、后交叉韧带": "前交叉韧带、后交叉韧带",
            "内、外侧副韧带": "内侧副韧带、外侧副韧带",
            "胫腓骨": "胫骨、腓骨",
            "髌内、外侧支持带": "髌内侧支持带、髌外侧支持带",
            "内侧半月板体部、后角": "内侧半月板体部、内侧半月板后角",
            "腓肠肌内、外侧头": "腓肠肌内侧头、腓肠肌外侧头",
            "内、外侧支持带": "内侧支持带、外侧支持带",
            "内侧半月板后角、后根部": "内侧半月板后角、内侧半月板后根部",
            "内侧半月板后角、根部": "内侧半月板后角、内侧半月板根部",
            "内侧半月板体后部、后角": "内侧半月板体后部、内侧半月板后角"

        }

        # 去掉关键词，如果包含这些关键词则这条短句
        self.keywords = r'(待排|为著|同前|较前|结合|对比|考虑|可能|详见|建议|必要|随诊|原因|假体|未见明显异常|复查)'




    # 将“上位部位的"-"补充完整
    def up_part_restore(self,row, up, normal):
        if row[up] == "-":
            return row[normal]
        else:
            return row[up]

    # 分割句子
    def expand_data_restore(self,df, column_name):
        df[column_name] = df[column_name].str.replace(r'\([^)]*\)', '；', regex=True)
        df[column_name] = df[column_name].str.rstrip('；')
        df[column_name] = df[column_name].str.split('；')
        df = df.explode(column_name, ignore_index=True)
        return df

    # 部位读取
    def read_buwei_keywords(self) -> None:
        self.buwei_dict = {} # 归一化部位中会有哪些别名
        self.buwei_alias_map = {} # 别名对应的归一化部位
        self.dict_part_upor = {} # 部位是否向上
        self.dict_part_normal_super = {} # 部位的上位部位

        # 恢复上位原有的词，现在是“-”
        self.df_part_key["上位部位"] = self.df_part_key.apply(self.up_part_restore, axis=1, args=("上位部位", "归一化部位",))
        # 将部位 “；”为分隔符展开
        self.df_part_key = self.expand_data_restore(self.df_part_key, "部位")


        for _, row in self.df_part_key.iterrows():
            norm_name = row["归一化部位"]
            alias = row["部位"]
            parent_name = row["上位部位"]

            item = BuWeiItem(norm_name, [alias], parent_name)

            # key: 部位名称, value: BuWeiItem
            if norm_name in self.buwei_dict:
                self.buwei_dict[norm_name].alias.append(alias)
            else:
                self.buwei_dict[norm_name] = item

            # key: 别名, value: 部位名称
            self.buwei_alias_map[alias] = norm_name
            # 向上的部位字典
            self.dict_part_normal_super[norm_name] = parent_name

            # 查看这个部位是否需要向上
            self.dict_part_upor[norm_name] = str(row["是否向上"])

        # for value in self.buwei_dict.values():
        #     print(value)

        if not self.check_buwei_keywords():
            pass

    # 子部位读取
    def read_zibuwei_keywords(self) -> None:
        self.zibuwei_dict = {}
        self.zibuwei_alias_map = {}
        self.dict_zibuwei_keepor = {}

        # 将部位 “；”为分隔符展开
        self.df_subPart_key = self.df_subPart_key[['归一化子部位', '子部位','是否保留']]
        self.df_subPart_key = self.expand_data_restore(self.df_subPart_key, "子部位")

        for _, row in self.df_subPart_key.iterrows():
            norm_name = row["归一化子部位"]
            alias = row["子部位"]

            item = ZiBuWeiItem(norm_name, [alias])

            # key: 子部位名称, value: ZiBuWeiItem
            if norm_name in self.zibuwei_dict:
                self.zibuwei_dict[norm_name].alias.append(alias)
            else:
                self.zibuwei_dict[norm_name] = item

            # key: 子别名, value: 子部位名称
            self.zibuwei_alias_map[alias] = norm_name

            # key: 子部位名称, value: 是否保留
            self.dict_zibuwei_keepor[norm_name] = str(row["是否保留"])

        # for value in self.zibuwei_dict.values():
        #     print(value)
        # print(self.zibuwei_alias_map)

    # 描述读取
    def read_describe_keywords(self) -> None:
        self.describe_dict = {}
        self.describe_alias_map = {}
        self.dict_desc_upor = {}
        self.dict_desc_normal_super = {}

        # 恢复上位原有的词，现在是“-”
        self.df_diagnosis_desc["上位描述词"] = self.df_diagnosis_desc.apply(self.up_part_restore, axis=1,
                                                              args=("上位描述词", "归一化描述词",))
        #  “；”为分隔符展开
        self.df_diagnosis_desc = self.expand_data_restore(self.df_diagnosis_desc, "描述词")

        for _, row in self.df_diagnosis_desc.iterrows():
            norm_name = row["归一化描述词"]
            alias = row["描述词"]
            parent_name = row["上位描述词"]

            item = BuWeiItem(norm_name, [alias], parent_name)

            # key: 部位名称, value: BuWeiItem
            if norm_name in self.describe_dict:
                self.describe_dict[norm_name].alias.append(alias)
            else:
                self.describe_dict[norm_name] = item

            # key: 别名, value: 部位名称
            self.describe_alias_map[alias] = norm_name
            # 向上的描述字典
            self.dict_desc_normal_super[norm_name] = parent_name
            # 查看这个描述是否需要向上
            self.dict_desc_upor[norm_name] = str(row["是否向上"])

        # for value in self.describe_dict.values():
        #     print(value)


    def check_buwei_keywords(self) -> bool:
        # 对已有的 self.buwei_dict 和 self.buwei_alias_map 进行检查
        pass

    def pre_parse_report(self, report: str) -> list:
        # report: 待解析的完整诊断结论报告
        # return: 解析结果格式待定义

        # 0. 预处理
        report = self.preprocess_report(report)

        # 1. 标点分句
        sentences = self.split_sentences(report)

        # 2. 逐句预处理
        sentences = [self.preprocess_sentence(sent) for sent in sentences]

        # 3. 顿号处理逻辑
        out_sentences = self.dunhao_split(sentences)
        # 将顿号小列表展开
        out_sentences = self.flatten_list_deep(out_sentences)

        # 4. 逐句解析
        results = [self.parse_sentence(sent) for sent in out_sentences]

        return results

    ### 解析相关函数
    def parse_report(self, report: str) -> list:
        results = self.pre_parse_report(report)
        # 5. 后处理
        # 利用一些规则拆分解析结果，比如“盘状半月板”、“内外侧半月板”
        # final_result = []
        final_result = [self.post_process(res) for res in results if res != ""]

        final_result = self.flatten_list_deep(final_result)

        final_result_2 = []
        for res_item in final_result:
            if res_item.confidence > 1:
                # 查看评分是否有问题
                print(f"分数大于1： {res_item}")
                logging.info(f"分数大于1： {res_item}")

            # origin_label = res_item.origin_buwei + res_item.origin_zibuwei + res_item.origin_desc
            # norm_label = res_item.norm_buwei + res_item.norm_zibuwei + res_item.norm_desc

            # res_item_2 = final_ResultItem(origin_label, norm_label, res_item.origin_sentence,res_item.confidence)
            # final_result_2.append(res_item_2)



        return final_result

    def parse_report_list(self, diagnose_list) -> list:
        # Process each diagnose item
        # processed_list = [self.parse_report(diagnose_item) for diagnose_item in diagnose_list]

        processed_list = [self.parse_report(diagnose_item) for diagnose_item in tqdm(diagnose_list, desc="Processing")]

        flat_list = self.flatten_list_deep(processed_list)

        # Create a list of dictionaries with the required information
        data = [
            {
                'norm_label': self.gen_norm_label(item),
                'sentence': item.origin_sentence,
                'confidence': item.confidence
            } for item in flat_list
        ]

        # 创建DataFrame
        df = pd.DataFrame(data)

        # 计算每个'sentence'的出现次数
        df['出现次数'] = df.groupby('sentence')['sentence'].transform('count')
        df = df[df['出现次数'] >= 5]
        # 移除'sentence'的重复项，并保留第一次出现的'norm_label'
        df_unique = df.drop_duplicates('sentence').reset_index(drop=True)

        # 根据norm_label聚类，并计算每个类的出现次数之和
        grouped = df_unique.groupby('norm_label')['出现次数'].sum().reset_index()

        # 给每个聚类分配一个新的标签
        grouped['cluster_label'] = grouped.index + 1

        # 将聚类的总出现次数合并回原始的DataFrame
        df_result_all = pd.merge(df_unique, grouped, on='norm_label', how='left')
        df_result_all.rename(columns={'出现次数_y': 'cluster_count', '出现次数_x': 'individual_count'}, inplace=True)

        # 按照聚类的总出现次数和单独的出现次数进行排序
        df_result_all.sort_values(by=['cluster_count', 'individual_count', 'cluster_label'],
                                  ascending=[False, False, True], inplace=True)

        # 重置索引
        df_result_all.reset_index(drop=True, inplace=True)

        # 保存到CSV文件
        fixed_path = "../output/output_diagnosis/label_counts.csv"
        df_result_all.to_csv(fixed_path, index=False)

        # label_counts = df['norm_label'].value_counts().reset_index()
        # label_counts.columns = ['norm_label', '出现次数']
        #
        # df_result_all_result.to_csv("../output/output_diagnosis/label_counts.csv", index=False)

        # 计算confidence的平均值

        total_confidence = 0
        res_length = 0
        for item in flat_list:
            try:
                if item.origin_sentence != '':
                    total_confidence += item.confidence
                    res_length += 1
            except AttributeError:
                # print(f"Skipped item due to AttributeError: {item}")
                logging.info(f"Skipped item due to AttributeError: {item}")
                print(f"Skipped item due to AttributeError: {item}")

        average_confidence = total_confidence / res_length
        print("平均confidence值:", average_confidence)
        logging.info(f"平均confidence值: {average_confidence}")
        return processed_list

    def get_upword_label(self, res_item):
        # 生成向上之后的label
        upword_label = []
        flag = 1  # 如果子部位中已经对字部分展开过一次，则子部位不用在处理
        flag_panzhuang = 1  # 如果是盘状半月板，则不用在处理
        # 在后处理中，部位和子部位都有可能是列表，也就是分开了，现在将整个label分开，形成一个列表
        if res_item.buwei_notchange != '' or res_item.zibuwei_notchange != '':
            if res_item.buwei_notchange != '':
                if "盘状半月板" in res_item.upword_buwei:
                    flag_panzhuang = 0
                    upword_label.append("盘状半月板")

                    if res_item.upword_desc != '':
                        upword_label.append("半月板" + res_item.upword_zibuwei + res_item.upword_desc)

                if flag_panzhuang:
                    for item in res_item.upword_buwei:
                        if isinstance(res_item.upword_zibuwei, list):
                            flag = 0
                            for zibuwei_item in res_item.upword_zibuwei:
                                upword_label.append(item + zibuwei_item + res_item.upword_desc)
                        else:
                            # print(item + '   ' +str(res_item.upword_zibuwei) + '   ' + str(res_item.upword_desc))
                            upword_label.append(item + res_item.upword_zibuwei + res_item.upword_desc)
            if flag:
                if res_item.zibuwei_notchange != '':
                    for item in res_item.norm_zibuwei:
                        upword_label.append(res_item.upword_buwei + item + res_item.upword_desc)
        else:
            upword_label.append(res_item.upword_buwei + res_item.upword_zibuwei + res_item.upword_desc)

        # 如果列表为一个元素，直接返回字符串，如果为多个，每个元素之间用“；”分隔
        if len(upword_label) == 1:
            upword_label = upword_label[0]
        else:
            # 展开列表，用“；”分隔
            upword_label = "；".join(upword_label)

        return upword_label


    def gen_norm_label(self,res_item):
        origin_label = []
        norm_label = []
        flag = 1 # 如果子部位中已经对字部分展开过一次，则子部位不用在处理
        flag_panzhuang = 1 # 如果是盘状半月板，则不用在处理
        # 在后处理中，部位和子部位都有可能是列表，也就是分开了，现在将整个label分开，形成一个列表
        if res_item.buwei_notchange != '' or res_item.zibuwei_notchange != '':
            if res_item.buwei_notchange != '':
                if "盘状半月板" in res_item.origin_buwei:
                    flag_panzhuang = 0
                    origin_label.append("盘状半月板")
                    norm_label.append("盘状半月板")

                    if res_item.origin_desc != '':
                        origin_label.append("半月板" + res_item.origin_zibuwei + res_item.origin_desc)
                        norm_label.append("半月板" + res_item.norm_zibuwei + res_item.norm_desc)

                if flag_panzhuang:
                    for item in res_item.norm_buwei:
                        if isinstance(res_item.origin_zibuwei, list):
                            flag = 0
                            for zibuwei_item in res_item.origin_zibuwei:
                                origin_label.append(item + zibuwei_item + res_item.origin_desc)
                                norm_label.append(item + zibuwei_item + res_item.norm_desc)
                        else:
                            # print(item + '   ' +str(res_item.origin_zibuwei) + '   ' + str(res_item.origin_desc))
                            origin_label.append(item + res_item.origin_zibuwei + res_item.origin_desc)
                            norm_label.append(item + res_item.norm_zibuwei + res_item.norm_desc)
            if flag:
                if res_item.zibuwei_notchange != '':
                    for item in res_item.norm_zibuwei:
                        origin_label.append(res_item.origin_buwei + item + res_item.origin_desc)
                        norm_label.append(res_item.norm_buwei + item + res_item.norm_desc)
        else:
            # print(str(res_item.origin_buwei) + str(res_item.origin_zibuwei) + str(res_item.origin_desc),
            #       "origin:" + str(res_item.zibuwei_notchange))
            origin_label.append(res_item.origin_buwei + res_item.origin_zibuwei + res_item.origin_desc)
            norm_label.append(res_item.origin_buwei + res_item.norm_zibuwei + res_item.norm_desc)

        # 如果列表为一个元素，直接返回字符串，如果为多个，每个元素之间用“；”分隔
        if len(norm_label) == 1:
            norm_label = origin_label[0]
        else:
            # 展开列表，用“；”分隔

            norm_label = "；".join(cleaned_list)

        return norm_label

    # 展开大列表
    def flatten_list_deep(self,items):
        flat_list = []
        for item in items:
            if isinstance(item, list):
                flat_list.extend(self.flatten_list_deep(item))
            else:
                flat_list.append(item)
        return flat_list

    def parse_sentence(self, sentence: str) -> ResultItem:
        # sentence: 待解析的句子
        # return: 解析结果
        # 结果是列表，支持一个句子多个结果的情况，比如 “盘状半月板撕裂”
        # 列表中的每个元素是一个 结果对象
        # 集成打分功能
        # norm的形成


        # 主要是去除“?"
        punctuation = r":;；：?？。.,，"
        pattern = re.compile(r"[%s]" % re.escape(punctuation))
        sentence = pattern.sub("", sentence)

        list_knee_keys = self.list_knee_keys
        list_subPart_key = self.list_subPart_key
        list_describe_key = self.list_describe_key

        # 提取的部位
        part = self.find_longest_keyword(sentence, list_knee_keys)
        # 去掉提取的部位剩下的部分
        remove_part = self.remove_substring(sentence, part)
        # 从剩下的部分中提取子部位
        sub_part = self.find_longest_keyword(remove_part, list_subPart_key)
        # 去掉子部位后就是描述
        desc = self.remove_substring_remain(remove_part, sub_part)

        result_item = ResultItem()
        result_item.origin_buwei = part

        # 因为前边在部位关键词以前添加了“膝关节”所以如果不去掉“膝关节“归一化部位就会为空
        # 例如，匹配到了“膝关节内侧半月板”，但是对应的归一化部位中只有“内侧半月板”，这时就需要去掉“膝关节”
        # 最后的粒度问题，如果“是否向上”或者子部位的”是否保留“为“是”，则需要向上

        part_remove = self.remove_substring_remain(part,'膝关节')

        # 部位相关的处理
        if part in self.buwei_alias_map:
            result_item.norm_buwei = self.buwei_alias_map[part]
        elif part_remove in self.buwei_alias_map:
            result_item.norm_buwei = self.buwei_alias_map[part_remove]

        # 部位是否向上
        if result_item.norm_buwei and self.dict_part_upor[result_item.norm_buwei] == "是":
            result_item.upword_buwei = self.dict_part_normal_super[result_item.norm_buwei]
        else:
            result_item.upword_buwei = result_item.norm_buwei

        # 子部位相关的处理
        result_item.origin_zibuwei = sub_part
        if sub_part in self.zibuwei_alias_map:
            result_item.norm_zibuwei = self.zibuwei_alias_map[sub_part]

        # 子部位是否保留
        if result_item.norm_zibuwei and self.dict_zibuwei_keepor[result_item.norm_zibuwei] == "否":
            result_item.upword_zibuwei = ''

        # 描述相关的处理
        result_item.origin_desc = desc
        if desc in self.describe_alias_map:
            result_item.norm_desc = self.describe_alias_map[desc]
        else:
            result_item.origin_desc = self.similar_words(desc,self.list_describe_key,True)
            if result_item.origin_desc:
                result_item.norm_desc = self.describe_alias_map[result_item.origin_desc]

        # 描述是否向上
        if result_item.norm_desc and self.dict_desc_upor[result_item.norm_desc] == "是":
            result_item.upword_desc = self.dict_desc_normal_super[result_item.norm_desc]
        else:
            result_item.upword_desc = result_item.norm_desc



        result_item.origin_sentence = sentence

        result_item.confidence = self.match_score_subpart(result_item)

        return result_item

    # 判读是否向上
    def word_upor(self,row, dict_part_upor, dict_diag_upor, dict_part_normal_super, dict_diag_normal_super):
        part = row['部位']
        desc = row['描述词']

        if part in dict_part_upor and dict_part_upor[part] == "是":
            part = dict_part_normal_super[part]
        if desc in dict_diag_upor and dict_diag_upor[desc] == "是":
            desc = dict_diag_normal_super[desc]

        res = part + desc
        return res

    # 模糊匹配
    def similar_words(self,word, list_words, one):
        if word:
            list_selected = process.extractBests(word, list_words, scorer=fuzz.ratio)  # ,score_cutoff = 80
            if one == False:
                return list_selected
            elif one == True and len(list_selected) > 0 and list_selected[0][1] > 50:
                return list_selected[0][0]
            elif one == True and list_selected[0][1] < 50:
                logging.info(f"匹配的分数小于50：原始值：{word}；匹配到的值：{list_selected[0][0]}")
                return ''
            else:
                return ''
        else:
            return ''


    # 分开部位的工具子函数
    def copy_attributes(self, source: ResultItem, target: ResultItem):
        attributes_to_copy = {
            'norm_buwei': '',
            'origin_buwei': '',
            'norm_zibuwei': '',
            'origin_zibuwei':'',
            'norm_desc': '',
            'origin_desc': '',
            'confidence': 0.0,  # 假设这是一个浮点数属性
            'origin_sentence': ''
        }

        for attr, default_value in attributes_to_copy.items():
            # 检查属性是否为字符串并且为空，或者是浮点数并且为0
            if isinstance(default_value, str) and getattr(target, attr, '') == '':
                setattr(target, attr, getattr(source, attr))
            elif isinstance(default_value, float) and getattr(target, attr, 0.0) == 0.0:
                setattr(target, attr, getattr(source, attr))

    # def post_process(self, res_object: ResultItem) -> List[ResultItem]:
    def post_process(self, res_object: ResultItem) -> ResultItem:
        result_list = []

        keyword = "盘状"
        if keyword in res_object.origin_buwei:
            res_object.buwei_notchange = res_object.origin_buwei
            if "内侧" in res_object.origin_buwei:
                res_object.origin_buwei = ["盘状半月板", "内侧半月板"]
                res_object.norm_buwei = ["盘状半月板", "内侧半月板"]
            elif "外侧" in res_object.origin_buwei:
                res_object.origin_buwei = ["盘状半月板", "外侧半月板"]
                res_object.norm_buwei = ["盘状半月板", "外侧半月板"]
            else:
                res_object.origin_buwei = ["盘状半月板","半月板"]
                res_object.norm_buwei = ["盘状半月板","半月板"]



        for keyword in ["内外", "前后", "肌肉软组织"]:
            if keyword in res_object.origin_buwei:
                res_object.buwei_notchange = res_object.origin_buwei
                if keyword == "肌肉软组织":
                    res_object.norm_buwei = ['肌肉', '软组织']
                else:
                    parts = res_object.origin_buwei.split(keyword)
                    parts_noraml = res_object.norm_buwei.split(keyword)
                    part1, part2 = parts[0], parts[1]
                    part_norm_1, part_norm_2 = parts_noraml[0], parts_noraml[1]
                    res_object.origin_buwei = [part1 + keyword[0] + part2, part1 + keyword[1] + part2]
                    res_object.norm_buwei = [part_norm_1 + keyword[0] + part_norm_2, part_norm_1 + keyword[1] + part_norm_2]

                    # res_1 = ResultItem(origin_buwei=part1 + keyword[0] + part2,norm_buwei=part_norm_1 + keyword[0] + part_norm_2)
                    # res_2 = ResultItem(origin_buwei=part1 + keyword[1] + part2,norm_buwei=part_norm_1 + keyword[1] + part_norm_2)

                # self.copy_attributes(res_object, res_1)
                # result_list.append(res_1)
                #
                # self.copy_attributes(res_object, res_2)
                # result_list.append(res_2)
        res_object.normal_label = self.gen_norm_label(res_object)
        res_object.upword_label = self.get_upword_label(res_object)

        # return result_list if result_list else [res_object]
        return res_object

    # “伴”字替换为“、”，若前边标点，去掉标点
    def preprocess_report(self, report: str) -> str:
        # report: 待预处理的报告
        # return: 预处理结果
        # 处理 “伴囊肿” 相关逻辑
        for item_char in self.chars_to_check:
            pattern = re.compile(rf'[，。！？；：,.;:]{item_char}')
            text_temp = re.sub(pattern, item_char, report)
            report = text_temp.replace(item_char, '、')

        return report

    def preprocess_sentence(self, sentence: str) -> str:
        # sentence: 待预处理的句子
        # return: 预处理结果
        # 为了顿号处理逻辑进行预处理
        # （1）去掉左右膝、（2）空格换顿号、（3）对顿号写法的统一规范
        # 如果包含去掉关键词则为空
        if re.search(self.keywords, sentence):
            return ''

        sentence = sentence.replace('左侧', '').replace('右侧', '').replace('左', '').replace('右', '')
        sentence = sentence.replace(' ', '、')
        if "半月板" not in sentence and "半月" in sentence:
            sentence = sentence.replace("半月", "半月板")

        # 开头“膝”或者“关节”不完整，补全为“膝关节”
        if "膝" in sentence and "膝关节" not in sentence:
            sentence = sentence.replace("膝", "膝关节")
        if sentence.startswith("关节"):
            sentence = sentence.replace("关节", "膝关节")

        # 对顿号写法的统一规范
        if "、" in sentence:
            sentence = sentence.replace("、、", "、")

            pattern = re.compile("|".join(map(re.escape, self.replacements.keys())))
            sentence = pattern.sub(lambda m: self.replacements[m.group()], sentence)

        return sentence

    # 按标点符号分句，[,，。；\n]
    def split_sentences(self, report: str) -> list:
        # report: 待分句的报告
        # return: 分句结果
        desc_list = re.split('[,，。；\n]', str(report))
        desc_list = [s.strip() for s in desc_list if s.strip()]
        return desc_list

    # 查找最长包含的字符串
    def find_longest_keyword(self,string, list_key):
        # 将关键词列表按照长度降序排序，以便后面优先匹配较长的关键词
        sorted_list_key = sorted(list_key, key=len, reverse=True)

        # 遍历关键词列表，查找最长的关键词
        longest_keyword = ""
        for keyword in sorted_list_key:
            if keyword in string:
                # 如果关键词在字符串中出现，更新最长关键词
                if len(keyword) > len(longest_keyword):
                    longest_keyword = keyword

        return longest_keyword

    # 取去掉小字符串后面的内容
    def remove_substring(self,main_string, substring):
        index = main_string.find(substring)
        if index != -1:
            return main_string[index + len(substring):]
        else:
            return main_string

    # 去掉小字符串后是否为空
    def if_remove_none(self,main_string, substring):
        remain = main_string.replace(substring, "")
        if remain == '':
            return True
        else:
            return False

    #
    def remove_substring_remain(self,main_string, substring):
        return main_string.replace(substring, "")

    # 对顿号进行规则补充
    # sentence: 待分顿号的句子
    #     # return: 分顿号结果
    def dunhao_split(self,list_data_test):
        res_supplement = []
        list_knee_keys = self.list_knee_keys
        list_subPart_key = self.list_subPart_key
        for list_item in list_data_test:
            if "、" in list_item:
                list_item = list_item.split("、")

                if len(list_item) == 1:
                    res_supplement.append(list_item)
                #             print('包含部位：',find_part)
                #             print('***补全后：',res_supplement)

                elif len(list_item) == 2:
                    #             print("原始数据：",list_item)
                    find_part = []
                    remove_part = []
                    for item in list_item:
                        result_part = self.find_longest_keyword(item, list_knee_keys)
                        # 去掉提取的部位剩下的部分
                        str_remove_part = self.remove_substring(item, result_part)
                        # 从剩下的部分中提取子部位
                        sub_part = self.find_longest_keyword(str_remove_part, list_subPart_key)
                        # 去掉子部位后就是描述
                        remove_subPart = self.remove_substring_remain(str_remove_part, sub_part)

                        result = '' + result_part + sub_part
                        find_part.append(result)
                        remove_part.append(remove_subPart)

                    # remove_part = []
                    # for i in range(len(find_part)):
                    #     remove_part.append(remove_substring(list_item[i], find_part[i]))

                    # 内侧半月板变性、损伤   11 01
                    if (find_part[0] != '' and remove_part[0] != '') and (find_part[1] == '' and remove_part[1] != ''):
                        list_item[1] = '' + find_part[0] + list_item[1]
                        res_supplement.append(list_item)
                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)

                    # 前交叉韧带、内侧副韧带损伤   10 11
                    elif (find_part[0] != '' and remove_part[0] == '') and (
                            find_part[1] != '' and remove_part[1] != ''):
                        list_item[0] = '' + list_item[0] + self.remove_substring(list_item[1], find_part[1])
                        res_supplement.append(list_item)
                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)

                    # 膝关节退行性变、髌骨骨软骨损伤    11 11
                    elif (find_part[0] != '' and remove_part[0] != '') and (
                            find_part[1] != '' and remove_part[1] != ''):
                        res_supplement.append(list_item)
                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)

                    # 需要补充关键词表或其他处理
                    elif find_part[0] == '' and find_part[1] == '':
                        # print("两个部位都为空：" + str(list_item))
                        logging.info(f"两个部位都为空： {str(list_item)}")

                    # 一个句子被分为两个句子的情况 10 01
                    elif (find_part[0] != '' and remove_part[0] == '') and (
                            find_part[1] == '' and remove_part[1] != ''):
                        res_supplement.append([' '.join(list_item)])
                    else:
                        # 直接分开，不做任何操作
                        res_supplement.append(list_item)
                        # print("其他1个数据：" + str(list_item))
                        logging.info(f"其他1个顿号的数据：{str(list_item)}")
                elif len(list_item) == 3:
                    #             print("原始数据：",list_item)
                    find_part = []
                    remove_part = []
                    for item in list_item:
                        result_part = self.find_longest_keyword(item, list_knee_keys)
                        # 去掉提取的部位剩下的部分
                        str_remove_part = self.remove_substring(item, result_part)
                        # 从剩下的部分中提取子部位
                        sub_part = self.find_longest_keyword(str_remove_part, list_subPart_key)
                        # 去掉子部位后就是描述
                        remove_subPart = self.remove_substring_remain(str_remove_part, sub_part)

                        result = '' + result_part + sub_part
                        find_part.append(result)
                        remove_part.append(remove_subPart)

                    # find_part = []
                    # for item in list_item:
                    #     result = find_longest_keyword(item, list_knee_keys)
                    #     find_part.append(result)
                    #
                    # remove_part = []
                    # for i in range(len(find_part)):
                    #     remove_part.append(remove_substring(list_item[i], find_part[i]))

                    # 关节积液、滑膜炎、滑膜囊肿    11 11 11
                    if (find_part[0] != '' and remove_part[0] != '') and (
                            find_part[1] != '' and remove_part[1] != '') and (
                            find_part[2] != '' and remove_part[2] != ''):
                        res_supplement.append(list_item)
                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)

                    # ['股骨', '胫骨骨挫伤', '微骨折']   10 11 01
                    elif (find_part[0] != '' and remove_part[0] == '') and (
                            find_part[1] != '' and remove_part[1] != '') and (
                            find_part[2] == '' and remove_part[2] != ''):
                        res1 = '' + find_part[0] + remove_part[1]
                        res2 = '' + find_part[1] + remove_part[1]
                        res3 = '' + find_part[0] + remove_part[2]
                        res4 = '' + find_part[1] + remove_part[2]
                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_temp.append(res4)
                        res_supplement.append(res_temp)

                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)

                    # 前交叉韧带、内侧副韧带、髌内侧支持带损伤   10 10 11
                    elif (find_part[0] != '' and remove_part[0] == '') and (
                            find_part[1] != '' and remove_part[1] == '') and (
                            find_part[2] != '' and remove_part[2] != ''):
                        res1 = '' + find_part[0] + remove_part[2]
                        res2 = '' + find_part[1] + remove_part[2]
                        res3 = '' + find_part[2] + remove_part[2]
                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_supplement.append(res_temp)

                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)
                    # 膝关节退行性骨关节病、骨软骨损伤、游离体   11 01 01
                    elif (find_part[0] != '' and remove_part[0] != '') and (
                            find_part[1] == '' and remove_part[1] != '') and (
                            find_part[2] == '' and remove_part[2] != ''):
                        res1 = '' + find_part[0] + remove_part[0]
                        res2 = '' + find_part[0] + remove_part[1]
                        res3 = '' + find_part[0] + remove_part[2]
                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_supplement.append(res_temp)

                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)

                    # 膝髌股关节半脱位、髌骨软骨损伤、骨挫伤   11 11 01
                    elif (find_part[0] != '' and remove_part[0] != '') and (
                            find_part[1] != '' and remove_part[1] != '') and (
                            find_part[2] == '' and remove_part[2] != ''):
                        res1 = '' + find_part[0] + remove_part[0]
                        res2 = '' + find_part[1] + remove_part[1]
                        res3 = '' + find_part[1] + remove_part[2]
                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_supplement.append(res_temp)

                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)
                    #        11 01 11
                    elif (find_part[0] != '' and remove_part[0] != '') and (
                            find_part[1] == '' and remove_part[1] != '') and (
                            find_part[2] != '' and remove_part[2] != ''):
                        res1 = '' + find_part[0] + remove_part[0]
                        res2 = '' + find_part[0] + remove_part[1]
                        res3 = '' + find_part[2] + remove_part[2]
                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_supplement.append(res_temp)

                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)

                    #         10 11 11
                    elif (find_part[0] != '' and remove_part[0] == '') and (
                            find_part[1] != '' and remove_part[1] != '') and (
                            find_part[2] != '' and remove_part[2] != ''):
                        res1 = '' + find_part[0] + remove_part[1]
                        res2 = '' + find_part[1] + remove_part[1]
                        res3 = '' + find_part[2] + remove_part[2]
                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_supplement.append(res_temp)

                    #                 print('包含部位：',find_part)
                    #                 print('***补全后：',res_supplement)
                    # 11 10 11
                    elif (find_part[0] != '' and remove_part[0] != '') and (
                            find_part[1] != '' and remove_part[1] == '') and (
                            find_part[2] != '' and remove_part[2] != ''):
                        res1 = '' + find_part[0] + remove_part[0]
                        res2 = '' + find_part[1] + remove_part[2]
                        res3 = '' + find_part[2] + remove_part[2]
                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_supplement.append(res_temp)

                    else:
                        # print("其他2个顿号数据：" + str(list_item))
                        logging.info(f"其他2个顿号数据：{str(list_item)}")


                elif len(list_item) == 4:
                    #             print("原始数据：",list_item)
                    find_part = []
                    remove_part = []
                    for item in list_item:
                        result_part = self.find_longest_keyword(item, list_knee_keys)
                        # 去掉提取的部位剩下的部分
                        str_remove_part = self.remove_substring(item, result_part)
                        # 从剩下的部分中提取子部位
                        sub_part = self.find_longest_keyword(str_remove_part, list_subPart_key)
                        # 去掉子部位后就是描述
                        remove_subPart = self.remove_substring_remain(str_remove_part, sub_part)

                        result = '' + result_part + sub_part
                        find_part.append(result)
                        remove_part.append(remove_subPart)

                    # find_part = []
                    # for item in list_item:
                    #     result = find_longest_keyword(item, list_knee_keys)
                    #     find_part.append(result)
                    #
                    # remove_part = []
                    # for i in range(len(find_part)):
                    #     remove_part.append(remove_substring(list_item[i], find_part[i]))

                    # 10 11 01 01
                    if (find_part[0] != '' and remove_part[0] == '') and (
                            find_part[1] != '' and remove_part[1] != '') and (
                            find_part[2] == '' and remove_part[2] != '') and (
                            find_part[3] == '' and remove_part[3] != ''):
                        res1 = '' + find_part[0] + remove_part[1]
                        res2 = '' + find_part[0] + remove_part[2]
                        res3 = '' + find_part[0] + remove_part[3]
                        res4 = '' + find_part[1] + remove_part[1]
                        res5 = '' + find_part[1] + remove_part[2]
                        res6 = '' + find_part[1] + remove_part[3]
                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_temp.append(res4)
                        res_temp.append(res5)
                        res_temp.append(res6)
                        res_supplement.append(res_temp)

                    # 10 10 10 11
                    elif (find_part[0] != '' and remove_part[0] == '') and (
                            find_part[1] != '' and remove_part[1] == '') and (
                            find_part[2] != '' and remove_part[2] == '') and (
                            find_part[3] != '' and remove_part[3] != ''):
                        res1 = '' + find_part[0] + remove_part[3]
                        res2 = '' + find_part[1] + remove_part[3]
                        res3 = '' + find_part[2] + remove_part[3]
                        res4 = '' + find_part[3] + remove_part[3]

                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_temp.append(res4)
                        res_supplement.append(res_temp)

                    # 10 10 11 01
                    elif (find_part[0] != '' and remove_part[0] == '') and (
                            find_part[1] != '' and remove_part[1] == '') and (
                            find_part[2] != '' and remove_part[2] != '') and (
                            find_part[3] == '' and remove_part[3] != ''):
                        res1 = '' + find_part[0] + remove_part[2]
                        res2 = '' + find_part[1] + remove_part[2]
                        res3 = '' + find_part[2] + remove_part[2]
                        res4 = '' + find_part[0] + remove_part[3]
                        res5 = '' + find_part[1] + remove_part[3]
                        res6 = '' + find_part[2] + remove_part[3]

                        res_temp = []
                        res_temp.append(res1)
                        res_temp.append(res2)
                        res_temp.append(res3)
                        res_temp.append(res4)
                        res_temp.append(res5)
                        res_temp.append(res6)
                        res_supplement.append(res_temp)

                    else:
                        # print("其他3个顿号数据： " + str(list_item))
                        logging.info(f"其他3个顿号数据： {str(list_item)}")

                else:
                    # print("该原始数据中不是0,1,2,3个顿号： " + str(list_item))
                    logging.info(f"该原始数据中不是0,1,2,3个顿号： {str(list_item)}")
            else:
                res_supplement.append(list_item)
        return res_supplement

    # 模糊匹配的最高分数
    def best_score(self,word, list_words):
        if word:
            list_selected = process.extractBests(word, list_words, scorer=fuzz.ratio)
            return list_selected[0][1]
        else:
            return 0
    # 切分子部位匹配分数
    def match_score_subpart(self,res_object : ResultItem):
        if res_object.buwei_notchange != '':
            origin_part = res_object.buwei_notchange
        else:
            origin_part = res_object.origin_buwei

        if res_object.zibuwei_notchange != '':
            origin_subpart = res_object.zibuwei_notchange
        else:
            origin_subpart = res_object.origin_zibuwei

        origin_part_length = len(origin_part)

        origin_subpart_length = len(origin_subpart)

        origin_desc = res_object.origin_desc

        sentence = res_object.origin_sentence

        # 去掉子部位后就是描述
        remove_part = self.remove_substring(sentence, origin_part)
        desc = self.remove_substring_remain(remove_part, origin_subpart)
        desc_length = len(desc)
        # print("origin_part:",origin_part,"origin_subpart:",origin_subpart,"desc:",desc,"origin_desc:", origin_desc,
        #     "sentence:", sentence)
        res_score = 0
        if sentence:
            sentence_length = len(sentence)

            diagnosis_split = []
            if origin_part:
                diagnosis_split = sentence.split(origin_part, 1)

            if desc in self.describe_alias_map.keys():
                res_score = round(
                    (origin_part_length + origin_subpart_length + desc_length) / sentence_length, 3)
            else:
                if len(diagnosis_split) > 1:
                    best_score = self.best_score(desc,self.describe_alias_map.keys())
                    # print("best_score:",best_score)
                    res_score = round(
                        (origin_part_length + origin_subpart_length + desc_length * round(
                            int(best_score) / 100, 3)) / sentence_length, 3)
                else:
                    # print("有问题：",sentence)
                    logging.info(f"打分有问题：{sentence}")
        # print("res_score:",res_score)
        return res_score

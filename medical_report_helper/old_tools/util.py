import re

# 生成标签后的后处理，例如重复值去掉，并且去掉空的元素
def post_label_srt(label_str: str):
    list_label = re.split(r'[；;]', label_str)
    # print(list_label)
    # list_label = label_str.split("；")
    list_label = list(set(list_label))
    for item in list_label:
        if item == '':
            list_label.remove(item)

    return "；".join(list_label)
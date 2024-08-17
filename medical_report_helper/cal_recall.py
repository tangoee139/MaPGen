def calculate_recall(f1_score, precision):
    # 确保分母不为零
    if 2 * precision - f1_score == 0:
        raise ValueError("Invalid values for precision and F1 score: cannot divide by zero.")
    
    # 根据公式计算Recall
    recall = (f1_score * precision) / (2 * precision - f1_score)
    return recall

# 示例使用
f1_score = 0.587  # 示例F1分数值
precision = 0.641  # 示例查准率值

try:
    recall = calculate_recall(f1_score, precision)
    print(f"Recall: {recall}")
except ValueError as e:
    print(e)
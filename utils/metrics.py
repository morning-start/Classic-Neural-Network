import numpy as np


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算模型预测的准确率

    参数:
    y_pred: np.ndarray - 预测的标签数组，每个元素是一个长度为10的概率分布
    y_true: np.ndarray - 真实的标签数组，每个元素是一个整数表示的类别

    返回:
    acc: float - 预测的准确率
    """
    # 将预测的概率分布转换为预测的类别
    y_pred = y_pred.argmax(axis=1)  # 1-10 那个概率最大
    # 计算预测正确的数量
    correct_predictions = (y_pred == y_true).sum()
    # 计算总样本数
    total_samples = y_true.size
    # 计算准确率
    acc = correct_predictions / total_samples
    return acc

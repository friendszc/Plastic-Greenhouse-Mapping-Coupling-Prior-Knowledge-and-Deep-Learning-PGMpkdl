import torch
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score

# 使用torchmetrics（需要安装：pip install torchmetrics）
def get_metrics(device='cuda'):
    metrics = {
        'precision': Precision(task='binary').to(device),
        'recall': Recall(task='binary').to(device),
        'f1': F1Score(task='binary').to(device)
    }
    return metrics

# 或者手动实现基础版本
def precision(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    true_positives = (y_pred * y_true).sum()
    predicted_positives = y_pred.sum()
    return (true_positives / (predicted_positives + 1e-7))

def recall(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    true_positives = (y_pred * y_true).sum()
    actual_positives = y_true.sum()
    return (true_positives / (actual_positives + 1e-7))

def fmeasure(y_pred, y_true, threshold=0.5):
    p = precision(y_pred, y_true, threshold)
    r = recall(y_pred, y_true, threshold)
    return 2 * (p * r) / (p + r + 1e-7)

# Kappa系数需要自己实现
def kappa_metrics(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    # 实现Cohen's Kappa计算逻辑
    # ...
    return kappa
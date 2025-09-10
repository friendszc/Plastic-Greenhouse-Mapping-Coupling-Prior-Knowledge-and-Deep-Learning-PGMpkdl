import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=1.5):
        super().__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        # 实现加权交叉熵损失
        # pos_weight = torch.tensor([self.weight], device=inputs.device)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        return loss
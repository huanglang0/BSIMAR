import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None):
        """
        纯MAE损失函数
        Args:
            y_pred: 预测值
            y_true: 真实值  
            weights: 权重（可选，为了保持接口一致）
        """
        abs_err = torch.abs(y_pred - y_true)
        
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
            abs_err = abs_err * weights
        
        mae = torch.mean(abs_err)
        return mae
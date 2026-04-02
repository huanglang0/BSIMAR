#工具函数
import numpy as np
import torch
import os

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(paths):
    """创建必要的目录"""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"创建目录: {path}")
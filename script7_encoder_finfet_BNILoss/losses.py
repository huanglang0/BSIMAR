import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang


def get_lds_kernel_window(kernel: str, ks: int, sigma: float):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
        kernel_window = kernel_window / kernel_window.max()
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:  # laplace
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = np.array([laplace(x) for x in range(-half_ks, half_ks + 1)])
        kernel_window = kernel_window / kernel_window.max()
    return kernel_window


def compute_lds_weights_per_target(y_train: np.ndarray,
                                   n_bins: int = 100,
                                   lds_kernel: str = 'gaussian',
                                   lds_ks: int = 5,
                                   lds_sigma: float = 2.0,
                                   strategy: str = 'uniform'):
    """
    Compute LDS weights for each target dimension.
    Input: y_train (N, D) numpy array
    Output: weights (N, D) numpy array
    """
    from sklearn.preprocessing import KBinsDiscretizer
    N, D = y_train.shape
    weights_all = np.ones((N, D), dtype=np.float32)

    for d in range(D):
        y_col = y_train[:, d:d+1]
        # Check if all values are the same
        if y_col.max() == y_col.min():
            weights_all[:, d] = 1.0
            continue

        disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        try:
            discrete = disc.fit_transform(y_col).flatten().astype(int)
        except Exception:
            weights_all[:, d] = 1.0
            continue

        # Empirical count
        counts = np.bincount(discrete, minlength=n_bins).astype(np.float32)
        counts = np.clip(counts, 1e-8, None)

        # Smooth with kernel
        kernel = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        smoothed = convolve1d(counts, weights=kernel, mode='constant')
        smoothed = np.clip(smoothed, 1e-8, None)

        # Inverse frequency
        # eff_counts = smoothed[discrete]
        # weights = 1.0 / eff_counts
        # weights = weights / weights.mean()  # normalize to mean=1
        eff_counts = smoothed[discrete]
        eff_counts = np.clip(eff_counts, 1e-4, None)  # 防止除零或过小
        weights = 1.0 / eff_counts
        weights = np.clip(weights, 0.01, 100.0)       # 限制权重范围
        weights = weights / weights.mean()
        weights_all[:, d] = weights

    return weights_all


class WeightedBNILoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None):
        if y_true.shape[0] <= 1:
            base_loss = torch.mean((y_pred - y_true) ** 2)
            if weights is not None:
                if weights.dim() == 1:
                    weights = weights.unsqueeze(1)
                base_loss = torch.mean((y_pred - y_true) ** 2 * weights)
            return base_loss

        #BNI归一化
        mean_true = y_true.mean(dim=0, keepdim=True)
        std_true = y_true.std(dim=0, keepdim=True) + self.epsilon

        y_true_norm = (y_true - mean_true) / std_true
        y_pred_norm = (y_pred - mean_true) / std_true  

        abs_err = torch.abs(y_pred_norm - y_true_norm)
        sq_err = (y_pred_norm - y_true_norm) ** 2

        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
            abs_err = abs_err * weights
            sq_err = sq_err * weights

        nae = torch.mean(abs_err)
        nse = torch.mean(sq_err)
        return 0.7 * nae + 0.3 * nse
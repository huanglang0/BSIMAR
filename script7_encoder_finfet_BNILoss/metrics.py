#评估指标
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, target_names):
    metrics = {}
    for i, target in enumerate(target_names):
        # 通过目标名称获取列索引
        col_idx = target_names.index(target)

        # 特殊处理qd目标：忽略真实值的绝对值小于0.05e-16的样本
        if target in ['qd', 'qg', 'qb']:
            # 使用绝对值进行过滤
            condition = np.abs(y_true[:, col_idx]) >= 0.05e-16
            y_t = y_true[condition, col_idx]
            y_p = y_pred[condition, col_idx]

            # 检查是否有有效样本
            if len(y_t) == 0:
                metrics[target] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'MRE(%)': np.nan
                }
                continue
        # 新增：特殊处理'ids', 'didsdvg', 'didsdvd'目标：忽略绝对值<=0.0002的样本
        elif target in ['ids', 'didsdvg']:
            # 使用绝对值进行过滤
            condition = np.abs(y_true[:, col_idx]) > 0.0002
            y_t = y_true[condition, col_idx]
            y_p = y_pred[condition, col_idx]

            # 检查是否有有效样本
            if len(y_t) == 0:
                metrics[target] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'MRE(%)': np.nan
                }
                continue
        elif target in ['didsdvd']:
            # 使用绝对值进行过滤
            condition = np.abs(y_true[:, col_idx]) > 0.00005
            y_t = y_true[condition, col_idx]
            y_p = y_pred[condition, col_idx]

            # 检查是否有有效样本
            if len(y_t) == 0:
                metrics[target] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'MRE(%)': np.nan
                }
                continue
        else:
            # 其他目标使用非零样本
            non_zero = y_true[:, col_idx] != 0
            y_t = y_true[non_zero, col_idx]
            y_p = y_pred[non_zero, col_idx]

            if len(y_t) == 0:
                metrics[target] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'MRE(%)': np.nan
                }
                continue

        # 计算指标
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_t, y_p)

        # MRE计算（对qd使用特殊过滤后的样本）
        mre = np.mean(np.abs((y_t - y_p) / y_t)) * 100

        metrics[target] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MRE(%)': mre
        }
    return metrics


def calculate_average_metrics(metrics_dict):
    """计算所有目标的平均指标"""
    avg_metrics = {
        'MSE': 0.0,
        'RMSE': 0.0,
        'R2': 0.0,
        'MRE(%)': 0.0
    }
    valid_targets = 0

    for target, metrics in metrics_dict.items():
        for k, v in metrics.items():
            # 跳过NaN值
            if not np.isnan(v):
                avg_metrics[k] += v
                # 只对有效目标计数
                if k == 'MSE':
                    valid_targets += 1

    # 计算平均值
    if valid_targets > 0:
        for k in avg_metrics:
            avg_metrics[k] /= valid_targets
            
    return avg_metrics
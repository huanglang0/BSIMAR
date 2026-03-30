import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# 只导入 BASE_PLOT_PATH
from config import BASE_PLOT_PATH

def plot_scatter_comparison(y_true, y_pred, target_names, save_path):
    """绘制测试集真实值与预测值的散点图比较"""
    plt.figure(figsize=(20, 15), dpi=100)
    
    for i, target in enumerate(target_names):
        plt.subplot(3, 3, i + 1)  # 3x3网格布局

        # 通过目标名称获取列索引
        col_idx = target_names.index(target)

        # 使用布尔索引过滤零值（保持与原始代码一致的安全检查）
        non_zero = y_true[:, col_idx] != 0
        y_t = y_true[non_zero, col_idx]
        y_p = y_pred[non_zero, col_idx]

        # 计算统计指标
        r2 = r2_score(y_t, y_p)
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)

        # 绘制散点图
        plt.scatter(y_t, y_p, alpha=0.5, label=f'R²={r2:.4f}\nRMSE={rmse:.2e}')
        max_val = max(y_t.max(), y_p.max())
        min_val = min(y_t.min(), y_p.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        # 设置坐标轴标签和刻度字体大小
        plt.xlabel('True Value', fontsize=15)
        plt.ylabel('Predicted Value', fontsize=15)
        plt.title(target, fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True)

        # 增大刻度标签字体
        plt.tick_params(axis='both', which='major', labelsize=25)
        plt.tick_params(axis='both', which='minor', labelsize=25)

    plt.tight_layout()

    # 保存图表 - 仅保存为PNG
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # PNG版本
    scatter_png_path = os.path.join(save_path, "scatter_comparison.png")
    plt.savefig(scatter_png_path, bbox_inches='tight')
    print(f"散点图(PNG)已保存至: {scatter_png_path}")

    plt.show()
    plt.close()

def plot_individual_scatter(y_true, y_pred, target_names, save_path):
    """为每个目标变量单独绘制散点图"""
    os.makedirs(save_path, exist_ok=True)
    
    for i, target in enumerate(target_names):
        plt.figure(figsize=(10, 8), dpi=100)
        
        # 通过目标名称获取列索引
        col_idx = target_names.index(target)

        # 使用布尔索引过滤零值
        non_zero = y_true[:, col_idx] != 0
        y_t = y_true[non_zero, col_idx]
        y_p = y_pred[non_zero, col_idx]

        # 计算统计指标
        r2 = r2_score(y_t, y_p)
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)

        # 绘制散点图
        plt.scatter(y_t, y_p, alpha=0.7, s=50, label=f'R²={r2:.4f}\nRMSE={rmse:.2e}')
        max_val = max(y_t.max(), y_p.max())
        min_val = min(y_t.min(), y_p.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # 设置坐标轴标签和刻度字体大小
        plt.xlabel('True Value', fontsize=16)
        plt.ylabel('Predicted Value', fontsize=16)
        plt.title(f'{target} - True vs Predicted', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)

        # 增大刻度标签字体
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()

        # 创建文件名安全的target名称
        safe_target = target.replace(' ', '_').replace('/', '_')
        
        # 保存单个散点图
        individual_path = os.path.join(save_path, f"scatter_{safe_target}.png")
        plt.savefig(individual_path, bbox_inches='tight')
        print(f"单个散点图 {target} 已保存至: {individual_path}")

        plt.close()

def plot_loss_curves(pretrain_losses, finetune_losses, pretrain_valid_losses=None, finetune_valid_losses=None):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(14, 6))
    
    # 预训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(pretrain_losses, 'b-', linewidth=2, label='训练损失')
    if pretrain_valid_losses is not None:
        plt.plot(pretrain_valid_losses, 'r-', linewidth=2, label='验证损失')
    plt.title('预训练损失曲线', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # 微调损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(finetune_losses, 'b-', linewidth=2, label='训练损失')
    if finetune_valid_losses is not None:
        plt.plot(finetune_valid_losses, 'r-', linewidth=2, label='验证损失')
    plt.title('微调损失曲线', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    
    # 保存损失曲线图 - 使用从config导入的BASE_PLOT_PATH
    os.makedirs(BASE_PLOT_PATH, exist_ok=True)
    loss_curve_path = os.path.join(BASE_PLOT_PATH, "loss_curves.png")
    plt.savefig(loss_curve_path, bbox_inches='tight', dpi=300)
    print(f"损失曲线图已保存至: {loss_curve_path}")
    
    plt.show()
    plt.close()

# 主要绘图函数 - 只包含散点图
def plot_test_results(y_true, y_pred, target_names, save_path):
    """绘制测试结果的可视化（只包含散点图）"""
    print("开始绘制测试结果可视化...")
    
    # 绘制组合散点图
    plot_scatter_comparison(y_true, y_pred, target_names, save_path)
    
    # 绘制单个散点图
    plot_individual_scatter(y_true, y_pred, target_names, save_path)
    
    print("测试结果可视化完成！")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from read_csv import read_csv
import os
import pandas as pd  # 添加pandas库用于CSV操作


# 定义带位置嵌入的模型（无多头注意力）
class DecoderOnlyModel(nn.Module):
    def __init__(self, input_dim=15, target_dim=9, pos_emb_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.pos_emb_dim = pos_emb_dim
        self.total_dim = input_dim + target_dim

        # 位置嵌入层
        self.pos_emb = nn.Embedding(self.total_dim, pos_emb_dim)

        # 特征投影层（处理原始特征+位置嵌入）
        self.feature_proj = nn.Linear(1 + pos_emb_dim, 1)

        # 解码器（保持不变）
        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, target_dim)
        )

    def forward(self, x):
        batch_size, seq_len = x.shape

        # 创建位置索引 [0, 1, 2, ..., total_dim-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        # 获取位置嵌入 [batch_size, seq_len, pos_emb_dim]
        pos_embeddings = self.pos_emb(positions)

        # 将输入特征重塑为三维并添加位置嵌入
        x_reshaped = x.view(batch_size, seq_len, 1)  # [batch, seq, 1]
        x_with_pos = torch.cat([x_reshaped, pos_embeddings], dim=-1)  # [batch, seq, 1+pos_emb_dim]

        # 特征投影
        projected = self.feature_proj(x_with_pos)  # [batch, seq, 1]

        # 移除多头注意力机制后，直接展平投影结果
        projected_flat = projected.squeeze(-1)  # [batch, seq]

        # 解码器
        return self.decoder(projected_flat)


# 新的数据加载函数
def load_dataset(file_list):
    all_inputs = []
    all_outputs = []
    input_head = None
    output_head = None

    for file in file_list:
        i_head, inputs, o_head, outputs = read_csv(file)
        if input_head is None:
            input_head = i_head
            output_head = o_head
        # 校验文件头一致性
        assert i_head == input_head, "输入特征头不匹配"
        assert o_head == output_head, "输出特征头不匹配"

        all_inputs.extend(inputs)
        all_outputs.extend(outputs)

    return (np.array(all_inputs, dtype=np.float64),
            np.array(all_outputs, dtype=np.float64),
            input_head,
            output_head)


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


# 修改后的散点图可视化函数 - 添加PDF保存
def plot_scatter_comparison(y_true, y_pred, target_names):
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

        # 绘制散点图
        plt.scatter(y_t, y_p, alpha=0.5, label=f'R²={r2:.4f}\nMSE={mse:.2e}')
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

    # 保存图表 - 同时保存为PNG和PDF
    os.makedirs("plots", exist_ok=True)
    base_path = "/home/huangl/myprojects/mos_model_nn/picture"

    # PNG版本
    scatter_png_path = os.path.join(base_path, "scatter_comparison.png")
    plt.savefig(scatter_png_path, bbox_inches='tight')
    print(f"散点图(PNG)已保存至: {scatter_png_path}")

    # PDF版本
    scatter_pdf_path = os.path.join(base_path, "scatter_comparison.pdf")
    plt.savefig(scatter_pdf_path, bbox_inches='tight', format='pdf')
    print(f"散点图(PDF)已保存至: {scatter_pdf_path}")

    plt.show()
    plt.close()


# 修改后的VGS变化曲线图函数 - 添加自定义颜色支持
def plot_vgs_comparison(X_test, y_true, y_pred, target_names, input_head):
    # 获取vgs和vds在输入特征中的索引
    vgs_idx = input_head.index('vgs') if 'vgs' in input_head else None
    vds_idx = input_head.index('vds') if 'vds' in input_head else None

    if vgs_idx is None or vds_idx is None:
        print("Warning: 'vgs' or 'vds' not found in input features, skipping VGS plots")
        return

    vgs_values = X_test[:, vgs_idx]
    vds_values = X_test[:, vds_idx]

    # 获取唯一的vds值
    unique_vds = np.unique(vds_values)
    print(f"Found {len(unique_vds)} unique VDS values: {unique_vds}")

    # ====== 自定义颜色方案 ======
    # 定义特定Vds值的自定义颜色
    custom_colors = {
        # 格式: Vds值: (真实值颜色, 预测值颜色)
        0.5: ('red', 'blue'),  # T_vds=0.5用红色，pre_vds=0.5用蓝色
        1: ('green', 'black')  # T_vds=1用绿色，pre_vds=1用紫色
    }

    # 为其他Vds值创建颜色映射
    other_vds = [v for v in unique_vds if v not in custom_colors]
    if other_vds:
        norm = mcolors.Normalize(vmin=min(other_vds), vmax=max(other_vds))
        color_map = cm.viridis
    else:
        norm = None
        color_map = None

    # 为每个目标单独创建图表并保存
    for i, target in enumerate(target_names):
        # 创建新图表
        plt.figure(figsize=(10, 8), dpi=100)  # 调整大小为10x8

        # 获取当前目标的列索引
        col_idx = target_names.index(target)

        # 为每个vds值绘制曲线
        for vds in unique_vds:
            # 筛选当前vds对应的样本
            mask = vds_values == vds
            if np.sum(mask) == 0:
                continue

            vgs_current = vgs_values[mask]
            y_true_current = y_true[mask, col_idx]
            y_pred_current = y_pred[mask, col_idx]

            # 按vgs排序以保证曲线连续
            sort_idx = np.argsort(vgs_current)
            vgs_sorted = vgs_current[sort_idx]
            y_true_sorted = y_true_current[sort_idx]
            y_pred_sorted = y_pred_current[sort_idx]

            # 获取当前vds对应的颜色
            if vds in custom_colors:
                true_color, pred_color = custom_colors[vds]
            else:
                if norm and color_map:
                    color_val = norm(vds)
                    true_color = color_map(color_val)
                    pred_color = color_map(color_val)
                else:
                    true_color = 'blue'
                    pred_color = 'red'

            # 绘制真实值和预测值随Vgs变化的曲线
            plt.plot(vgs_sorted, y_true_sorted, 'o-', markersize=4,
                     label=f'T_vds={vds:.2f}', color=true_color, alpha=0.7)
            plt.plot(vgs_sorted, y_pred_sorted, 'x-', markersize=4,
                     label=f'pre_vds={vds:.2f}', color=pred_color, alpha=0.7, linestyle='--')

        # 设置坐标轴标签和刻度字体大小
        plt.xlabel('Vgs', fontsize=20)
        plt.ylabel(target, fontsize=20)
        plt.title(f'{target} vs Vgs', fontsize=20)
        plt.grid(True)

        # 增大刻度标签字体
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=20)

        # 添加图例
        handles, labels = plt.gca().get_legend_handles_labels()
        # 简化图例：只显示每个vds值的一个条目
        unique_labels = {}
        for handle, label in zip(handles, labels):
            # 移除重复标签
            if label not in unique_labels:
                unique_labels[label] = handle

        # 创建自定义图例
        plt.legend(unique_labels.values(), unique_labels.keys(), fontsize=14, loc='best',
                   ncol=2)  # 分两列显示

        plt.tight_layout()

        # 保存图表 - 为每个目标单独保存PDF
        base_path = "/home/huangl/myprojects/mos_model_nn/picture/fine"

        # 创建文件名安全的target名称
        safe_target = target.replace(' ', '_').replace('/', '_')

        # PDF版本
        vgs_pdf_path = os.path.join(base_path, f"vgs_comparison_{safe_target}.pdf")
        plt.savefig(vgs_pdf_path, bbox_inches='tight', format='pdf')
        print(f"{target} VGS变化曲线图(PDF)已保存至: {vgs_pdf_path}")

        plt.close()


# 随机种子设置
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 保存预测结果到CSV
def save_predictions_to_csv(X_test, y_pred, input_head, target_names, filename):
    """
    将输入特征和预测结果保存到CSV文件

    参数:
        X_test: 测试集输入特征 (numpy数组)
        y_pred: 预测结果 (numpy数组)
        input_head: 输入特征名称列表
        target_names: 目标名称列表
        filename: 要保存的CSV文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 创建DataFrame用于输入特征
    df_inputs = pd.DataFrame(X_test, columns=input_head)

    # 创建DataFrame用于预测结果
    df_predictions = pd.DataFrame(y_pred, columns=target_names)

    # 合并两个DataFrame
    df_combined = pd.concat([df_inputs, df_predictions], axis=1)

    # 保存到CSV
    df_combined.to_csv(filename, index=False)
    print(f"预测结果已保存至: {filename}")


# 主函数
def main():
    set_seed()
    device_id = 2  # 指定使用 GPU 1
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # ================== 配置参数 ==================
    batch_size = 65536
    finetune_epochs = 500

    # ================== 数据加载配置 ==================
    # 加载训练集、验证集和测试集
    train_files = [
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/nmos/SIM/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/nmos/SIM1/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/nmos/SIM2/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/pmos/SIM3/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/pmos/SIM4/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/pmos/SIM5/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/nmos/SIM/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/nmos/SIM1/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/nmos/SIM2/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/pmos/SIM3/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/pmos/SIM5/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/nmos/SIM/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/nmos/SIM1/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/nmos/SIM2/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/pmos/SIM3/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/pmos/SIM4/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/pmos/SIM5/output.csv",
    ]
    valid_files = ["/home/huangl/myprojects/mos_model_nn/data/Valid/finfet_7nm_2000/nch_svt_2000/output.csv"]
    test_files = ["/home/huangl/myprojects/mos_model_nn/data/Test/finfet_7nm_20000/nch_svt_vds=1/output.csv"]

    # ================== 数据加载 ==================
    print("Loading training data for fitting scalers...")
    X_train, y_train_full, input_head, output_head = load_dataset(train_files)
    print("Loading validation data for fine-tuning...")
    X_valid, y_valid_full, _, _ = load_dataset(valid_files)
    print("Loading test data...")
    X_test, y_test_full, _, _ = load_dataset(test_files)

    TARGETS = ['qg', 'qb', 'qd', 'ids', 'didsdvg', 'didsdvd', 'cgg', 'cgd', 'cgs']
    LOG_TARGETS = ['ids', 'didsdvg', 'didsdvd']  # 需要Log处理的目标
    ABS_TARGETS = ['ids', 'didsdvg', 'didsdvd']  # 需要取绝对值的目标

    # ================== 数据处理 ==================
    # 对特定目标取绝对值
    print("\nApplying absolute value to specific targets...")
    abs_indices = [output_head.index(t) for t in ABS_TARGETS]
    for i in abs_indices:
        y_train_full[:, i] = np.abs(y_train_full[:, i])
        y_valid_full[:, i] = np.abs(y_valid_full[:, i])
        y_test_full[:, i] = np.abs(y_test_full[:, i])

    # 提取目标列
    y_train = y_train_full[:, [output_head.index(t) for t in TARGETS]]
    y_valid = y_valid_full[:, [output_head.index(t) for t in TARGETS]]
    y_test = y_test_full[:, [output_head.index(t) for t in TARGETS]]

    # 过滤包含零值的样本
    print("\nFiltering samples where any target is zero...")
    non_zero_mask_train = np.all(y_train != 0, axis=1)
    non_zero_mask_valid = np.all(y_valid != 0, axis=1)
    non_zero_mask_test = np.all(y_test != 0, axis=1)

    X_train, y_train = X_train[non_zero_mask_train], y_train[non_zero_mask_train]
    X_valid, y_valid = X_valid[non_zero_mask_valid], y_valid[non_zero_mask_valid]
    X_test, y_test = X_test[non_zero_mask_test], y_test[non_zero_mask_test]

    print(f"Training samples after filtering: {X_train.shape[0]}")
    print(f"Fine-tuning samples after filtering: {X_valid.shape[0]}")
    print(f"Test samples after filtering: {X_test.shape[0]}")

    # 对特定目标进行Log10处理
    print("\nApplying log10 transformation to specific targets...")
    log_indices = [TARGETS.index(t) for t in LOG_TARGETS]
    for i in log_indices:
        y_train[:, i] = np.log10(y_train[:, i])
        y_valid[:, i] = np.log10(y_valid[:, i])
        y_test[:, i] = np.log10(y_test[:, i])

    # ================== 数据标准化 ==================
    # 使用训练集拟合标准化器
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)

    # 使用训练集的标准化器转换验证集和测试集
    X_valid_scaled = scaler_X.transform(X_valid)
    X_test_scaled = scaler_X.transform(X_test)

    # 为每个目标创建单独的标准化器，使用训练集拟合
    scalers_y = []
    y_train_scaled = np.zeros_like(y_train)
    y_valid_scaled = np.zeros_like(y_valid)
    y_test_scaled = np.zeros_like(y_test)

    for i in range(len(TARGETS)):
        scaler = StandardScaler()
        # 仅使用训练集拟合标准化器
        y_train_scaled[:, i] = scaler.fit_transform(y_train[:, i].reshape(-1, 1)).flatten()
        # 使用训练集拟合的标准化器转换验证集和测试集
        y_valid_scaled[:, i] = scaler.transform(y_valid[:, i].reshape(-1, 1)).flatten()
        y_test_scaled[:, i] = scaler.transform(y_test[:, i].reshape(-1, 1)).flatten()
        scalers_y.append(scaler)
        print(f"Target {TARGETS[i]} - Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")

    # ================== 模型定义 ==================
    model = DecoderOnlyModel(
        input_dim=X_train_scaled.shape[1],
        target_dim=len(TARGETS),
        pos_emb_dim=32
    ).to(device)

    # 模型路径
    pretrain_model_path = "/home/huangl/myprojects/mos_model_nn/model/best_pretrain_model_all.pth"
    finetune_model_path = "/home/huangl/myprojects/mos_model_nn/model/best_finetune_model_n.pth"

    # ================== 加载预训练模型 ==================
    print(f"\nLoading pretrained model from: {pretrain_model_path}")
    model.load_state_dict(torch.load(pretrain_model_path))
    print("Pretrained model loaded successfully")

    criterion = nn.MSELoss()
    # ================== 微调阶段 ==================
    print("\n开始微调...")
    ft_optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_finetune_loss = float('inf')
    finetune_losses = []  # 记录微调损失

    # 使用验证集数据进行微调
    finetune_data = X_valid_scaled
    finetune_targets = y_valid_scaled

    for epoch in range(finetune_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(finetune_data))

        for batch_start in range(0, len(finetune_data), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            X_batch = finetune_data[batch_indices]
            y_batch = finetune_targets[batch_indices]

            ft_optimizer.zero_grad()

            # 初始化输入为目标全为0
            inputs = np.hstack((X_batch, np.zeros((len(batch_indices), len(TARGETS)))))
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

            # 创建真实值张量
            true_tensor = torch.tensor(y_batch, dtype=torch.float32).to(device)

            # 为每个样本生成独立的随机顺序
            orders = np.array([np.random.permutation(len(TARGETS)) for _ in range(len(X_batch))])

            # 存储每个目标的预测结果 - 使用张量
            preds = torch.zeros_like(true_tensor)

            # 逐步预测每个目标
            for step in range(len(TARGETS)):
                # 获取当前步每个样本要预测的目标索引
                current_targets = orders[:, step]

                # 前向传播
                outputs = model(inputs)

                # 收集当前步的预测结果 - 保持张量在计算图中
                row_indices = torch.arange(len(outputs)).to(device)
                col_indices = torch.tensor(current_targets).to(device)
                preds[row_indices, col_indices] = outputs[row_indices, col_indices]

                # 使用真实值更新输入
                inputs[row_indices, X_batch.shape[1] + col_indices] = true_tensor[row_indices, col_indices]

            # 计算损失
            loss = criterion(preds, true_tensor)

            loss_value = loss.item()
            epoch_loss += loss_value
            num_batches += 1

            loss.backward()
            ft_optimizer.step()

        # 计算epoch平均损失
        avg_epoch_loss = epoch_loss / num_batches
        finetune_losses.append(avg_epoch_loss)  # 记录损失

        # 保存最佳微调模型
        if avg_epoch_loss < best_finetune_loss:
            best_finetune_loss = avg_epoch_loss
            torch.save(model.state_dict(), finetune_model_path)
            print(f"保存新的最佳微调模型，损失: {best_finetune_loss:.6f}")

        if epoch % 10 == 0:
            print(f"Finetune Epoch {epoch} | Loss: {avg_epoch_loss:.4f}")

    # 加载微调后的模型
    print(f"\n加载微调模型")
    model.load_state_dict(torch.load(finetune_model_path))

    # ================== 测试阶段 ==================
    model.eval()
    with torch.no_grad():
        pred_scaled = np.zeros_like(y_test_scaled)

        for batch_start in range(0, len(X_test_scaled), batch_size):
            batch_end = batch_start + batch_size
            X_batch = X_test_scaled[batch_start:batch_end]

            # 初始化输入为目标全为0
            test_inputs = np.hstack((X_batch, np.zeros((len(X_batch), len(TARGETS)))))
            test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)

            order = np.arange(len(TARGETS))  # 固定顺序
            preds = {}

            for step in order:
                outputs = model(test_inputs)
                pred = outputs[:, step]
                preds[step] = pred

                # 用当前预测值更新输入
                test_inputs = test_inputs.clone()
                test_inputs[:, X_batch.shape[1] + step] = pred

            # 保存结果
            for t in range(len(TARGETS)):
                pred_t = preds[t].cpu().numpy()
                pred_scaled[batch_start:batch_end, t] = pred_t

        # 逆标准化
        y_pred_log = np.zeros_like(pred_scaled)
        for i in range(len(TARGETS)):
            y_pred_log[:, i] = scalers_y[i].inverse_transform(
                pred_scaled[:, i].reshape(-1, 1)).flatten()

        # 对Log目标进行逆变换
        y_pred = y_pred_log.copy()
        for i in log_indices:
            y_pred[:, i] = 10 ** y_pred[:, i]

        # 恢复原始测试集目标值
        y_true = np.zeros_like(y_test)
        for i in range(len(TARGETS)):
            if i in log_indices:
                y_true[:, i] = 10 ** y_test[:, i]
            else:
                y_true[:, i] = y_test[:, i]

    # ================== 保存预测结果到CSV ==================
    # 创建保存预测结果的路径
    predictions_path = "/home/huangl/myprojects/mos_model_nn/results/fine_csv"
    csv_filename = os.path.join(predictions_path, "model_predictions.csv")

    # 保存预测结果
    save_predictions_to_csv(X_test, y_pred, input_head, TARGETS, csv_filename)

    # ================== 结果可视化与评估 ==================
    print("\n生成散点图...")
    plot_scatter_comparison(y_true, y_pred, TARGETS)

    print("\n生成VGS变化曲线图...")
    plot_vgs_comparison(X_test, y_true, y_pred, TARGETS, input_head)

    print("\n全局评估指标:")
    global_metrics = calculate_metrics(y_true, y_pred, TARGETS)

    # 计算所有目标的平均指标
    avg_metrics = {
        'MSE': 0.0,
        'RMSE': 0.0,
        'R2': 0.0,
        'MRE(%)': 0.0
    }
    valid_targets = 0

    for target, metrics in global_metrics.items():
        print(f"\n{target}:")
        for k, v in metrics.items():
            # 跳过NaN值
            if not np.isnan(v):
                avg_metrics[k] += v
                # 只对有效目标计数
                if k == 'MSE':
                    valid_targets += 1
            print(f"{k}: {v:.4e}" if k != 'MRE(%)' else f"{k}: {v:.2f}%")

    # 计算平均值
    if valid_targets > 0:
        avg_metrics['MSE'] /= valid_targets
        avg_metrics['RMSE'] /= valid_targets
        avg_metrics['R2'] /= valid_targets
        avg_metrics['MRE(%)'] /= valid_targets

    # 打印平均指标
    print("\n所有目标的平均指标:")
    for metric, value in avg_metrics.items():
        if metric != 'MRE(%)':
            print(f"平均{metric}: {value:.4e}")
        else:
            print(f"平均{metric}: {value:.2f}%")


if __name__ == "__main__":
    main()
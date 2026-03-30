# 整合所有组件的主流程，在6的基础上+教师强制，但是没有加随机mask，教师强制是否用于样本，加起始符号
import os
import torch
import numpy as np
from config import *
from model import TransformerEncoderModel  # 修改导入的模型类
from data_processing import load_dataset, preprocess_data, scale_data
from train import pretrain_model, finetune_model, test_model
from metrics import calculate_metrics, calculate_average_metrics
from visualization import plot_scatter_comparison, plot_individual_scatter, plot_loss_curves, plot_test_results
from utils import set_seed, create_directories
import datetime

def save_experiment_report(global_metrics, avg_metrics, save_path):
    """保存实验报告"""
    report_content = f"""
{'='*80}
实验报告 - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}

一、训练参数
{'='*40}
批次大小 (BATCH_SIZE): {BATCH_SIZE}
预训练轮数 (PRETRAIN_EPOCHS): {PRETRAIN_EPOCHS}
微调轮数 (FINETUNE_EPOCHS): {FINETUNE_EPOCHS}
预训练学习率 (LEARNING_RATE_PRETRAIN): {LEARNING_RATE_PRETRAIN}
微调学习率 (LEARNING_RATE_FINETUNE): {LEARNING_RATE_FINETUNE}
权重衰减 (WEIGHT_DECAY): {WEIGHT_DECAY}

二、Early Stopping 参数
{'='*40}
预训练耐心值 (PRETRAIN_PATIENCE): {PRETRAIN_PATIENCE}
微调耐心值 (FINETUNE_PATIENCE): {FINETUNE_PATIENCE}
改进阈值 (DELTA): {DELTA}

三、模型参数
{'='*40}
模型维度 (D_MODEL): {D_MODEL}
注意力头数 (NHEAD): {NHEAD}
编码器层数 (NUM_LAYERS): {NUM_LAYERS}
前馈网络维度 (DIM_FEEDFORWARD): {DIM_FEEDFORWARD}
丢弃率 (DROPOUT): {DROPOUT}

四、目标变量
{'='*40}
{', '.join(TARGETS)}

五、测试结果详细指标
{'='*40}
"""
    
    # 添加每个目标的详细指标
    for target, metrics in global_metrics.items():
        report_content += f"\n{target}:\n"
        report_content += "-" * 30 + "\n"
        for metric_name, value in metrics.items():
            if metric_name == 'MRE(%)':
                report_content += f"  {metric_name}: {value:.4f}%\n"
            else:
                report_content += f"  {metric_name}: {value:.6e}\n"
    
    # 添加平均指标
    report_content += f"\n六、所有目标平均指标\n{'='*40}\n"
    for metric_name, value in avg_metrics.items():
        if metric_name == 'MRE(%)':
            report_content += f"平均{metric_name}: {value:.4f}%\n"
        else:
            report_content += f"平均{metric_name}: {value:.6e}\n"
    
    # 添加文件路径信息
    report_content += f"""
七、文件路径
{'='*40}
预训练模型路径: {PRETRAIN_MODEL_PATH}
微调模型路径: {FINETUNE_MODEL_PATH}
可视化结果路径: {BASE_PLOT_PATH}

八、数据统计
{'='*40}
训练文件数量: {len(TRAIN_FILES)}
验证文件数量: {len(VALID_FILES)}
测试文件数量: {len(TEST_FILES)}
目标变量数量: {len(TARGETS)}
"""
    
    # 保存报告
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n实验报告已保存至: {save_path}")
    return save_path

def main():
    set_seed()
    torch.cuda.set_device(DEVICE_ID)
    
    # 创建必要的目录
    create_directories([BASE_PLOT_PATH, MODEL_SAVE_PATH])
    
    # ================== 数据加载 ==================
    print("Loading training data...")
    X_train, y_train_full, input_head, output_head = load_dataset(TRAIN_FILES)
    print("Loading validation data...")
    X_valid, y_valid_full, _, _ = load_dataset(VALID_FILES)
    print("Loading test data...")
    X_test, y_test_full, _, _ = load_dataset(TEST_FILES)

    # ================== 数据处理 ==================
    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(
        X_train, y_train_full, X_valid, y_valid_full, X_test, y_test_full,
        input_head, output_head, TARGETS
    )
    
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled, scalers_y, scaler_X = scale_data(
        X_train, X_valid, X_test, y_train, y_valid, y_test, TARGETS
    )

    # ================== 模型定义 ==================
    model = TransformerEncoderModel(  # 修改为使用编码器模型
        input_dim=X_train_scaled.shape[1],
        target_dim=len(TARGETS),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    # ================== 预训练阶段 ==================
    # 方法1: 直接传递 patience 值
    pretrain_losses, best_pretrain_loss = pretrain_model(
        model, X_train_scaled, y_train_scaled, TARGETS, DEVICE,
        BATCH_SIZE, PRETRAIN_EPOCHS, LEARNING_RATE_PRETRAIN, WEIGHT_DECAY, 
        PRETRAIN_MODEL_PATH, PRETRAIN_PATIENCE  # 直接传递值，不使用关键字参数
    )
    
    # 或者方法2: 如果函数没有 patience 参数，从 config 中移除并硬编码
    # pretrain_losses, best_pretrain_loss = pretrain_model(
    #     model, X_train_scaled, y_train_scaled, TARGETS, DEVICE,
    #     BATCH_SIZE, PRETRAIN_EPOCHS, LEARNING_RATE_PRETRAIN, WEIGHT_DECAY, 
    #     PRETRAIN_MODEL_PATH
    # )
    
    # 加载最佳预训练模型（早停机制已自动加载）
    print(f"\n加载最佳预训练模型，训练损失: {best_pretrain_loss:.6f}")

    # ================== 微调阶段 ==================
    # 使用验证集作为微调数据
    # 方法1: 直接传递 patience 值
    finetune_losses, best_finetune_loss = finetune_model(
        model, X_valid_scaled, y_valid_scaled, TARGETS, DEVICE,
        BATCH_SIZE, FINETUNE_EPOCHS, LEARNING_RATE_FINETUNE, 
        FINETUNE_MODEL_PATH, FINETUNE_PATIENCE  # 直接传递值，不使用关键字参数
    )
    
    # 或者方法2: 如果函数没有 patience 参数，从 config 中移除并硬编码
    # finetune_losses, best_finetune_loss = finetune_model(
    #     model, X_valid_scaled, y_valid_scaled, TARGETS, DEVICE,
    #     BATCH_SIZE, FINETUNE_EPOCHS, LEARNING_RATE_FINETUNE, 
    #     FINETUNE_MODEL_PATH
    # )
    
    # 加载微调后的模型（早停机制已自动加载）
    print(f"\n加载微调模型，训练损失: {best_finetune_loss:.6f}")

    # 绘制损失曲线（只绘制训练损失）
    print("\n绘制损失曲线...")
    plot_loss_curves(pretrain_losses, finetune_losses)

    # ================== 测试阶段 ==================
    # 使用并行计算进行测试（提供目标值但不用于教师强制）
    y_true, y_pred = test_model(
        model, X_test_scaled, y_test_scaled, TARGETS, DEVICE, 
        BATCH_SIZE, scalers_y
    )

    # ================== 结果可视化与评估 ==================
    print("\n生成测试结果可视化...")
    # 使用新的可视化函数，只生成散点图
    plot_test_results(y_true, y_pred, TARGETS, BASE_PLOT_PATH)

    print("\n全局评估指标:")
    global_metrics = calculate_metrics(y_true, y_pred, TARGETS)

    # 打印每个目标的指标
    for target, metrics in global_metrics.items():
        print(f"\n{target}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4e}" if k != 'MRE(%)' else f"{k}: {v:.2f}%")

    # 计算并打印平均指标
    avg_metrics = calculate_average_metrics(global_metrics)
    print("\n所有目标的平均指标:")
    for metric, value in avg_metrics.items():
        if metric != 'MRE(%)':
            print(f"平均{metric}: {value:.4e}")
        else:
            print(f"平均{metric}: {value:.2f}%")

    # ================== 保存实验报告 ==================
    print("\n保存实验报告...")
    report_path = os.path.join(MODEL_SAVE_PATH, "experiment_report.txt")
    save_experiment_report(global_metrics, avg_metrics, report_path)

    print("\n=== 流程完成 ===")
    print(f"预训练模型: {PRETRAIN_MODEL_PATH}")
    print(f"微调模型: {FINETUNE_MODEL_PATH}")
    print(f"测试结果已保存到: {BASE_PLOT_PATH}")
    print(f"实验报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
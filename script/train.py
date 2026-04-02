import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm
from losses import WeightedBNILoss, compute_lds_weights_per_target
from config import PRETRAIN_PATIENCE, FINETUNE_PATIENCE, DELTA


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, train_loss, model):
        if train_loss < self.best_loss - self.min_delta:
            self.best_loss = train_loss
            self.counter = 0
            # 保存最佳模型
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def pretrain_model(model, X_train_scaled, y_train_scaled, TARGETS, device, 
                  batch_size, pretrain_epochs, learning_rate, weight_decay, model_save_path, patience):
    
    # ✅ 预计算 LDS 权重（只做一次）
    print("正在计算 LDS 权重...")
    lds_weights_np = compute_lds_weights_per_target(
        y_train_scaled, 
        n_bins=100, 
        lds_kernel='gaussian', 
        lds_ks=5, 
        lds_sigma=2.0, 
        strategy='uniform'
    )  # (N, D)
    lds_weights_tensor = torch.tensor(lds_weights_np, dtype=torch.float32).to(device)
    
    criterion = WeightedBNILoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 初始化早停（使用config中的设置）
    early_stopping = EarlyStopping(patience=PRETRAIN_PATIENCE, min_delta=DELTA, save_path=model_save_path)

    print("\n开始预训练...")
    best_pretrain_loss = float('inf')
    pretrain_losses = []

    pbar = tqdm(range(pretrain_epochs), desc="预训练进度", unit="epoch")
    
    for epoch in pbar:
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(X_train_scaled))

        for batch_start in range(0, len(X_train_scaled), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            X_batch = X_train_scaled[batch_indices]
            y_batch = y_train_scaled[batch_indices]

            optimizer.zero_grad()

            inputs = torch.tensor(X_batch, dtype=torch.float32).to(device)
            targets = torch.tensor(y_batch, dtype=torch.float32).to(device)
            weights_batch = lds_weights_tensor[batch_indices]

            outputs = model(inputs, targets)
            loss = criterion(outputs, targets, weights=weights_batch)
            loss_value = loss.item()
            epoch_loss += loss_value
            num_batches += 1

            loss.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / num_batches
        pretrain_losses.append(avg_epoch_loss)

        epoch_time = time.time() - start_time
        pbar.set_postfix({
            'train_loss': f'{avg_epoch_loss:.4f}',
            'best': f'{best_pretrain_loss:.4f}',
            'patience': f'{early_stopping.counter}/{early_stopping.patience}',
            'time': f'{epoch_time:.2f}s'
        })

        if avg_epoch_loss < best_pretrain_loss:
            best_pretrain_loss = avg_epoch_loss

        # 检查早停（基于训练损失）
        if early_stopping(avg_epoch_loss, model):
            pbar.write(f"早停触发！在 epoch {epoch+1} 停止训练")
            break

    pbar.close()
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    return pretrain_losses, best_pretrain_loss


def finetune_model(model, X_finetune, y_finetune, TARGETS, device, 
                  batch_size, finetune_epochs, learning_rate, model_save_path, patience):
    
    # ✅ 微调阶段也建议用 LDS 权重（基于微调数据分布）
    print("正在计算微调数据的 LDS 权重...")
    lds_weights_np = compute_lds_weights_per_target(
        y_finetune, 
        n_bins=100, 
        lds_kernel='gaussian', 
        lds_ks=5, 
        lds_sigma=2.0, 
        strategy='uniform'
    )
    lds_weights_tensor = torch.tensor(lds_weights_np, dtype=torch.float32).to(device)
    
    ft_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = WeightedBNILoss()
    
    # 初始化早停（使用config中的设置）
    early_stopping = EarlyStopping(patience=FINETUNE_PATIENCE, min_delta=DELTA, save_path=model_save_path)
    
    print("\n开始微调...")
    best_finetune_loss = float('inf')
    finetune_losses = []

    pbar = tqdm(range(finetune_epochs), desc="微调进度", unit="epoch")
    
    for epoch in pbar:
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(X_finetune))

        for batch_start in range(0, len(X_finetune), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            X_batch = X_finetune[batch_indices]
            y_batch = y_finetune[batch_indices]

            ft_optimizer.zero_grad()

            inputs = torch.tensor(X_batch, dtype=torch.float32).to(device)
            targets = torch.tensor(y_batch, dtype=torch.float32).to(device)
            weights_batch = lds_weights_tensor[batch_indices]

            outputs = model(inputs, targets)
            loss = criterion(outputs, targets, weights=weights_batch)
            loss_value = loss.item()
            epoch_loss += loss_value
            num_batches += 1

            loss.backward()
            ft_optimizer.step()

        avg_epoch_loss = epoch_loss / num_batches
        finetune_losses.append(avg_epoch_loss)

        epoch_time = time.time() - start_time
        pbar.set_postfix({
            'train_loss': f'{avg_epoch_loss:.4f}',
            'best': f'{best_finetune_loss:.4f}',
            'patience': f'{early_stopping.counter}/{early_stopping.patience}',
            'time': f'{epoch_time:.2f}s'
        })

        if avg_epoch_loss < best_finetune_loss:
            best_finetune_loss = avg_epoch_loss

        # 检查早停（基于训练损失）
        if early_stopping(avg_epoch_loss, model):
            pbar.write(f"早停触发！在 epoch {epoch+1} 停止训练")
            break

    pbar.close()
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    return finetune_losses, best_finetune_loss


def test_model(model, X_test_scaled, y_test_scaled, TARGETS, device, batch_size, scalers_y):
    model.eval()
    with torch.no_grad():
        pred_scaled = np.zeros_like(y_test_scaled)
        pbar = tqdm(range(0, len(X_test_scaled), batch_size), desc="测试进度", unit="batch")
        
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(X_test_scaled))
            X_batch = X_test_scaled[batch_start:batch_end]

            inputs = torch.tensor(X_batch, dtype=torch.float32).to(device)
            outputs = model(inputs)  
            pred_scaled[batch_start:batch_end] = outputs.cpu().numpy()

        pbar.close()

        y_pred = np.zeros_like(pred_scaled)
        y_true = np.zeros_like(y_test_scaled)
        for i in range(len(TARGETS)):
            y_pred[:, i] = scalers_y[i].inverse_transform(pred_scaled[:, i].reshape(-1, 1)).flatten()
            y_true[:, i] = scalers_y[i].inverse_transform(y_test_scaled[:, i].reshape(-1, 1)).flatten()
                
    return y_true, y_pred
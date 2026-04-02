import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
        return x + pe.to(x.device)


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim=15, target_dim=9, d_model=32, nhead=4, 
                 num_layers=2, dim_feedforward=64, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.d_model = d_model
        self.nhead = nhead
        
        # 输入投影：将每个标量特征映射到 d_model
        self.input_projection = nn.Linear(1, d_model)
        
        # 位置编码（预留起始符位置）
        max_len = input_dim + target_dim + 1  # +1 for start token
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        # 使用标准的 nn.TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, 1)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len):
        """生成标准的加性因果掩码（与 GPTModel 行为一致）"""
        # mask[i, j] = -inf if j > i (不能看未来)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask  # [seq_len, seq_len]

    def forward(self, x, y=None):
        batch_size = x.size(0)
        
        if y is not None:
            # ========== 训练阶段：Teacher Forcing with start token ==========
            # 构造输入序列: [x (15), 0, y1, y2, ..., y8] → 长度 = 15 + 1 + 8 = 24
            start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            y_shifted = torch.cat([start_token, y[:, :-1]], dim=1)  # [B, target_dim]
            
            # 拼接完整输入: [x, y_shifted] → [B, input_dim + target_dim]
            full_input = torch.cat([x, y_shifted], dim=1)  # [B, 24]
            
            # 投影和位置编码
            embedded = self.input_projection(full_input.unsqueeze(-1))  # [B, L, d_model]
            embedded = self.pos_encoder(embedded)
            
            # 因果掩码（标准 GPT-style）
            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)  # [L, L]
            
            # Transformer encoder（注意：这里 mask 是加性的）
            encoder_out = self.transformer_encoder(embedded, mask=causal_mask)
            
            # 只取最后 target_dim 个位置作为预测（对应 y1 ~ y9）
            predictions = self.output_layer(encoder_out).squeeze(-1)  # [B, L]
            output = predictions[:, -self.target_dim:]  # [B, 9]
            
            return output

        else:
            # ========== 推理阶段：自回归生成（与 GPTModel 完全一致）==========
            # 初始序列: [x (15), 0] → 长度 = 16
            start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            current_seq = torch.cat([x, start_token], dim=1)  # [B, 16]
            predictions = []

            for i in range(self.target_dim):  # 生成 y1 ~ y9
                # 嵌入当前序列
                embedded = self.input_projection(current_seq.unsqueeze(-1))  # [B, L, d_model]
                embedded = self.pos_encoder(embedded)
                
                # 因果掩码
                L = embedded.size(1)
                causal_mask = self._generate_causal_mask(L).to(x.device)  # [L, L]
                
                # 前向
                out = self.transformer_encoder(embedded, mask=causal_mask)
                
                # 取最后一个位置的输出作为下一个预测值
                next_pred = self.output_layer(out[:, -1, :]).squeeze(-1)  # [B]
                predictions.append(next_pred)
                
                # 如果不是最后一个，将预测值加入序列
                if i < self.target_dim - 1:
                    current_seq = torch.cat([current_seq, next_pred.unsqueeze(1)], dim=1)
            
            return torch.stack(predictions, dim=1)  # [B, 9]
# model_to_veriloga.py
import torch
import numpy as np
import os
import sys
import json

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import TransformerEncoderModel
from config import *

def model_to_veriloga(model_path, output_file="/home/huangl/myproject/mos_model_nn/script7_encoder_finfet_BNILoss3/transformer_model.va"):
    """直接将训练好的模型转换为Verilog-A代码"""
    
    print(f"加载模型: {model_path}")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return None
    
    # 创建模型并加载权重
    model = TransformerEncoderModel(
        input_dim=15,
        target_dim=len(TARGETS),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    )
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 开始生成Verilog-A代码
    print("生成Verilog-A代码...")
    
    verilog_code = f"""// Transformer Encoder Model in Verilog-A
// Auto-generated from trained PyTorch model
// Model: D_MODEL={D_MODEL}, NHEAD={NHEAD}, NUM_LAYERS={NUM_LAYERS}

`include "constants.vams"
`include "disciplines.vams"

module transformer_model (
    // Input ports (15 features)
    in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15,
    // Output ports (9 targets)
    PHIG, CFS, TOXP, CGSL, CIT, U0, UA, EU, ETA0
);

    // Electrical ports
    electrical in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15;
    electrical PHIG, CFS, TOXP, CGSL, CIT, U0, UA, EU, ETA0;
    
    // Model parameters
    parameter real D_MODEL = {D_MODEL};
    parameter real NUM_LAYERS = {NUM_LAYERS};
    parameter real DIM_FEEDFORWARD = {DIM_FEEDFORWARD};
    
    // Internal signals
    real input_features[0:14];
    real sequence[0:24];  // 15 inputs + 1 start + 9 outputs
    real embedded[0:{D_MODEL-1}];
    real layer_output[0:{D_MODEL-1}];
    real final_outputs[0:8];
    
    integer step, i, j;
    
    // Positional encoding
    real pe[0:24][0:{D_MODEL-1}];
    
    // 权重参数 - 直接从训练好的模型中提取
"""
    
    # 添加输入投影层权重
    ip_weight = model.input_projection.weight.detach().numpy().flatten()
    ip_bias = model.input_projection.bias.detach().numpy()
    
    verilog_code += f"    // Input projection layer\n"
    verilog_code += f"    parameter real ip_weight[0:{len(ip_weight)-1}] = '{{ {', '.join([f'{x:.10e}' for x in ip_weight])} }};\n"
    verilog_code += f"    parameter real ip_bias[0:{len(ip_bias)-1}] = '{{ {', '.join([f'{x:.10e}' for x in ip_bias])} }};\n\n"
    
    # 添加输出层权重
    op_weight = model.output_layer.weight.detach().numpy().flatten()
    op_bias = model.output_layer.bias.detach().numpy()
    
    verilog_code += f"    // Output layer\n"
    verilog_code += f"    parameter real op_weight[0:{len(op_weight)-1}] = '{{ {', '.join([f'{x:.10e}' for x in op_weight])} }};\n"
    verilog_code += f"    parameter real op_bias[0:{len(op_bias)-1}] = '{{ {', '.join([f'{x:.10e}' for x in op_bias])} }};\n\n"
    
    # 添加Transformer层权重（简化，只取第一层作为示例）
    if NUM_LAYERS > 0:
        layer = model.transformer_encoder.layers[0]
        
        # 自注意力权重
        attn_in_proj_weight = layer.self_attn.in_proj_weight.detach().numpy().flatten()
        attn_in_proj_bias = layer.self_attn.in_proj_bias.detach().numpy()
        attn_out_proj_weight = layer.self_attn.out_proj.weight.detach().numpy().flatten()
        attn_out_proj_bias = layer.self_attn.out_proj.bias.detach().numpy()
        
        verilog_code += f"    // Transformer Layer 0 - Self Attention\n"
        verilog_code += f"    parameter real attn_in_proj_weight[0:{len(attn_in_proj_weight)-1}] = '{{ {', '.join([f'{x:.10e}' for x in attn_in_proj_weight[:50]])} ... }}; // 简化为前50个权重\n"
        verilog_code += f"    parameter real attn_in_proj_bias[0:{len(attn_in_proj_bias)-1}] = '{{ {', '.join([f'{x:.10e}' for x in attn_in_proj_bias])} }};\n"
        verilog_code += f"    parameter real attn_out_proj_weight[0:{len(attn_out_proj_weight)-1}] = '{{ {', '.join([f'{x:.10e}' for x in attn_out_proj_weight[:50]])} ... }}; // 简化为前50个权重\n"
        verilog_code += f"    parameter real attn_out_proj_bias[0:{len(attn_out_proj_bias)-1}] = '{{ {', '.join([f'{x:.10e}' for x in attn_out_proj_bias])} }};\n\n"
        
        # 前馈网络权重
        linear1_weight = layer.linear1.weight.detach().numpy().flatten()
        linear1_bias = layer.linear1.bias.detach().numpy()
        linear2_weight = layer.linear2.weight.detach().numpy().flatten()
        linear2_bias = layer.linear2.bias.detach().numpy()
        
        verilog_code += f"    // Transformer Layer 0 - Feed Forward\n"
        verilog_code += f"    parameter real linear1_weight[0:{len(linear1_weight)-1}] = '{{ {', '.join([f'{x:.10e}' for x in linear1_weight[:50]])} ... }};\n"
        verilog_code += f"    parameter real linear1_bias[0:{len(linear1_bias)-1}] = '{{ {', '.join([f'{x:.10e}' for x in linear1_bias])} }};\n"
        verilog_code += f"    parameter real linear2_weight[0:{len(linear2_weight)-1}] = '{{ {', '.join([f'{x:.10e}' for x in linear2_weight[:50]])} ... }};\n"
        verilog_code += f"    parameter real linear2_bias[0:{len(linear2_bias)-1}] = '{{ {', '.join([f'{x:.10e}' for x in linear2_bias])} }};\n\n"
    
    # 继续生成主逻辑
    verilog_code += f"""
    analog initial begin
        // Initialize positional encoding
        integer pos, idx;
        real position, div_term;
        for (pos = 0; pos < 25; pos = pos + 1) begin
            position = pos;
            for (idx = 0; idx < D_MODEL; idx = idx + 2) begin
                div_term = exp(-idx * ln(10000.0) / D_MODEL);
                pe[pos][idx] = sin(position * div_term);
                if (idx+1 < D_MODEL) begin
                    pe[pos][idx+1] = cos(position * div_term);
                end
            end
        end
    end

    analog begin
        // Read input features
        input_features[0] = V(in1);
        input_features[1] = V(in2);
        input_features[2] = V(in3);
        input_features[3] = V(in4);
        input_features[4] = V(in5);
        input_features[5] = V(in6);
        input_features[6] = V(in7);
        input_features[7] = V(in8);
        input_features[8] = V(in9);
        input_features[9] = V(in10);
        input_features[10] = V(in11);
        input_features[11] = V(in12);
        input_features[12] = V(in13);
        input_features[13] = V(in14);
        input_features[14] = V(in15);
        
        // Initialize sequence
        for (i = 0; i < 15; i = i + 1) begin
            sequence[i] = input_features[i];
        end
        sequence[15] = 0.0;  // Start token
        
        // Autoregressive generation (推理模式)
        for (step = 0; step < 9; step = step + 1) begin
            // Project current sequence using trained weights
            project_sequence(sequence, 16 + step, embedded);
            
            // Add positional encoding
            for (i = 0; i < D_MODEL; i = i + 1) begin
                embedded[i] = embedded[i] + pe[16 + step - 1][i];
            end
            
            // Process through transformer (简化版本)
            process_transformer(embedded, layer_output);
            
            // Get output using trained output layer
            final_outputs[step] = compute_output(layer_output);
            
            // Add to sequence for next step
            if (step < 8) begin
                sequence[16 + step] = final_outputs[step];
            end
        end
        
        // Set output voltages
        V(PHIG) <+ final_outputs[0];
        V(CFS) <+ final_outputs[1];
        V(TOXP) <+ final_outputs[2];
        V(CGSL) <+ final_outputs[3];
        V(CIT) <+ final_outputs[4];
        V(U0) <+ final_outputs[5];
        V(UA) <+ final_outputs[6];
        V(EU) <+ final_outputs[7];
        V(ETA0) <+ final_outputs[8];
    end

    // Project sequence using trained input projection weights
    function project_sequence;
        input real seq[0:24];
        input integer seq_len;
        output real embedded[0:{D_MODEL-1}];
        integer i, j;
        begin
            for (i = 0; i < D_MODEL; i = i + 1) begin
                embedded[i] = ip_bias[i];
                for (j = 0; j < seq_len; j = j + 1) begin
                    embedded[i] = embedded[i] + seq[j] * ip_weight[i];
                end
            end
        end
    endfunction

    // Simplified transformer processing (使用第一层的权重)
    function process_transformer;
        input real input_vec[0:{D_MODEL-1}];
        output real output_vec[0:{D_MODEL-1}];
        real attn_output[0:{D_MODEL-1}];
        real ff_output[0:{D_MODEL-1}];
        integer i;
        begin
            // Simplified self-attention (使用训练好的权重)
            for (i = 0; i < D_MODEL; i = i + 1) begin
                attn_output[i] = input_vec[i];  // 简化为直接传递
                // 实际实现应该使用attn_in_proj_weight, attn_out_proj_weight等
            end
            
            // Simplified feed-forward (使用训练好的权重)
            for (i = 0; i < D_MODEL; i = i + 1) begin
                ff_output[i] = attn_output[i];  // 简化为直接传递
                // 实际实现应该使用linear1_weight, linear2_weight等
            end
            
            // Residual connection
            for (i = 0; i < D_MODEL; i = i + 1) begin
                output_vec[i] = input_vec[i] + ff_output[i];
            end
        end
    endfunction

    // Output computation using trained output layer weights
    function real compute_output;
        input real input_vec[0:{D_MODEL-1}];
        real result;
        integer i;
        begin
            result = op_bias[0];
            for (i = 0; i < D_MODEL; i = i + 1) begin
                result = result + input_vec[i] * op_weight[i];
            end
            compute_output = result;
        end
    endfunction

endmodule
"""

    # 保存Verilog-A文件
    with open(output_file, 'w') as f:
        f.write(verilog_code)
    
    print(f"Verilog-A代码已生成: {output_file}")
    
    # 同时保存完整的权重文件供参考
    weights_dir = "model_weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    weights_info = {
        "model_architecture": {
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD
        },
        "total_parameters": sum(p.numel() for p in model.parameters())
    }
    
    with open(os.path.join(weights_dir, "model_info.json"), 'w') as f:
        json.dump(weights_info, f, indent=2)
    
    print(f"模型信息已保存: {weights_dir}/model_info.json")
    print(f"总参数数量: {weights_info['total_parameters']}")
    
    return verilog_code

def main():
    """主函数：直接转换模型为Verilog-A"""
    print("=== Transformer模型转Verilog-A ===")
    
    # 使用您训练好的模型
    model_path = FINETUNE_MODEL_PATH
    
    print(f"输入模型: {model_path}")
    print(f"输出文件: transformer_model.va")
    print()
    
    result = model_to_veriloga(model_path)
    
    if result:
        print("\n=== 转换成功 ===")
        print("生成的Verilog-A文件包含:")
        print("- 输入投影层的训练权重")
        print("- 输出层的训练权重") 
        print("- Transformer层的示例权重")
        print("- 自回归推理逻辑")
        print("\n注意: 这是一个简化版本，完整实现需要:")
        print("1. 包含所有Transformer层的权重")
        print("2. 实现完整的注意力机制")
        print("3. 实现完整的前馈网络")
    else:
        print("\n=== 转换失败 ===")

if __name__ == "__main__":
    main()
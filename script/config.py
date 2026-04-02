#配置参数
import torch

# 设备配置
DEVICE_ID = 1
torch.cuda.set_device(DEVICE_ID)
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

# 训练参数
BATCH_SIZE = 1024
PRETRAIN_EPOCHS = 500
FINETUNE_EPOCHS = 300
LEARNING_RATE_PRETRAIN = 0.0008
LEARNING_RATE_FINETUNE = 0.0005
WEIGHT_DECAY = 1e-4

# Early Stopping 参数
PRETRAIN_PATIENCE = 10  # 预训练早停耐心值
FINETUNE_PATIENCE = 8   # 微调早停耐心值
DELTA = 1e-5           # 改进阈值

# 模型参数
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.2

# 目标列配置
TARGETS = ['qg', 'qb', 'qd', 'ids', 'didsdvg', 'didsdvd', 'cgg', 'cgd', 'cgs']

# 数据文件路径
TRAIN_FILES = [
        #300k
        "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/nmos/SIM/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/nmos/SIM1/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/nmos/SIM2/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/pmos/SIM3/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/pmos/SIM4/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/pmos/SIM5/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/nmos/SIM/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/nmos/SIM1/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/nmos/SIM2/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/pmos/SIM3/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/pmos/SIM4/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/pmos/SIM5/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/nmos/SIM/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/nmos/SIM1/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/nmos/SIM2/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/pmos/SIM3/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/pmos/SIM4/output.csv",
        "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/pmos/SIM5/output.csv",

        #20k
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs20000/nmos/SIM/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs20000/nmos/SIM1/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs20000/nmos/SIM2/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs20000/pmos/SIM3/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs20000/pmos/SIM4/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs20000/pmos/SIM5/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs20000/nmos/SIM/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs20000/nmos/SIM1/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs20000/nmos/SIM2/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs20000/pmos/SIM3/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs20000/pmos/SIM4/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs20000/pmos/SIM5/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs20000/nmos/SIM/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs20000/nmos/SIM1/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs20000/nmos/SIM2/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs20000/pmos/SIM3/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs20000/pmos/SIM4/output.csv",
        # "/home/huangl/myproject/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs20000/pmos/SIM5/output.csv",
]

VALID_FILES = ["/home/huangl/myproject/mos_model_nn/data/Valid/finfet_7nm_2000/nch_svt_2000/output.csv"]

TEST_FILES = ["/home/huangl/myproject/mos_model_nn/data/Test/finfet_7nm_20000/nch_svt_vds=1/output.csv"]

# 路径配置
BASE_PLOT_PATH = "/home/huangl/myproject/mos_model_nn/script7_encoder_finfet_BNILoss/picture"
FINE_PLOT_PATH = "/home/huangl/myproject/mos_model_nn/script7_encoder_finfet_BNILoss/picture"
MODEL_SAVE_PATH = "/home/huangl/myproject/mos_model_nn/script7_encoder_finfet_BNILoss/saved_models"
PRETRAIN_MODEL_PATH = f"{MODEL_SAVE_PATH}/best_pretrain_model.pth"
FINETUNE_MODEL_PATH = f"{MODEL_SAVE_PATH}/best_finetune_model.pth"
REPORT_SAVE_PATH = f"{MODEL_SAVE_PATH}/experiment_report.txt"  # 实验报告保存路径
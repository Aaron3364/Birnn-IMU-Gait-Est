import torch


class Config:
    def __init__(self):
        # --- 数据与路径参数 ---
        # 把数据文件的路径统一放在这里管理
        self.data_path = 'E:\MyProjects\IMU_Gait_Model\DIP_IMU_and_Others\DIP_IMU'  # 如果文件在其他盘，可以写绝对路径，比如 'E:/data/02.pkl'

        self.sequence_length = 60
        self.input_size = 204  # 17个传感器 * 12维(3维加速度+9维姿态矩阵)
        self.output_size = 72  # 24个关节 * 3维欧拉角

        # --- 模型参数 ---
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout_rate = 0.3

        # --- 训练参数 ---
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 30

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch


class Config:
    def __init__(self):
        # --- 数据与路径参数 ---
        # 把数据文件的路径统一放在这里管理
        self.data_path = 'E:\MyProjects\IMU_Gait_Model\DIP_IMU_and_Others\DIP_IMU'  # 如果文件在其他盘，可以写绝对路径，比如 'E:/data/02.pkl'
        # ================= 🚀 新增：IMU 传感器选择面板 =================
        # 17 个传感器的编号是 0 到 16。
        # 全量测试：使用全部 17 个
        #self.sensor_indices = list(range(17))

        # The order of IMUs is: [head, spine2, belly, lchest, rchest, lshoulder, rshoulder, lelbow, relbow, lhip,
        # rhip, lknee, rknee, lwrist, lwrist, lankle, rankle].
        self.sensor_indices = [0, 1,2,9, 10, 11, 12, 15,16]
        # ================================================================
        self.sequence_length = 60
        # 🌟 动态计算 input_size：选中的传感器数量 * 每个传感器的特征数 (3个加速度 + 9个姿态矩阵元素 = 12)
        self.input_size = len(self.sensor_indices) * 12
        self.output_size = 72  # 24个关节 * 3维欧拉角

        # --- 模型参数 ---
        self.hidden_size = 512  # 从256增加到512
        self.num_layers = 3  # 从2增加到3
        self.dropout_rate = 0.5  # 从0.45增加到0.5

        # --- 训练参数 ---
        self.batch_size = 32  # 从64减少到32
        self.learning_rate = 0.0005  # 从0.001减少到0.0005
        self.epochs = 100  # 从50增加到100

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
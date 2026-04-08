import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R
import random
from sklearn.model_selection import GroupKFold
# 原有的其他 import ...
class DIPIMUDataset(Dataset):
    def __init__(self, root_dir, seq_length=60, transform=True):
        self.seq_length = seq_length

        # 1. 扫描所有 .pkl 文件
        search_pattern = os.path.join(root_dir, '**', '*.pkl')
        pkl_files = glob.glob(search_pattern, recursive=True)
        random.seed(42)  # 设定随机种子保证每次打乱结果一致，方便复现
        random.shuffle(pkl_files)
        if not pkl_files:
            raise ValueError(f"❌ 在 {root_dir} 下没有找到任何 .pkl 文件！请检查路径。")

        print(f"🔍 共扫描到 {len(pkl_files)} 个数据文件，准备提取数据...")

        self.all_X = []
        self.all_Y = []
        self.valid_indices = []  # 存储合法的滑动窗口索引 (文件序号, 起始帧)

        for file_idx, file_path in enumerate(pkl_files):
            with open(file_path, 'rb') as f:
                try:
                    data = pickle.load(f, encoding='latin1')
                except Exception:
                    f.seek(0)
                    data = pickle.load(f)

            keys = data.keys()

            # ================= 动态键名校验 =================
            # 根据最新需求，标签键名变更为 'gt'


            # ================= 提取特征 (X) =================
            imu_acc = data['imu_acc'].astype(np.float32)
            imu_ori = data['imu_ori'].astype(np.float32)
            num_frames = imu_acc.shape[0]

            # 展平姿态矩阵 (N, 17, 3, 3) -> (N, 17, 9)
            imu_ori_flat = imu_ori.reshape(num_frames, -1, 9)
            # 拼接加速度和姿态 (N, 17, 12)
            features = np.concatenate([imu_acc, imu_ori_flat], axis=2)
            X_seq = features.reshape(num_frames, -1)  # -> (N, 204)

            # ================= 提取标签 (Y) =================
            gt_data = data['gt'].astype(np.float32)

            # 智能维度识别：判断 'gt' 是包含了全局位移的 85 维，还是纯净的 72 维
            if gt_data.shape[1] == 85:
                poses_axis_angle = gt_data[:, 3:75]
            elif gt_data.shape[1] == 72:
                poses_axis_angle = gt_data
            else:
                print(f"⚠️ 标签维度异常跳过: {os.path.basename(file_path)} (gt 维度为 {gt_data.shape[1]})")
                continue

            # 轴角转欧拉角
            poses_reshaped = poses_axis_angle.reshape(-1, 3)
            rot = R.from_rotvec(poses_reshaped)
            poses_euler = rot.as_euler('xyz', degrees=True)
            Y_seq = poses_euler.reshape(num_frames, 72).astype(np.float32)

            # ================= 高级时序数据清洗 =================
            # 对 X 进行线性插值填补
            df_X = pd.DataFrame(X_seq)
            df_X = df_X.interpolate(method='linear', limit_direction='both').ffill().bfill()
            X_seq = df_X.values.astype(np.float32)

            # 对 Y 进行线性插值填补
            df_Y = pd.DataFrame(Y_seq)
            df_Y = df_Y.interpolate(method='linear', limit_direction='both').ffill().bfill()
            Y_seq = df_Y.values.astype(np.float32)

            # 终极防线：插值后仍有 NaN 说明该序列完全损坏，直接抛弃
            if np.isnan(X_seq).any() or np.isnan(Y_seq).any():
                print(f"⚠️ 丢弃严重破损数据: {os.path.basename(file_path)}")
                continue
                # ===================================================

            # 存入列表
            self.all_X.append(X_seq)
            self.all_Y.append(Y_seq)

            # 建立滑动窗口的坐标索引映射
            if num_frames >= self.seq_length:
                for i in range(num_frames - self.seq_length + 1):
                    self.valid_indices.append((file_idx, i))

        # ================= 分步全局标准化 =================
        if transform:
            print("📊 正在计算全局数据的标准化参数 (防内存溢出)...")
            self.scaler = StandardScaler()

            for X_seq in self.all_X:
                self.scaler.partial_fit(X_seq)

            print("✨ 正在应用标准化...")
            for i in range(len(self.all_X)):
                self.all_X[i] = self.scaler.transform(self.all_X[i])

        print(f"✅ 数据管道就绪！共生成 {len(self.valid_indices)} 个干净的滑动窗口样本。")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.valid_indices[idx]

        # 按索引切取特征窗口 (60帧)
        window_x = self.all_X[file_idx][frame_idx: frame_idx + self.seq_length]
        # 获取窗口最后一帧对应的目标角度
        target_y = self.all_Y[file_idx][frame_idx + self.seq_length - 1]

        return torch.from_numpy(window_x).float(), torch.from_numpy(target_y).float()


# ================= 封装调用逻辑 =================
'''def get_dataloaders(root_dir, config):
    dataset = DIPIMUDataset(
        root_dir=root_dir,
        seq_length=config.sequence_length
    )

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # 避免时序数据泄露，不使用 random_split，按时间顺序硬切分
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # 训练集打乱窗口顺序，验证集保持原始顺序
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader '''


# ================= 🚀 新增：分组 K 折交叉验证划分逻辑 =================
def get_kfold_dataloaders(dataset, config, k_splits=5):
    """
    使用 GroupKFold 根据文件 ID 进行 K 折划分。
    返回一个包含 K 个元组的列表，每个元组为 (fold_idx, train_loader, val_loader)
    """
    # 提取每个滑动窗口对应的“文件序号 (file_idx)”，作为分组的依据
    groups = [file_idx for file_idx, frame_idx in dataset.valid_indices]

    # 实例化 GroupKFold
    gkf = GroupKFold(n_splits=k_splits)

    # 占位符 X，GroupKFold 只需要知道数据的总长度和分组标签
    dummy_x = np.zeros(len(dataset))

    folds_loaders = []

    print(f"✂️ 正在进行严格的 {k_splits} 折文件级隔离划分...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(dummy_x, groups=groups)):
        # 根据索引创建 Subset
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        # 构建当前折的 DataLoader
        # 注意：训练集内部需要 shuffle=True，验证集保持 False
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False
        )

        folds_loaders.append((fold + 1, train_loader, val_loader))

    return folds_loaders
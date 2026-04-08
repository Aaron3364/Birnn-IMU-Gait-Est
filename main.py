import torch
from configuration import Config
from models import BiRNN_Gait_Estimator
from train import train_model

# 导入新的 Dataset 和 K 折划分函数
from dataset import DIPIMUDataset, get_kfold_dataloaders


def main():
    config = Config()

    # 1. 实例化全局数据集 (数据只读一次，节省时间)
    print(f"正在从 {config.data_path} 加载全局数据...")
    dataset = DIPIMUDataset(config.data_path, seq_length=config.sequence_length)

    # 2. 获取 5 折交叉验证的 Dataloader 列表
    k_splits = 5
    folds_loaders = get_kfold_dataloaders(dataset, config, k_splits=k_splits)

    # ================= 🚀 启动 K 折循环训练 =================
    for fold, train_loader, val_loader in folds_loaders:
        print(f"\n{'=' * 30}")
        print(f"🚀 开始训练第 {fold}/{k_splits} 折 (Fold {fold})")
        print(f"{'=' * 30}")

        # ⚠️ 极其关键：每一折必须重新初始化一个全新的模型！
        # 绝对不能用上一折训练好的模型接着练，否则数据泄露严重！
        model = BiRNN_Gait_Estimator(config)

        # 启动训练
        # 注意：这里我们给 train_model 多传了一个 fold 参数，方便保存文件时区分
        trained_model = train_model(model, train_loader, val_loader, config, fold)


if __name__ == "__main__":
    main()
import torch
import matplotlib.pyplot as plt
import numpy as np
from configuration import Config
from models import BiRNN_Gait_Estimator
from dataset import DIPIMUDataset, get_kfold_dataloaders


def visualize_gait_prediction(model, val_loader, config, num_frames=300, joint_idx=4):
    """
    可视化模型预测的关节欧拉角 vs 真实欧拉角 (时间序列)

    参数:
        model: 训练好的模型
        val_loader: 验证集 DataLoader (必须保证 shuffle=False)
        config: 配置项
        num_frames: 要可视化的连续帧数 (60帧大约是1秒)
        joint_idx: 要可视化的关节索引 (0 到 23)
    """
    model.eval()
    model.to(config.device)

    all_preds = []
    all_gts = []

    print(f"🎨 正在推理前 {num_frames} 帧数据...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(config.device)
            outputs = model(inputs)

            # 将 Tensor 转移到 CPU 并转为 NumPy 数组
            all_preds.append(outputs.cpu().numpy())
            all_gts.append(targets.cpu().numpy())

            # 如果收集到的数据量足够了，就提前退出循环
            if len(all_preds) * config.batch_size >= num_frames:
                break

    # 把多个 batch 拼接成一个长的时间序列
    all_preds = np.concatenate(all_preds, axis=0)[:num_frames]
    all_gts = np.concatenate(all_gts, axis=0)[:num_frames]

    # 提取特定关节的 3 个欧拉角 (Roll, Pitch, Yaw)
    # 因为 72 维数据中，第 i 个关节占据的是 i*3, i*3+1, i*3+2
    start_idx = joint_idx * 3
    end_idx = start_idx + 3

    preds_joint = all_preds[:, start_idx:end_idx]
    gts_joint = all_gts[:, start_idx:end_idx]

    # ================= 开始使用 Matplotlib 绘图 =================
    time_steps = np.arange(num_frames)
    axis_names = ['X-axis (Roll / 翻滚角)', 'Y-axis (Pitch / 俯仰角)', 'Z-axis (Yaw / 偏航角)']

    plt.figure(figsize=(15, 10))
    # 动态标题
    plt.suptitle(f'Joint {joint_idx} Angle Prediction vs Ground Truth', fontsize=18, fontweight='bold')

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        # 真实的 Ground Truth 用黑色的实线表示
        plt.plot(time_steps, gts_joint[:, i], label='Ground Truth (真实动作)', color='black', linewidth=2.5, alpha=0.8)
        # 模型预测的 Prediction 用红色的虚线表示
        plt.plot(time_steps, preds_joint[:, i], label='Prediction (模型预测)', color='red', linestyle='--', linewidth=2)

        plt.ylabel('Angle (Degrees)', fontsize=12)
        plt.title(axis_names[i], fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.7)

    plt.xlabel('Time Steps (Frames / 60Hz)', fontsize=12)
    plt.tight_layout()  # 自动调整间距防止重叠
    # 稍微给总标题留点空间
    plt.subplots_adjust(top=0.92)
    plt.show()


# ================= 独立运行的测试模块 =================
if __name__ == "__main__":
    # 1. 初始化配置
    config = Config()

    # 2. 必须先实例化模型，并加载你训练好的权重 (.pth 文件)
    model = BiRNN_Gait_Estimator(config)

    # ⚠️ 请把下面这个路径替换成你刚才 checkpoints 文件夹里跑出来的 best_model.pth 的真实路径！
    best_model_path = "E:\MyProjects\IMU_Gait_Model\\train\checkpoints\\run_20260407_153004\\best_model.pth"

    try:
        model.load_state_dict(torch.load(best_model_path))
        print("✅ 成功加载最优模型权重！")
    except Exception as e:
        print(f"❌ 加载模型失败，请检查路径是否正确: {e}")
        exit()

    # 3. 加载验证集数据 (为了图省事，这里快速跑一次 K折 划分，提取第一折的验证集)
    dataset = DIPIMUDataset(config.data_path, seq_length=config.sequence_length)
    folds_loaders = get_kfold_dataloaders(dataset, config, k_splits=5)

    # 我们随便取第一折 (fold 1) 的 val_loader 来画图
    _, _, val_loader = folds_loaders[0]

    # 4. 调用画图函数
    # SMPL 骨骼节点参考：4 是左膝盖，5 是右膝盖，7 是左脚踝，8 是右脚踝
    visualize_gait_prediction(model, val_loader, config, num_frames=300, joint_idx=4)
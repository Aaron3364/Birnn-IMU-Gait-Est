import torch
import numpy as np

print("=======================================")
print(f"PyTorch 版本: {torch.__version__}")
print(f"NumPy 版本: {np.__version__}")

# 检查 CUDA (GPU) 是否可用
if torch.cuda.is_available():
    print("✅ 检测到可用 GPU!")
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 数量: {torch.cuda.device_count()}")
else:
    print("❌ 未检测到可用 GPU，模型将使用 CPU 运行。")
print("=======================================")
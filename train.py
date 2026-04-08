import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from datetime import datetime


def train_model(model, train_loader, val_loader, config,fold=1):
    model = model.to(config.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    current_time = datetime.now().strftime('%Y%m%d')  # 这里只用到日期就行了
    # 现在的路径会变成类似: checkpoints/run_20260406/fold_1/
    save_dir = os.path.join('checkpoints', f'run_{current_time}', f'fold_{fold}')

    os.makedirs(save_dir, exist_ok=True)
    print(f" 本折模型将保存在: {save_dir}")
    # ======================================================================

    print(f"开始训练，使用设备: {config.device}")

    for epoch in range(config.epochs):
        # ================= 训练阶段 =================
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪防爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)

        # ================= 验证阶段 =================
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # 更新学习率调度器
        scheduler.step(epoch_val_loss)

        # ================= 💾 新增：按 Epoch 保存模型 =================
        # 保存当前 epoch 的模型 (例如: model_epoch_1.pth)
        epoch_model_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        # ================================================================

        # ================= 保存全局最优模型 =================
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(best_model_weights, best_model_path)
            print(f" 发现更优模型！已覆盖保存至: {best_model_path}")

    print("训练结束！最优模型已加载。")
    model.load_state_dict(best_model_weights)
    return model
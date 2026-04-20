# 使用说明

这个脚本用于从CSV文件绘制loss曲线和其他指标的可视化图表。

## 功能特性

- ✅ 自动识别CSV文件中的loss列
- ✅ 支持多个loss列（如train_loss, val_loss等）
- ✅ 支持绘制MAE、准确率等其他指标
- ✅ 提供简单模式和详细模式两种绘图方式
- ✅ 自动保存图表为高质量PNG文件

## 使用方法

### 基础用法（简单模式）
```bash
python plot_loss.py training_log.csv
```

### 详细模式（包含多个指标）
```bash
python plot_loss.py training_log.csv --mode detailed
```

### 自定义输出目录
```bash
python plot_loss.py training_log.csv --output-dir ./my_plots
```

## CSV文件格式

CSV文件应该包含以下列（至少包含loss相关的列）：

### 示例1：基础格式
```
epoch,train_loss,val_loss
0,0.5234,0.5445
1,0.4892,0.5123
2,0.4567,0.4890
...
```

### 示例2：详细格式
```
epoch,train_loss,val_loss,train_mae,val_mae,learning_rate
0,0.5234,0.5445,0.2341,0.2456,0.001
1,0.4892,0.5123,0.2145,0.2234,0.001
2,0.4567,0.4890,0.1987,0.2012,0.0005
...
```

## 创建CSV日志的方法

### 方法1：从TensorBoard导出
```bash
tensorboard --logdir checkpoints --data_dir data
```

### 方法2：手动从训练日志提取
```python
import csv

# 解析训练日志并写入CSV
with open('training_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss'])
    writer.writerow([0, 0.5234, 0.5445])
    writer.writerow([1, 0.4892, 0.5123])
    # ... 更多数据
```

### 方法3：从Python训练脚本导出
```python
import csv

# 在训练循环中记录数据
logs = []
for epoch in range(num_epochs):
    # ... 训练代码 ...
    logs.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss
    })

# 保存为CSV
import pandas as pd
df = pd.DataFrame(logs)
df.to_csv('training_log.csv', index=False)
```

## 输出结果

- 简单模式：生成 `loss_plot.png`
- 详细模式：生成 `metrics_plot.png`

图表会自动保存在指定的输出目录中，同时会在屏幕上显示。

## 依赖库

- pandas
- matplotlib

安装：
```bash
pip install pandas matplotlib
```

## 常见问题

**Q: 脚本找不到loss列？**
A: 确保CSV文件中的列名包含'loss'字样（大小写不敏感）

**Q: 图表显示不完整？**
A: 可能需要增加图表的宽度或高度，修改脚本中的figsize参数

**Q: 如何修改图表的颜色和样式？**
A: 可以修改脚本中的plot()函数调用，添加color、linestyle等参数

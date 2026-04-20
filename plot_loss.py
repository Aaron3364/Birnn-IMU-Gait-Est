import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss_from_csv(csv_file, output_dir='./plots'):
    """
    从CSV文件读取loss数据并绘制图表
    
    参数：
    csv_file: CSV文件路径
    output_dir: 输出图表的目录
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ 成功读取CSV文件: {csv_file}")
        print(f"📊 数据形状: {df.shape}")
        print(f"📋 列名: {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ 文件不存在: {csv_file}")
        return
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return
    
    # 创建图表
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    
    # 自动识别loss相关的列
    loss_columns = [col for col in df.columns if 'loss' in col.lower() or 'Loss' in col]
    
    if not loss_columns:
        print(f"⚠️ 未找到包含'loss'字样的列")
        print(f"可用的列: {list(df.columns)}")
        return
    
    print(f"📈 检测到的loss列: {loss_columns}")
    
    # 绘制每个loss列
    for col in loss_columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            axes.plot(df.index, df[col], marker='o', linewidth=2, label=col, markersize=4)
    
    axes.set_xlabel('Epoch', fontsize=12)
    axes.set_ylabel('Loss', fontsize=12)
    axes.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes.legend(fontsize=10)
    axes.grid(True, alpha=0.3)
    
    # 保存图表
    output_path = os.path.join(output_dir, 'loss_plot.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()


def plot_multiple_metrics(csv_file, output_dir='./plots'):
    """
    从CSV文件读取多个指标并绘制子图
    
    参数：
    csv_file: CSV文件路径
    output_dir: 输出图表的目录
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ 成功读取CSV文件: {csv_file}")
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return
    
    # 分类不同类型的指标
    loss_cols = [col for col in df.columns if 'loss' in col.lower()]
    mae_cols = [col for col in df.columns if 'mae' in col.lower()]
    other_cols = [col for col in df.columns if col not in loss_cols + mae_cols]
    
    # 创建子图
    num_plots = (len(loss_cols) > 0) + (len(mae_cols) > 0) + (len(other_cols) > 0)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # 绘制Loss
    if loss_cols:
        for col in loss_cols:
            axes[plot_idx].plot(df.index, df[col], marker='o', linewidth=2, label=col, markersize=4)
        axes[plot_idx].set_ylabel('Loss', fontsize=11)
        axes[plot_idx].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[plot_idx].legend(fontsize=9)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # 绘制MAE
    if mae_cols:
        for col in mae_cols:
            axes[plot_idx].plot(df.index, df[col], marker='s', linewidth=2, label=col, markersize=4)
        axes[plot_idx].set_ylabel('MAE', fontsize=11)
        axes[plot_idx].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[plot_idx].legend(fontsize=9)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # 绘制其他指标
    if other_cols:
        for col in other_cols:
            if col != 'epoch' and col.lower() != 'epoch':
                axes[plot_idx].plot(df.index, df[col], marker='^', linewidth=2, label=col, markersize=4)
        axes[plot_idx].set_ylabel('Value', fontsize=11)
        axes[plot_idx].set_title('Other Metrics', fontsize=12, fontweight='bold')
        axes[plot_idx].legend(fontsize=9)
        axes[plot_idx].grid(True, alpha=0.3)
    
    # 设置底部坐标标签
    axes[-1].set_xlabel('Epoch', fontsize=12)
    
    # 保存图表
    output_path = os.path.join(output_dir, 'metrics_plot.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从CSV文件绘制loss图表')
    parser.add_argument('csv_file', type=str, help='CSV文件路径')
    parser.add_argument('--output-dir', type=str, default='./plots', help='输出图表目录 (默认: ./plots)')
    parser.add_argument('--mode', type=str, choices=['simple', 'detailed'], default='simple',
                        help='绘图模式: simple (简单模式，只绘制loss) 或 detailed (详细模式，绘制多个指标)')
    
    args = parser.parse_args()
    
    if args.mode == 'simple':
        plot_loss_from_csv(args.csv_file, args.output_dir)
    else:
        plot_multiple_metrics(args.csv_file, args.output_dir)

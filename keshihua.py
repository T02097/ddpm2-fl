import matplotlib.pyplot as plt
import numpy as np


def visualize_loss_curve(loss_file_path="./Checkpoints/FashionMNIST/loss.txt"):
    """
    可视化训练损失曲线
    """
    # 读取损失数据
    epochs = []
    losses = []

    with open(loss_file_path, 'r') as f:
        lines = f.readlines()
        # 跳过标题行
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                epoch, loss = parts
                epochs.append(int(epoch))
                losses.append(float(loss))

    # 创建可视化图表
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', linewidth=1.5, markersize=4)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图表
    plt.savefig('./Checkpoints/FashionMNIST/loss_curve.png', dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()

    print(f"可视化完成！共绘制 {len(epochs)} 个epoch的数据")
    print(f"最小损失值: {min(losses):.6f} (epoch {epochs[losses.index(min(losses))]})")
    print(f"最终损失值: {losses[-1]:.6f}")


# 运行可视化
if __name__ == "__main__":
    visualize_loss_curve()

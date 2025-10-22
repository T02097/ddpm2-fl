# 联邦学习DDPM框架使用说明

## 概述

本框架将联邦学习与DDPM（Denoising Diffusion Probabilistic Models）模型结合，支持在多个客户端上分布式训练扩散模型，同时保护数据隐私。

## 主要特性

- ✅ 完全适配现有的DDPM模型结构
- ✅ 支持IID和Non-IID数据分布
- ✅ 使用FedAvg聚合策略
- ✅ 支持断点续训
- ✅ 自动模型保存和检查点管理
- ✅ 完整的日志记录

## 文件结构

```
federated/
├── server.py          # 联邦学习服务器
├── client.py          # 联邦学习客户端
├── train_federated.py # 联邦学习训练主程序
└── sample_flddpm.py   # 联邦学习模型推理
```

## 快速开始

### 1. 基本联邦学习训练

```python
from federated.train_federated import main

# 使用默认配置
main()
```

### 2. 自定义配置训练

```python
from federated.train_federated import main

# 自定义联邦学习配置
custom_config = {
    "num_clients": 5,           # 客户端数量
    "num_rounds": 50,           # 联邦学习轮数
    "epochs_per_round": 3,      # 每轮本地训练epoch数
    "batch_size": 80,           # 批次大小
    "learning_rate": 1e-4,      # 学习率
    "data_distribution": "non_iid",  # 数据分布类型
    "save_interval": 10,        # 模型保存间隔
    "model_save_path": "./Checkpoints/Federated/",
    "log_path": "./logs/federated/",
}

main(custom_config)
```

### 3. 模型推理

```python
from federated.sample_flddpm import infer

# 使用最新的联邦学习模型生成样本
infer(
    model_path=None,  # 自动查找最新检查点
    save_dir="./SampledImgs/federated_samples",
    num_samples=100
)
```

## 配置参数说明

### 联邦学习参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_clients` | int | 5 | 客户端数量 |
| `num_rounds` | int | 50 | 联邦学习轮数 |
| `epochs_per_round` | int | 3 | 每轮本地训练epoch数 |
| `batch_size` | int | 80 | 批次大小 |
| `learning_rate` | float | 1e-4 | 学习率 |
| `client_selection_ratio` | float | 1.0 | 客户端选择比例 |
| `data_distribution` | str | "non_iid" | 数据分布类型 ("iid" 或 "non_iid") |
| `save_interval` | int | 10 | 模型保存间隔 |
| `model_save_path` | str | "./Checkpoints/Federated/" | 模型保存路径 |
| `log_path` | str | "./logs/federated/" | 日志保存路径 |

### 模型参数

模型参数与 `Main.py` 中的配置保持一致：

```python
model_params = {
    "T": 1000,                    # 扩散步数
    "channel": 128,               # 基础通道数
    "channel_mult": [1, 2, 2, 4], # 通道倍数
    "attn": [1],                  # 注意力层位置
    "num_res_blocks": 2,          # 残差块数量
    "dropout": 0.15,              # Dropout率
    "device": "cuda:0",           # 设备
    "grad_clip": 1.0,             # 梯度裁剪
}
```

## 数据分布策略

### IID分布
支持两种IID模式：

#### 1. 标准IID模式（默认）
- 随机打乱FashionMNIST数据集
- 数据在所有客户端间均匀分配
- 每个客户端包含所有类别的数据
- 使用参数：`"use_custom_iid_folders": False`

#### 2. 自定义IID模式
- 使用预处理的文件夹结构
- 文件夹数量与客户端数量对应
- 每个客户端文件夹包含所有类别的数据
- 使用参数：`"use_custom_iid_folders": True, "custom_iid_path": "./custom_iid_data"`

### Non-IID分布
- **2个客户端**: 前5个类别给客户端0，后5个类别给客户端1
- **5个客户端**: 每2个类别分配给一个客户端
- **10个客户端**: 每个客户端只包含一个类别的数据

## 运行示例

### 命令行运行

```bash
# 基本训练（标准IID模式）
python federated/train_federated.py

# 模型推理
python federated/sample_flddpm.py --num_samples 200 --save_dir ./my_samples
```

### IID模式使用

#### 标准IID模式
```python
from federated.train_federated import main

config = {
    "num_clients": 2,
    "data_distribution": "iid",
    "use_custom_iid_folders": False,  # 使用标准IID
}
main(config)
```

#### 自定义IID模式
```bash
# 1. 准备自定义IID数据
python prepare_custom_iid_data.py --num_clients 2 --output_path ./custom_iid_data

# 2. 使用自定义IID数据训练
python example_iid_modes.py  # 选择选项2
```

### 测试集成

```bash
# 运行集成测试
python test_federated_integration.py

# 测试IID分布
python test_iid_distribution.py

# 演示IID模式
python example_iid_modes.py
```

## 输出文件

### 模型检查点
- 位置: `./Checkpoints/Federated/`
- 格式: `global_model_round_{round_idx}.pth`
- 包含: 模型状态、轮数信息、配置信息

### 日志文件
- 位置: `./logs/federated/`
- 文件: `server.log`
- 内容: 训练过程、客户端选择、聚合信息

### 生成样本
- 位置: `./SampledImgs/federated_samples/`
- 格式: `federated_sample_{idx}.png`
- 内容: 联邦学习模型生成的FashionMNIST样本

## 注意事项

1. **GPU内存**: 确保有足够的GPU内存支持多客户端训练
2. **数据路径**: 确保FashionMNIST数据集可以正常下载和访问
3. **检查点**: 定期保存检查点，支持断点续训
4. **日志监控**: 关注日志文件中的训练进度和错误信息

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少 `batch_size`
   - 减少 `num_clients`
   - 使用CPU训练（修改 `device` 参数）

2. **数据加载失败**
   - 检查网络连接（FashionMNIST自动下载）
   - 确认数据路径权限

3. **模型保存失败**
   - 检查保存路径权限
   - 确保磁盘空间充足

## 扩展功能

### 添加新的聚合策略
在 `server.py` 中修改 `aggregate_models` 方法

### 支持其他数据集
在 `client.py` 中修改 `load_data` 方法

### 自定义客户端选择策略
在 `server.py` 中修改 `select_clients` 方法

## 性能优化建议

1. **并行训练**: 使用多GPU并行训练不同客户端
2. **数据预处理**: 预先处理数据以减少训练时间
3. **模型压缩**: 使用模型压缩技术减少通信开销
4. **异步更新**: 实现异步联邦学习提高效率

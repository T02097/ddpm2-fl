import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple
from Diffusion.Model import UNet
from Diffusion.Diffusion import GaussianDiffusionTrainer
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from collections import Counter
from federated.MyDataset import MyDataset,create_dataloader


class FederatedClient:
    def __init__(self, client_id, model_params, diffusion_params, federated_config):
        self.client_id = client_id
        self.config = federated_config
        self.model_config = model_params
        self.diffusion_params = diffusion_params
        
        # 使用新的UNet模型（从Diffusion/Model.py）
        self.model = UNet(
            T=model_params['T'],
            ch=model_params['channel'],
            ch_mult=model_params['channel_mult'],
            attn=model_params['attn'],
            num_res_blocks=model_params['num_res_blocks'],
            dropout=model_params['dropout']
        )
        
        # 初始化时就移到GPU
        device = torch.device(model_params.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.device = device
        self.model = self.model.to(device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
        
        # 使用GaussianDiffusionTrainer
        self.trainer = GaussianDiffusionTrainer(
            self.model,
            diffusion_params['beta_1'],
            diffusion_params['beta_T'],
            diffusion_params['T'],
            # warmup_steps=40  # 添加warmup_steps参数
        ).to(device)
        
        self.split = self.config.get('data_split', 'train')

    def load_data_with_params(self, dataset_params: Dict):
        """使用MyDataset加载数据 - 新的数据加载方法"""
        print(f"客户端 {self.client_id}: 使用MyDataset加载数据...")
        
        # 从参数中获取配置
        im_path = dataset_params['im_path']
        img_size = dataset_params['img_size']
        in_channels = dataset_params['in_channels']
        out_channels = dataset_params['out_channels']
        im_ext = dataset_params.get('im_ext', 'png')
        normalize_range = dataset_params.get('normalize_range', (-1, 1))
        use_augmentation = dataset_params.get('use_augmentation', True)
        
        if self.config['data_distribution'] == 'iid':
            # IID模式：使用MyDataset加载完整数据集，然后分配给客户端
            if self.config.get('use_custom_iid_folders', False):
                # 自定义IID文件夹模式
                custom_iid_path = self.config.get('custom_iid_path', './custom_iid_data')
                client_folder = os.path.join(custom_iid_path, f'client_{self.client_id}')
                
                if not os.path.exists(client_folder):
                    raise FileNotFoundError(f"客户端 {self.client_id} 的数据文件夹不存在: {client_folder}")
                
                # 使用MyDataset加载指定文件夹的数据
                self.dataset = MyDataset(
                    split='train',
                    im_path=client_folder,
                    im_ext=im_ext,
                    img_size=img_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    normalize_range=normalize_range,
                    use_augmentation=use_augmentation
                )
                
                print(f"客户端 {self.client_id}: 从自定义文件夹 {client_folder} 加载了 {len(self.dataset)} 个样本")
                
            else:
                # 标准IID模式：加载完整数据集然后随机分配
                full_dataset = MyDataset(
                    split='train',
                    im_path=im_path,
                    im_ext=im_ext,
                    img_size=img_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    normalize_range=normalize_range,
                    use_augmentation=use_augmentation
                )
                
                # 随机分配数据给客户端
                import random
                all_indices = list(range(len(full_dataset)))
                random.seed(42 + self.client_id)  # 每个客户端使用不同的种子
                random.shuffle(all_indices)
                
                # 计算每个客户端的数据量
                total_samples = len(full_dataset)
                samples_per_client = total_samples // self.config['num_clients']
                start_idx = self.client_id * samples_per_client
                end_idx = start_idx + samples_per_client if self.client_id < self.config['num_clients'] - 1 else total_samples
                
                # 获取分配给当前客户端的索引
                client_indices = all_indices[start_idx:end_idx]
                self.dataset = Subset(full_dataset, client_indices)
                
                print(f"客户端 {self.client_id}: 从完整数据集分配了 {len(client_indices)} 个样本")
                
        else:
            # Non-IID模式：按类别分配
            full_dataset = MyDataset(
                split='train',
                im_path=im_path,
                im_ext=im_ext,
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                normalize_range=normalize_range,
                use_augmentation=use_augmentation
            )
            
            # 获取类别信息
            if hasattr(full_dataset, 'get_class_distribution'):
                class_distribution = full_dataset.get_class_distribution()
                num_categories = len(class_distribution)
            else:
                # 假设有10个类别（如FashionMNIST）
                num_categories = 10
            
            # 根据客户端数量确定类别分配
            if self.config['num_clients'] == 2:
                assigned_categories = list(range(num_categories // 2)) if self.client_id == 0 else list(range(num_categories // 2, num_categories))
            elif self.config['num_clients'] == 5:
                start_idx = self.client_id * 2
                end_idx = start_idx + 2 if self.client_id < self.config['num_clients'] - 1 else num_categories
                assigned_categories = list(range(start_idx, end_idx))
            elif self.config['num_clients'] == 10:
                assigned_categories = [self.client_id]
            else:
                raise ValueError(f"不支持的客户端数量: {self.config['num_clients']}")
            
            # 获取对应类别的数据索引
            indices = []
            for idx in range(len(full_dataset)):
                _, label = full_dataset[idx]
                if label in assigned_categories:
                    indices.append(idx)
            
            self.dataset = Subset(full_dataset, indices)
            print(f"客户端 {self.client_id}: Non-IID模式，分配类别 {assigned_categories}，共 {len(indices)} 个样本")
        
        # 创建数据加载器
        self.data_loader = create_dataloader(
            self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        
        print(f"客户端 {self.client_id}: 数据加载完成，共 {len(self.dataset)} 个样本")

    def load_data(self, data_path: str):
        """加载客户端数据 - 支持普通IID和指定IID两种模式"""
        
        # 数据变换
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 调整到32x32以匹配模型
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # FashionMNIST是单通道
        ])
        
        if self.config['data_distribution'] == 'iid':
            # 检查是否使用指定的IID文件夹
            if self.config.get('use_custom_iid_folders', False):
                # 指定IID模式：使用预处理的文件夹
                custom_iid_path = self.config.get('custom_iid_path', './custom_iid_data')
                client_folder = os.path.join(custom_iid_path, f'client_{self.client_id}')
                
                if not os.path.exists(client_folder):
                    raise FileNotFoundError(f"客户端 {self.client_id} 的数据文件夹不存在: {client_folder}")
                
                # 使用ImageFolder加载指定文件夹的数据
                from torchvision.datasets import ImageFolder
                self.dataset = ImageFolder(root=client_folder, transform=transform)
                
                # 统计类别分布
                category_counts = Counter()
                for _, label in self.dataset:
                    category_counts[label] += 1
                
                logging.info(f"Client {self.client_id}: Custom IID from folder {client_folder} with {len(self.dataset)} samples")
                logging.info(f"Client {self.client_id} category distribution: {dict(category_counts)}")
                print(f"Client {self.client_id}: Custom IID from folder {client_folder} with {len(self.dataset)} samples")
                print(f"Client {self.client_id} category distribution: {dict(category_counts)}")
                
            else:
                # 普通IID模式：随机打乱FashionMNIST数据
                # 加载完整的FashionMNIST数据集
                full_dataset = FashionMNIST(
                    root=data_path,
                    train=True,
                    download=True,
                    transform=transform
                )
                
                # 随机打乱所有数据索引
                import random
                all_indices = list(range(len(full_dataset)))
                random.seed(42)  # 设置固定种子确保可重现性
                random.shuffle(all_indices)
                
                # 为每个客户端分配数据
                total_samples = len(full_dataset)
                samples_per_client = total_samples // self.config['num_clients']
                start_idx = self.client_id * samples_per_client
                end_idx = start_idx + samples_per_client if self.client_id < self.config['num_clients'] - 1 else total_samples
                
                # 获取分配给当前客户端的随机索引
                client_indices = all_indices[start_idx:end_idx]
                self.dataset = Subset(full_dataset, client_indices)
                
                # 统计类别分布以验证IID
                category_counts = Counter()
                for idx in client_indices:
                    _, label = full_dataset[idx]
                    category_counts[label] += 1
                
                logging.info(f"Client {self.client_id}: Standard IID distribution with {len(client_indices)} samples")
                logging.info(f"Client {self.client_id} category distribution: {dict(category_counts)}")
                print(f"Client {self.client_id}: Standard IID distribution with {len(client_indices)} samples")
                print(f"Client {self.client_id} category distribution: {dict(category_counts)}")
        else:
            # Non-IID情况：按类别划分
            num_categories = 10  # FashionMNIST有10个类别
            
            # 根据客户端数量确定类别分配
            if self.config['num_clients'] == 2:
                # 前一半类别给客户端0，后一半给客户端1
                assigned_categories = list(range(num_categories // 2)) if self.client_id == 0 else list(range(num_categories // 2, num_categories))
            elif self.config['num_clients'] == 5:
                # 每两个类别给一个客户端
                start_idx = self.client_id * 2
                end_idx = start_idx + 2 if self.client_id < self.config['num_clients'] - 1 else num_categories
                assigned_categories = list(range(start_idx, end_idx))
            elif self.config['num_clients'] == 10:
                # 每个客户端一个类别
                assigned_categories = [self.client_id]
            else:
                raise ValueError(f"Unsupported number of clients: {self.config['num_clients']}")
            
            # 获取对应类别的数据索引
            indices = []
            for idx in range(len(full_dataset)):
                _, label = full_dataset[idx]
                if label in assigned_categories:
                    indices.append(idx)
            
            # 创建客户端数据集
            self.dataset = Subset(full_dataset, indices)
            
            # 打印客户端数据类别分布
            logging.info(f"Client {self.client_id} assigned categories: {assigned_categories}")
            print(f"Client {self.client_id} assigned categories: {assigned_categories}")
            
            # 统计类别分布
            category_counts = Counter()
            for idx in indices:
                _, label = full_dataset[idx]
                category_counts[label] += 1
            logging.info(f"Client {self.client_id} category distribution: {dict(category_counts)}")
            print(f"Client {self.client_id} category distribution: {dict(category_counts)}")

        # 创建数据加载器
        # IID情况下每个epoch都重新打乱，Non-IID情况下也打乱以增加随机性
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,  # 每个epoch都重新打乱
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        logging.info(f"Client {self.client_id} loaded {len(self.dataset)} samples")
        print(f"Client {self.client_id} loaded {len(self.dataset)} samples")

    def train(self, global_model_state: Dict, round_idx: int) -> Tuple[Dict, int]:
        """本地模型训练 - 使用GaussianDiffusionTrainer"""
        # 加载全局模型参数
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
        # 记录训练损失
        total_loss = 0
        num_batches = 0
        
        # 本地训练
        for epoch in range(self.config['epochs_per_round']):
            epoch_loss = 0
            for batch_idx, (images, labels) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                
                # 将数据移到设备
                x_0 = images.to(self.device)
                
                # 使用客户端自己的trainer计算损失
                # trainer会自动生成噪声、时间步并添加噪声
                loss = self.trainer(x_0).sum() / 1000.
                
                # 反向传播和梯度裁剪
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.model_config.get('grad_clip', 1.0)
                )
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 每10个batch打印一次loss
                if batch_idx % 10 == 0:
                    print(f"Round {round_idx}, Client {self.client_id}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(self.data_loader)
            print(f"Round {round_idx}, Client {self.client_id}, Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
            logging.info(f"Client {self.client_id}, Round {round_idx}, Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
        
        # 返回更新后的模型参数和数据集大小
        return self.model.state_dict(), len(self.dataset)

import glob
import os
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from collections import Counter


class MyDataset(Dataset):
    r"""
    自定义数据集类，支持任意尺寸和通道的图像数据
    可以处理任何预处理后的数据集，自动适应不同的图像尺寸和通道数
    """

    def __init__(self, split, im_path, im_ext='png', img_size=64, in_channels=3, out_channels=3, 
                 normalize_range=(-1, 1), use_augmentation=True):
        r"""
        初始化数据集
        :param split: train/test 数据集分割标识
        :param im_path: 图像根目录路径
        :param im_ext: 图像文件扩展名，默认为'png'
        :param img_size: 目标图像尺寸，默认为64
        :param in_channels: 输入通道数，默认为3（RGB）
        :param out_channels: 输出通道数，默认为3（RGB）
        :param normalize_range: 归一化范围，默认为(-1, 1)
        :param use_augmentation: 是否使用数据增强，默认为True
        """
        self.split = split
        self.im_ext = im_ext
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize_range = normalize_range
        self.use_augmentation = use_augmentation
        
        # 加载图像路径和标签
        self.images, self.labels = self.load_images(im_path)
        
        # 构建预处理流程
        self.transform = self._build_transform()
        
        print(f'MyDataset初始化完成: {len(self.images)} 张图像, 尺寸: {img_size}x{img_size}, 通道: {in_channels}')

    def _build_transform(self):
        """构建图像预处理流程"""
        tfms = []
        
        # 1. 调整图像尺寸
        tfms.append(transforms.Resize((self.img_size, self.img_size)))
        
        # 2. 数据增强（仅在训练时使用）
        if self.use_augmentation and self.split == 'train':
            tfms.append(transforms.RandomHorizontalFlip(p=0.5))
            tfms.append(transforms.RandomRotation(degrees=10))
            tfms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        
        # 3. 通道处理
        if self.in_channels == 1:
            # 单通道（灰度图）
            tfms.append(transforms.Grayscale(num_output_channels=1))
        elif self.in_channels == 3:
            # 三通道（RGB）
            # 如果原图是灰度图，转换为RGB
            tfms.append(transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img))
        else:
            raise ValueError(f"不支持的输入通道数: {self.in_channels}")
        
        # 4. 转换为张量
        tfms.append(transforms.ToTensor())
        
        # 5. 归一化
        if self.normalize_range == (-1, 1):
            # 归一化到[-1, 1]范围
            if self.in_channels == 3:
                tfms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            else:
                tfms.append(transforms.Normalize((0.5,), (0.5,)))
        elif self.normalize_range == (0, 1):
            # 归一化到[0, 1]范围（ToTensor已经做了这个）
            pass
        else:
            # 自定义归一化范围
            min_val, max_val = self.normalize_range
            tfms.append(transforms.Lambda(lambda x: x * (max_val - min_val) + min_val))
        
        # 6. 输出通道调整（如果需要）
        if self.out_channels != self.in_channels:
            if self.out_channels == 1 and self.in_channels == 3:
                # RGB转灰度
                tfms.append(transforms.Grayscale(num_output_channels=1))
            elif self.out_channels == 3 and self.in_channels == 1:
                # 灰度转RGB（复制通道）
                tfms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
        return transforms.Compose(tfms)

    def load_images(self, im_path):
        r"""
        从指定路径加载所有图像
        :param im_path: 图像根目录路径
        :return: 图像路径列表和标签列表
        """
        assert os.path.exists(im_path), f"图像路径 {im_path} 不存在"
        
        ims = []
        labels = []
        
        # 检查是否为分类数据集（包含子目录）
        if any(os.path.isdir(os.path.join(im_path, d)) for d in os.listdir(im_path)):
            # 分类数据集：每个子目录代表一个类别
            print(f"检测到分类数据集结构，正在加载...")
            for d_name in tqdm(os.listdir(im_path), desc="加载类别"):
                class_path = os.path.join(im_path, d_name)
                if os.path.isdir(class_path):
                    for fname in glob.glob(os.path.join(class_path, f'*.{self.im_ext}')):
                        ims.append(fname)
                        labels.append(int(d_name) if d_name.isdigit() else d_name)
        else:
            # 非分类数据集：所有图像在同一目录
            print(f"检测到非分类数据集结构，正在加载...")
            for fname in tqdm(glob.glob(os.path.join(im_path, f'*.{self.im_ext}')), desc="加载图像"):
                ims.append(fname)
                labels.append(0)  # 所有图像使用相同标签
        
        print(f'找到 {len(ims)} 张图像用于 {self.split} 数据集')
        
        # 统计类别分布
        if labels:
            label_counts = Counter(labels)
            print(f'类别分布: {dict(label_counts)}')
        
        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """获取指定索引的图像"""
        try:
            # 加载图像
            im = Image.open(self.images[index])
            
            # 应用预处理
            im_tensor = self.transform(im)
            
            # 返回图像张量和标签
            return im_tensor, self.labels[index]
            
        except Exception as e:
            print(f"加载图像失败 {self.images[index]}: {e}")
            # 返回一个空白图像作为fallback
            if self.in_channels == 3:
                blank_tensor = torch.zeros(self.in_channels, self.img_size, self.img_size)
            else:
                blank_tensor = torch.zeros(1, self.img_size, self.img_size)
            return blank_tensor, 0

    def get_class_distribution(self):
        """获取类别分布统计"""
        return Counter(self.labels)

    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'total_samples': len(self.images),
            'image_size': self.img_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'class_distribution': dict(self.get_class_distribution()),
            'split': self.split
        }


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True):
    """
    创建数据加载器
    :param dataset: 数据集对象
    :param batch_size: 批次大小
    :param shuffle: 是否打乱数据
    :param num_workers: 工作进程数
    :param drop_last: 是否丢弃最后一个不完整的批次
    :return: DataLoader对象
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )


# 便捷函数：创建训练和测试数据集
def create_train_test_datasets(data_path, img_size=64, in_channels=3, out_channels=3, 
                              train_ratio=0.8, batch_size=32, **kwargs):
    """
    创建训练和测试数据集
    :param data_path: 数据路径
    :param img_size: 图像尺寸
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param train_ratio: 训练集比例
    :param batch_size: 批次大小
    :param kwargs: 其他参数
    :return: (train_dataset, test_dataset, train_loader, test_loader)
    """
    # 创建完整数据集
    full_dataset = MyDataset(
        split='full',
        im_path=data_path,
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        **kwargs
    )
    
    # 分割训练和测试集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, test_dataset, train_loader, test_loader

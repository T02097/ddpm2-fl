import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import inception_v3
import pickle
from scipy import linalg
from PIL import Image
import argparse
from torchvision.datasets import CIFAR10


class FIDCalculator:
    """
    FID计算器，用于计算生成图像与真实图像之间的FID分数
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.inception_model = self._load_inception_model()

    def _load_inception_model(self):
        """
        加载预训练的Inception-v3模型用于特征提取
        """
        # 使用Inception-v3模型，设置为评估模式
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()  # 移除最后的分类层
        inception.eval()
        inception.to(self.device)
        return inception

    def _preprocess_image(self, image):
        """
        预处理图像以适应Inception-v3模型的输入要求
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
            else:
                raise ValueError("Unsupported image array shape")

        if isinstance(image, Image.Image):
            # 调整大小为299x299
            image = image.resize((299, 299), Image.BILINEAR)
            # 转换为张量
            image = transforms.ToTensor()(image)

        # 标准化到[0, 1]范围（如果需要）
        if image.min() < 0:
            # 已经在[-1, 1]范围内
            image = image * 0.5 + 0.5  # 转换到[0, 1]

        # 标准化到Inception模型期望的范围
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        return image.unsqueeze(0)  # 添加批次维度

    def _get_activations(self, images):
        """
        使用Inception模型获取图像的特征激活
        """
        activations = []
        with torch.no_grad():
            for image in images:
                preprocessed = self._preprocess_image(image)
                preprocessed = preprocessed.to(self.device)
                activation = self.inception_model(preprocessed)
                activations.append(activation.cpu().numpy())
        return np.concatenate(activations, axis=0)

    def _calculate_statistics(self, activations):
        """
        计算特征激活的均值和协方差
        """
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def save_dataset_features(self, dataset_name, feature_file=None, root='./FashionMNIST', train=True):
        """
        从常见数据集生成真实图像特征文件

        Args:
            dataset_name: 数据集名称 ('cifar10', 'mnist', 'fashion_mnist' 等)
            feature_file: 保存特征的文件路径
            root: 数据集存储根目录
            train: 是否使用训练集
        """
        # 设置默认特征文件路径
        if feature_file is None:
            feature_file = f'./Features/{dataset_name}_real_features.pkl'

        # 加载对应的数据集
        print(f"Loading {dataset_name} dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 根据数据集名称选择合适的数据集类
        if dataset_name.lower() == 'cifar10':
            from torchvision.datasets import CIFAR10
            dataset = CIFAR10(
                root=root,
                train=train,
                download=True,
                transform=transform
            )
            num_channels = 3
        elif dataset_name.lower() == 'mnist':
            from torchvision.datasets import MNIST
            dataset = MNIST(
                root=root,
                train=train,
                download=True,
                transform=transform
            )
            num_channels = 1
        elif dataset_name.lower() == 'fashion_mnist':
            from torchvision.datasets import FashionMNIST
            dataset = FashionMNIST(
                root=root,
                train=train,
                download=True,
                transform=transform
            )
            num_channels = 1
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: cifar10, mnist, fashion_mnist")

        # 选择5000张图像用于计算FID
        print("Selecting 5000 images from dataset...")
        dataloader = DataLoader(dataset, batch_size=5000, shuffle=True)
        images, _ = next(iter(dataloader))

        # 转换为PIL图像列表
        pil_images = []
        for i in range(images.shape[0]):
            if num_channels == 3:
                # RGB图像 (如CIFAR10)
                img_array = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                # 灰度图像 (如MNIST, FashionMNIST)
                img_array = (images[i].squeeze().numpy() * 255).astype(np.uint8)
                # 转换为RGB
                img_array = np.stack([img_array] * 3, axis=-1)

            pil_image = Image.fromarray(img_array)
            pil_images.append(pil_image)

        # 提取特征
        print("Extracting features from real images...")
        activations = self._get_activations(pil_images)

        # 计算统计信息
        mu, sigma = self._calculate_statistics(activations)

        # 保存到文件
        features = {
            'mu': mu,
            'sigma': sigma,
            'num_images': len(pil_images)
        }

        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(feature_file), exist_ok=True)

        with open(feature_file, 'wb') as f:
            pickle.dump(features, f)

        print(f"Real image features saved to {feature_file}")
        return features

    def save_features(self, image_dir, feature_file):
        """
        从图像目录中提取特征并保存到文件

        Args:
            image_dir: 包含图像的目录路径
            feature_file: 保存特征的文件路径
        """
        # 收集所有图像文件
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

        # 加载图像
        images = []
        for file in image_files[:5000]:  # 限制为最多5000张图像
            try:
                image = Image.open(file).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Error loading image {file}: {e}")

        # 提取特征
        print(f"Extracting features from {len(images)} images...")
        activations = self._get_activations(images)

        # 计算统计信息
        mu, sigma = self._calculate_statistics(activations)

        # 保存到文件
        features = {
            'mu': mu,
            'sigma': sigma,
            'num_images': len(images)
        }

        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(feature_file), exist_ok=True)

        with open(feature_file, 'wb') as f:
            pickle.dump(features, f)

        print(f"Features saved to {feature_file}")
        return features

    def calculate_fid(self, real_features_file, generated_features_file):
        """
        根据两个特征文件计算FID分数

        Args:
            real_features_file: 真实图像特征文件路径
            generated_features_file: 生成图像特征文件路径

        Returns:
            FID分数
        """
        # 加载特征文件
        with open(real_features_file, 'rb') as f:
            real_features = pickle.load(f)

        with open(generated_features_file, 'rb') as f:
            generated_features = pickle.load(f)

        # 计算FID分数
        mu1, sigma1 = real_features['mu'], real_features['sigma']
        mu2, sigma2 = generated_features['mu'], generated_features['sigma']

        fid_score = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_score

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        计算两个多元高斯分布之间的Frechet距离
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # 计算协方差矩阵的乘积
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ("fid calculation produces singular product; "
                   "adding %s to diagonal of cov estimates") % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # 数值误差可能给我们一个略微负的值
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) +
                np.trace(sigma1) +
                np.trace(sigma2) -
                2 * tr_covmean)

    def calculate_fid_from_images(self, real_images_dir, generated_images_dir):
        """
        直接从图像目录计算FID分数

        Args:
            real_images_dir: 真实图像目录
            generated_images_dir: 生成图像目录

        Returns:
            FID分数
        """
        # 创建临时特征文件路径
        real_features_file = './Features/temp_real_features.pkl'
        generated_features_file = './Features/temp_generated_features.pkl'

        # 提取并保存特征
        self.save_features(real_images_dir, real_features_file)
        self.save_features(generated_images_dir, generated_features_file)

        # 计算FID分数
        fid_score = self.calculate_fid(real_features_file, generated_features_file)

        # 清理临时文件
        os.remove(real_features_file)
        os.remove(generated_features_file)

        return fid_score


def main():
    parser = argparse.ArgumentParser(description='Calculate FID score for generated images')
    parser.add_argument('--mode', type=str, choices=['generate_real', 'calculate_fid'],
                        default='calculate_fid',
                        help='Mode: generate_real for generating real image features, calculate_fid for calculating FID score')
    parser.add_argument('--dataset', type=str, default='fashion-mnist',
                        help='Dataset name for generating real features (cifar10, mnist, fashion_mnist)')
    parser.add_argument('--real_features', type=str,
                        default=None,
                        help='Path to real image features file')
    parser.add_argument('--generated_dir', type=str,
                        default='./SampledImgs/federated_samples_mse_e200_2',
                        help='Directory containing generated images')
    parser.add_argument('--output_file', type=str,
                        default='./fid_e200_2.txt',
                        help='File to save the FID score')

    args = parser.parse_args()

    # 设置默认的real_features路径
    if args.real_features is None:
        args.real_features = f'./Features/{args.dataset}_real_features.pkl'

    # 创建FID计算器
    fid_calculator = FIDCalculator()

    if args.mode == 'generate_real':
        # 生成真实图像特征文件
        fid_calculator.save_dataset_features(args.dataset, args.real_features)
        print(f"Real image features saved to {args.real_features}")
    else:
        # 计算FID分数
        # 检查真实图像特征文件是否存在
        if not os.path.exists(args.real_features):
            print(f"Real features file {args.real_features} not found.")
            print("Generating real image features first...")
            fid_calculator.save_dataset_features(args.dataset, args.real_features)

        # 生成图像特征文件路径
        generated_features_file = './Features/fmnist_generated_features.pkl'

        # 提取生成图像的特征
        print("Extracting features from generated images...")
        fid_calculator.save_features(args.generated_dir, generated_features_file)

        # 计算FID分数
        print("Calculating FID score...")
        fid_score = fid_calculator.calculate_fid(args.real_features, generated_features_file)

        # 保存和打印FID分数
        with open(args.output_file, 'w') as f:
            f.write(str(fid_score))

        print(f"FID Score: {fid_score}")
        print(f"FID Score saved to {args.output_file}")



if __name__ == "__main__":
    '''
# 生成真实图像特征
python fid_calculator.py --mode generate_real --dataset fashion_mnist
# 计算FID分数（需要先生成真实图像特征）
python fid_calculator.py --mode calculate_fid --dataset fashion_mnist
    '''
    main()

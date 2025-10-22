import torch
import torchvision
import argparse
import os
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from Diffusion.Model import UNet
from Diffusion.Diffusion import GaussianDiffusionSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_federated_model(model_path, save_dir, num_samples=5000, batch_size=10):
    """
    使用联邦学习训练的模型生成样本
    
    Args:
        model_path: 模型检查点路径
        save_dir: 保存生成图像的目录
        num_samples: 要生成的图像总数
        batch_size: 每批生成的图像数量
    """
    # 模型参数 - 与训练时保持一致
    model_params = {
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 4],
        "attn": [1],
        "num_res_blocks": 2,
        "dropout": 0.0,  # 推理时不需要dropout
    }
    
    # 扩散模型参数
    diffusion_params = {
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "T": 1000,
    }
    
    # 创建模型
    model = UNet(
        T=model_params['T'],
        ch=model_params['channel'],
        ch_mult=model_params['channel_mult'],
        attn=model_params['attn'],
        num_res_blocks=model_params['num_res_blocks'],
        dropout=model_params['dropout']
    ).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model from round {checkpoint.get('round', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    model.eval()

    # 创建采样器
    sampler = GaussianDiffusionSampler(
        model,
        diffusion_params['beta_1'],
        diffusion_params['beta_T'],
        diffusion_params['T']
    ).to(device)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    generated_count = 0
    with torch.no_grad():
        while generated_count < num_samples:
            remaining = num_samples - generated_count
            current_batch_size = min(batch_size, remaining)

            # 生成噪声图像 (单通道，32x32，适用于FashionMNIST)
            noisy_image = torch.randn(
                size=[current_batch_size, 1, 32, 32], device=device
            )
            
            # 使用采样器生成图像
            sampled_images = sampler(noisy_image)
            sampled_images = sampled_images * 0.5 + 0.5  # 转换到 [0, 1] 范围
            
            # 保存图像
            for idx in range(current_batch_size):
                save_path = os.path.join(save_dir, f'federated_sample_{generated_count + idx}.png')
                save_image(sampled_images[idx], save_path)

            generated_count += current_batch_size
            print(f"Generated {generated_count} / {num_samples} images")

    print(f"All {num_samples} images saved to {save_dir}")
    torch.cuda.empty_cache()

def infer(model_path=None, save_dir="./SampledImgs/federated_samples", num_samples=5000, round_num=None):
    """
    推理函数 - 使用联邦学习训练的模型生成样本
    
    Args:
        model_path: 模型路径，如果为None则自动查找检查点
        save_dir: 保存目录
        num_samples: 生成图像数量
        round_num: 指定轮数，如果为None则使用最新检查点
    """
    if model_path is None:
        checkpoint_dir = "./Checkpoints/Federated/"
        if not os.path.exists(checkpoint_dir):
            print(f"检查点目录不存在: {checkpoint_dir}")
            return
        
        if round_num is not None:
            # 指定轮数检查点
            model_path = os.path.join(checkpoint_dir, f'global_model_round_{round_num}.pth')
            if not os.path.exists(model_path):
                print(f"指定的轮数 {round_num} 检查点不存在: {model_path}")
                return
            print(f"使用指定轮数的检查点: round_{round_num}")
        else:
            # 自动查找最新的联邦学习模型检查点（已注释）
            # checkpoint_files = []
            # for file in os.listdir(checkpoint_dir):
            #     if file.startswith('global_model_round_') and file.endswith('.pth'):
            #         try:
            #             round_num = int(file.split('_')[-1].split('.')[0])
            #             checkpoint_files.append((round_num, file))
            #         except:
            #             continue
            
            # if not checkpoint_files:
            #     print("未找到联邦学习模型检查点")
            #     return
            
            # # 使用最新的检查点
            # latest_round, latest_file = max(checkpoint_files, key=lambda x: x[0])
            # model_path = os.path.join(checkpoint_dir, latest_file)
            # print(f"使用最新的检查点: {latest_file} (轮数: {latest_round})")
            
            # 查找所有可用的检查点
            checkpoint_files = []
            for file in os.listdir(checkpoint_dir):
                if file.startswith('global_model_round_') and file.endswith('.pth'):
                    try:
                        round_num = int(file.split('_')[-1].split('.')[0])
                        checkpoint_files.append((round_num, file))
                    except:
                        continue
            
            if not checkpoint_files:
                print("未找到联邦学习模型检查点")
                return
            
            # 显示所有可用的检查点
            print("可用的检查点:")
            for round_num, filename in sorted(checkpoint_files):
                print(f"  轮数 {round_num}: {filename}")
            
            # 使用最新的检查点
            latest_round, latest_file = max(checkpoint_files, key=lambda x: x[0])
            model_path = os.path.join(checkpoint_dir, latest_file)
            print(f"使用最新的检查点: {latest_file} (轮数: {latest_round})")
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    print(f"开始生成 {num_samples} 张图像...")
    sample_federated_model(model_path, save_dir, num_samples, batch_size=10)


if __name__ == '__main__':
    '''
    python -m federated.sample_flddpm
    python -m federated.sample_flddpm --round_num 99
    '''
    parser = argparse.ArgumentParser(description='Federated DDPM Image Generation')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型检查点路径，如果为None则自动查找检查点')
    parser.add_argument('--save_dir', type=str, default='./SampledImgs/federated_samples_mse_e200_2',
                       help='保存生成图像的目录')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='要生成的图像数量')
    parser.add_argument('--round_num', type=int, default=None,
                       help='指定轮数，如果为None则使用最新检查点')
    args = parser.parse_args()
    
    infer(model_path=args.model_path, save_dir=args.save_dir, num_samples=args.num_samples, round_num=args.round_num)

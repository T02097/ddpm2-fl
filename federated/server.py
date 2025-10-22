import os
import torch
import logging
import numpy as np
from typing import List, Dict
from Diffusion.Model import UNet
from Diffusion.Diffusion import GaussianDiffusionTrainer

class FederatedServer:
    def __init__(self, model_params, diffusion_params, federated_config):
        self.config = federated_config
        self.model_params = model_params
        self.diffusion_params = diffusion_params
        
        # 使用新的UNet模型
        self.global_model = UNet(
            T=model_params['T'],
            ch=model_params['channel'],
            ch_mult=model_params['channel_mult'],
            attn=model_params['attn'],
            num_res_blocks=model_params['num_res_blocks'],
            dropout=model_params['dropout']
        )
        
        # 设备初始化
        device = torch.device(model_params.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.device = device
        self.global_model = self.global_model.to(device)
        
        # 使用GaussianDiffusionTrainer
        self.trainer = GaussianDiffusionTrainer(
            self.global_model,
            diffusion_params['beta_1'],
            diffusion_params['beta_T'],
            diffusion_params['T'],
            # warmup_steps=40  # 添加warmup_steps参数
        ).to(device)
        
        self.clients = []  # 确保初始化
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        if not os.path.exists(self.config['log_path']):
            os.makedirs(self.config['log_path'])
        logging.basicConfig(
            filename=os.path.join(self.config['log_path'], 'server.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def initialize_global_model(self, model_params, diffusion_params):
        """初始化全局模型"""
        self.global_model = UNet(
            T=model_params['T'],
            ch=model_params['channel'],
            ch_mult=model_params['channel_mult'],
            attn=model_params['attn'],
            num_res_blocks=model_params['num_res_blocks'],
            dropout=model_params['dropout']
        )
        self.global_model = self.global_model.to(self.device)
        self.global_model.train()
        
        # 初始化GaussianDiffusionTrainer
        self.trainer = GaussianDiffusionTrainer(
            self.global_model,
            diffusion_params['beta_1'],
            diffusion_params['beta_T'],
            diffusion_params['T'],
            # warmup_steps=40  # 添加warmup_steps参数
        ).to(self.device)
    
    def load_checkpoint(self, checkpoint_path=None, start_round=0):
        """
        加载检查点，支持断点续训
        
        Args:
            checkpoint_path: 检查点文件路径，如果为None则自动查找最新的检查点
            start_round: 开始训练的轮数，如果为0则从检查点中获取
            
        Returns:
            start_round: 实际开始训练的轮数
        """
        if checkpoint_path is None:
            # 自动查找最新的检查点
            result = self._find_latest_checkpoint()
            if result is None:
                print("未找到检查点，将从第0轮开始训练")
                return 0
            checkpoint_path, file_round = result
        else:
            # 从文件名中提取轮数
            try:
                file_round = int(checkpoint_path.split('_')[-1].split('.')[0])
            except:
                file_round = 0
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # 加载模型状态
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # 如果是完整的检查点（包含轮数信息）
                if isinstance(checkpoint, dict) and 'round' in checkpoint:
                    self.global_model.load_state_dict(checkpoint['model_state'])
                    start_round = checkpoint['round'] + 1
                    logging.info(f"Loaded checkpoint from round {checkpoint['round']}, continuing from round {start_round}")
                else:
                    # 如果只是模型状态字典（旧格式），从文件名推断轮数
                    self.global_model.load_state_dict(checkpoint)
                    start_round = file_round + 1
                    logging.info(f"Loaded model from {checkpoint_path}, inferred round {file_round}, continuing from round {start_round}")
                
                self.global_model = self.global_model.to(self.device)
                
                print(f"成功加载检查点: {checkpoint_path}")
                print(f"将从第 {start_round} 轮开始继续训练 (检查点轮数: {file_round})")
                
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}")
                print(f"加载检查点失败: {e}")
                start_round = 0
        else:
            print("未找到检查点，将从第0轮开始训练")
            start_round = 0
        
        return start_round
    
    def _find_latest_checkpoint(self):
        """查找最新的检查点文件"""
        if not os.path.exists(self.config['model_save_path']):
            return None
        
        checkpoint_files = []
        for file in os.listdir(self.config['model_save_path']):
            if file.startswith('global_model_round_') and file.endswith('.pth'):
                try:
                    round_num = int(file.split('_')[-1].split('.')[0])
                    checkpoint_files.append((round_num, file))
                except:
                    continue
        
        if not checkpoint_files:
            return None
        
        # 返回轮数最大的检查点
        latest_round, latest_file = max(checkpoint_files, key=lambda x: x[0])
        return os.path.join(self.config['model_save_path'], latest_file), latest_round
        
    def select_clients(self) -> List[int]:
        """选择参与训练的客户端"""
        num_selected = max(1, int(self.config['num_clients'] * self.config['client_selection_ratio']))
        return np.random.choice(self.config['num_clients'], num_selected, replace=False).tolist()
    
    def aggregate_models(self, client_models: List[Dict], client_sizes: List[int]):
        """使用FedAvg聚合客户端模型"""
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]
        
        # 初始化聚合后的模型参数
        aggregated_params = {}
        for key in client_models[0].keys():
            aggregated_params[key] = torch.zeros_like(client_models[0][key])
        
        # 加权平均
        for client_model, weight in zip(client_models, weights):
            for key in aggregated_params.keys():
                aggregated_params[key] += weight * client_model[key]
        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated_params)
    
    def save_model(self, round_idx: int):
        """保存模型"""
        if not os.path.exists(self.config['model_save_path']):
            os.makedirs(self.config['model_save_path'])
            
        save_path = os.path.join(
            self.config['model_save_path'],
            f'global_model_round_{round_idx}.pth'
        )
        
        # 保存完整的检查点信息
        checkpoint = {
            'round': round_idx,
            'model_state': self.global_model.state_dict(),
            'config': {
                'num_clients': self.config['num_clients'],
                'num_rounds': self.config['num_rounds'],
                'epochs_per_round': self.config['epochs_per_round'],
                'batch_size': self.config['batch_size'],
                'learning_rate': self.config['learning_rate']
            }
        }
        
        torch.save(checkpoint, save_path)
        logging.info(f'Saved global model at round {round_idx}')
        print(f"保存模型检查点: round_{round_idx}")
    
    def run_federated_learning(self, start_round=0):
        """运行联邦学习过程"""
        logging.info(f"Starting federated learning from round {start_round}...")
        print(f"开始联邦学习，从第 {start_round} 轮开始")
        
        for round_idx in range(start_round, self.config['num_rounds']):
            current_round = round_idx + 1
            print(f"\n第 {current_round}/{self.config['num_rounds']} 轮训练")
            
            # 选择客户端
            selected_clients = self.select_clients()
            logging.info(f"Round {round_idx}: Selected clients {selected_clients}")
            print(f"选择的客户端: {selected_clients}")
            
            # 获取客户端模型更新
            client_models = []
            client_sizes = []
            
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                model_update, data_size = client.train(
                    self.global_model.state_dict(),
                    round_idx
                )
                client_models.append(model_update)
                client_sizes.append(data_size)
                
            # 使用FedAvg聚合模型
            self.aggregate_models(client_models, client_sizes)
            
            # 保存模型
            if current_round % self.config['save_interval'] == 0:
                self.save_model(round_idx)
                
            logging.info(f"Completed round {round_idx}")
            print(f"第 {current_round} 轮训练完成")
            
        logging.info("Federated learning completed!")
        print("联邦学习训练完成！")
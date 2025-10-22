import argparse
import os
from federated.server import FederatedServer
from federated.client import FederatedClient

def main(federated_config=None, checkpoint_path=None, start_round=0, resume=False):
    # 联邦学习配置 - 使用类似Main.py的方式
    default_fed_config = {
        # 联邦学习基本参数
        "num_clients": 2,
        "num_rounds": 200,
        "epochs_per_round": 1,
        "batch_size": 80,
        "learning_rate": 1e-4,
        
        # 客户端选择策略
        "client_selection_ratio": 1.0,  # 每轮选择所有客户端
        
        # 聚合策略
        "aggregation_method": "fedavg",
        
        # 数据分布设置
        "data_distribution": "iid",  # "iid" 或 "non_iid"
        "non_iid_alpha": 0.5,
        
        # IID模式设置
        "use_custom_iid_folders": False,  # 是否使用自定义IID文件夹
        "custom_iid_path": "./custom_iid_data",  # 自定义IID数据路径
        
        # 模型保存设置
        "save_interval": 10,
        "model_save_path": "./Checkpoints/Federated_mse_e200/",
        
        # 日志设置
        "log_interval": 1,
        "log_path": "./logs/federated_mse_e200/",
    }
    
    # 使用传入的配置或默认配置
    if federated_config is not None:
        fed_config = federated_config
    else:
        fed_config = default_fed_config
    
    # 模型参数 - 与Main.py保持一致
    model_params = {
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 4],
        "attn": [1],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "device": "cuda:0",
        "grad_clip": 1.0,
    }
    
    # 扩散模型参数 - 与Main.py保持一致
    diffusion_params = {
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "T": 1000,
    }
    
    # 数据集参数
    dataset_params = {
        "im_path": "./FashionMNIST",
    }

    # 打印配置信息
    print("="*30)
    print("联邦学习DDPM配置:")
    print(f"  客户端数量: {fed_config['num_clients']}")
    print(f"  联邦学习轮数: {fed_config['num_rounds']}")
    print(f"  数据集为{dataset_params['im_path']}")
    print(f"  每轮本地训练epoch: {fed_config['epochs_per_round']}")
    print(f"  批次大小: {fed_config['batch_size']}")
    print(f"  学习率: {fed_config['learning_rate']}")
    print(f"  数据分布: {fed_config['data_distribution']}")
    if fed_config['data_distribution'] == 'iid':
        if fed_config.get('use_custom_iid_folders', False):
            print(f"  IID模式: 自定义文件夹 ({fed_config.get('custom_iid_path', './custom_iid_data')})")
        else:
            print(f"  IID模式: 标准随机分配")
    print(f"  模型: T={model_params['T']}, channel={model_params['channel']}")
    print(f"  Diffusion: beta_1={diffusion_params['beta_1']}, beta_T={diffusion_params['beta_T']}")
    print("="*30)

    # 服务器初始化
    server = FederatedServer(model_params, diffusion_params, fed_config)
    server.initialize_global_model(model_params, diffusion_params)

    # 客户端初始化
    print("\n初始化客户端...")
    for client_id in range(fed_config['num_clients']):
        print(f"  正在初始化客户端 {client_id}...")
        client = FederatedClient(
            client_id=client_id,
            model_params=model_params,
            diffusion_params=diffusion_params,
            federated_config=fed_config
        )
        # 直接传递数据集根目录，客户端会自动处理
        client.load_data(dataset_params['im_path'])
        server.clients.append(client)
    
    print(f"\n成功初始化 {len(server.clients)} 个客户端")
    
    assert len(server.clients) > 0, "No clients registered to the server!"

    # 处理继续训练
    actual_start_round = 0
    if resume or checkpoint_path is not None:
        print("尝试从检查点恢复训练...")
        actual_start_round = server.load_checkpoint(
            checkpoint_path=checkpoint_path,
            start_round=start_round
        )
    
    # 开始训练
    server.run_federated_learning(start_round=actual_start_round)

if __name__ == '__main__':
    '''
    python -m federated.train_federated

    # 从最新检查点自动继续
    python -m federated.train_federated --resume
    '''
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='检查点文件路径，如果为None则自动查找最新的检查点')
    parser.add_argument('--start_round', type=int, default=0,
                       help='开始训练的轮数，如果为0则从检查点中获取')
    parser.add_argument('--resume', action='store_true',
                       help='是否从检查点恢复训练')
    
    args = parser.parse_args()
    
    main(
        federated_config=None,
        checkpoint_path=args.checkpoint,
        start_round=args.start_round,
        resume=args.resume
    )
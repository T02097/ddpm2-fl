import yaml
import os

class FederatedConfig:
    def __init__(self, config_path='config/default.yaml'):
        # 加载配置文件
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 获取联邦学习参数
        fed_params = config['federated_params']

        # 联邦学习基本参数
        self.num_clients = fed_params['num_clients']
        self.num_rounds = fed_params['num_rounds']
        self.epochs_per_round = fed_params['epochs_per_round']
        self.batch_size = fed_params['batch_size']
        self.learning_rate = fed_params['learning_rate']

        # 客户端选择策略
        self.client_selection_ratio = fed_params['client_selection_ratio']

        # 聚合策略
        self.aggregation_method = fed_params['aggregation_method']

        # 通信设置
        self.communication_rounds = fed_params['communication_rounds']

        # 数据分布设置
        self.data_distribution = fed_params['data_distribution']
        self.non_iid_alpha = fed_params['non_iid_alpha']

        # 模型保存设置
        self.save_interval = fed_params['save_interval']
        self.model_save_path = fed_params['model_save_path']

        # 日志设置
        self.log_interval = fed_params['log_interval']
        self.log_path = fed_params['log_path']

        # 创建必要的目录
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
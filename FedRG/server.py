import copy
import logging
import warnings
from collections import OrderedDict
import numpy as np
import torch
import os

from data_preprocessing.own_transforms import get_stransform
from data_preprocessing.personalized_dataset import Dataset_Normal
from FedRG import strategies

from utils.MetricTracker import MetricTracker
from model.utils import build_SemiFed_model

from utils.record import record

from model.model_manager import get_model

class TCFNLLServer(object):
    def __init__(self, 
                 args, 
                 device, 
                 index, 
                 X_train, 
                 y_train, 
                 X_test, 
                 y_test) -> None:
        
        self.role='TCFNLLserver'

        self.args = args
        self.device = device
        self.index = index

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._initial_setup()
        self.global_acc = 0.0
        self.best_model = None

    def _initial_setup(self):
        self.model = self._build_model()
        self._build_test_dl()

    def _build_model(self, model_name=None):
        if model_name is not None:
            model = model_name
        else:
            model = self.args.model
        num_classes = self.args.datasets.num_classes
        input_channels = self.args.datasets.input_channels
        net = build_SemiFed_model(self.args,
                                    num_classes,
                                    input_channels,
                                    semi_model=self.args.algorithms.semi_model,
                                    base_model=model)
        return net

    def _forward(self, inputs):
        f, logits = self.model.inference(inputs)
        return f, logits
    
    def _set_test_mode(self):
        self.model.eval()
        self.model.to(self.device)

    def test(self,
             round_idx,
             testloader):
        
        self._set_test_mode()
        metric_tracker = MetricTracker(device=self.device)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                bs = inputs.size(0)
                _, outputs = self._forward(inputs)
                metric_tracker.update(outputs=outputs, targets=targets)
        results = metric_tracker.compute()

        # # 检查全局模型参数的均值和标准差
        # total_mean = 0
        # total_std = 0
        # param_count = 0

        # for name, param in self.model.named_parameters():
        #     mean = param.data.mean().item()
        #     std = param.data.std().item()
        #     param_count += 1
        #     total_mean += mean
        #     total_std += std
        #     print(f"[Round {round_idx}] Param: {name}, Mean: {mean:.6f}, Std: {std:.6f}")

        # 可选：记录总均值和标准差
        # print(
        #     f"[Round {round_idx}] Global Model Param Summary — Avg Mean: {total_mean / param_count:.6f}, Avg Std: {total_std / param_count:.6f}")

        scalar_dict = {
            f'{self.role}:{self.index} real top1 acc': results['real_top1_acc'],
            f'{self.role}:{self.index} real top5 acc': results['real_top5_acc'],
            f'{self.role}:{self.index} old top1 acc': results['old_top1_acc'],
            f'{self.role}:{self.index} old top5 acc': results['old_top5_acc'],
            f'{self.role}:{self.index} avg loss': results['avg_loss'],
            f'{self.role}:{self.index} precision': results['precision'],
            f'{self.role}:{self.index} recall': results['recall'],
            f'{self.role}:{self.index} f1_score': results['f1_score']
        }
        print(scalar_dict)
        if self.args.record:
            record(record_tool=self.args.record_tool, scalar = scalar_dict, step = round_idx)
        if results['real_top1_acc'] > self.global_acc:
            logging.info(f"This round: {round_idx}, Global acc is: {results['real_top1_acc']}, new best acc is: {results['real_top1_acc']}")
            self.global_acc = results['real_top1_acc']
            self.best_model = self.get_model_params()
            # 保存最优模型
            # torch.save(self.best_model, f"best_model_round_{round_idx}.pt")
        self.model.cpu()
        return results['real_top1_acc']
    
    def _build_test_dl(self):
        test_trans = get_stransform(self.args.datasets.dataset_name, train=False)

        test_ds = Dataset_Normal(data=self.X_test,
                                 targets=self.y_test,
                                 dataset=self.args.datasets.dataset_name,
                                 transform=test_trans)
        self.test_dl = torch.utils.data.DataLoader(dataset=test_ds,
                                                   batch_size=self.args.bs, shuffle=True,
                                                   drop_last=False)

    def test_server(self, round):
        acc = self.test(round, self.test_dl)
        logging.info(f"This round: {round}, Global acc is: {acc}")
        return acc

    def download_info(self):
        down_info = {"GLOBAL_MODEL_PARAM": copy.deepcopy(self.get_model_params())}
        return down_info
    
    def receive_message(self, upload_message):

        self.clients_uploaded_message = upload_message
        """
        a dict of clients info 
        {index: 
                {"ROLE": server/client, 
                "LOCAL_MODEL_PARAM": {key:params},
                "SAMPLE_NUM": client_num}, 
            , ... }
        """
    def _reorganize_uploded_info(self, target_key):
        key_uploaded_info = dict()
        # key is the index of client
        for key in self.clients_uploaded_message.keys():
            if target_key in self.clients_uploaded_message[key]:
                key_uploaded_info[key] = self.clients_uploaded_message[key][target_key]
        
        return key_uploaded_info

    def calculate_client_model_differences(self, clients_params):
        """Calculate the difference between each client's model parameters and the server model parameters.
        
        Args:
            clients_params (dict): Dictionary of client parameters {client_idx: model_params}
        
        Returns:
            dict: Dictionary containing differences for each client {client_idx: {param_name: difference}}
        """
        total_differences = {}
        stat_diff = {}
        server_params = self.get_model_params()
        for client_idx, client_params in clients_params.items():
            client_diff = {}
            scalar_dict = {}
            total_differences[client_idx] = 0
            stat_diff[client_idx] = 0
            for param_name, server_param in server_params.items():
                if param_name in client_params:
                    # Calculate difference using L2 norm
                    diff = torch.norm(client_params[param_name].float() - server_param.float()).item()
                    total_differences[client_idx] += diff
                    if 'running' in param_name:
                        stat_diff[client_idx] += diff
                    scalar_dict[f'{client_idx}_diff_{param_name}'] = diff
            scalar_dict[f'{client_idx}_stat_diff'] = stat_diff[client_idx]
            scalar_dict[f'{client_idx}_total_diff'] = total_differences[client_idx]
            if self.args.record:
                record(record_tool=self.args.record_tool,scalar = scalar_dict, step = 1)            
            logging.info(f'Client {client_idx} total difference: {total_differences[client_idx]}')
            logging.info(f'Client {client_idx} stat difference: {stat_diff[client_idx]}')

    def aggregation(self,round=None,agg='general'):
        data_num_weighted = self._reorganize_uploded_info("SAMPLE_NUM")
        clients_params = self._reorganize_uploded_info("LOCAL_MODEL_PARAM")
        # self.calculate_client_model_differences(copy.deepcopy(clients_params))
        if len(self.clients_uploaded_message) == 0:
            warnings.warn("At aggregation stage, there is no client model")
        else:
            logging.info("updata global model by the number of data weighted")
            if agg == 'general' :
                model_params = strategies.federated_averaging_by_params(clients_params, data_num_weighted)
                if model_params is None:
                    logging.info("No client model to aggregate")
                    return
            # self.model_differences = self.calculate_model_update_direction(model_params)
            self.set_model_params(model_params)
        self.clients_uploaded_message.clear()


    def get_model_params(self):
        return {k: copy.deepcopy(val.cpu())
                for k, val in self.model.state_dict().items()}
    
    def get_model_params_wo_stat_info(self):
        return [copy.deepcopy(val.cpu())
                for val in self.model.parameters()]
    
    def set_model_params(self, params):
        self.model.load_state_dict(params) #没有的话就保持不变
        self.model.cpu()
    
    def set_model_params_wo_stat_info(self, params):
        for idx, val in enumerate(self.model.parameters()):
            val.data.copy_(params[idx])
    
    def calculate_model_update_direction(self, model_params):
        server_params = self.get_model_params()
        model_update_direction = OrderedDict()
        for key in server_params.keys():
            model_update_direction[key] = server_params[key] - model_params[key]
        return model_update_direction
    
    def save_model(self):

        # 创建基础checkpoints目录
        base_path = os.path.join("checkpoints-no-projector", self.args.algorithms.algorithm_name)
        os.makedirs(base_path, exist_ok=True)
        algorithm_not_contain_list = ['VHL_data',
                                    'VHL_label_from',
                                    'VHL_label_style',
                                    'VHL_label_shift',
                                    'VHL_feat_align',
                                    'VHL_inter_domain_mapping',
                                    'VHL_class_match',
                                    'VHL_feat_detach',
                                    'VHL_noise_contrastive',
                                    'VHL_feat_align_inter_domain_weight',
                                    'VHL_feat_align_inter_cls_weight',
                                    'VHL_noise_supcon_weight',
                                    'VHL_ft_prox_weight',
                                    'model_feature_dim',
                                    'VHL_alpha',
                                    'VHL_num',
                                    'VHL_dataset_list',
                                    'generative_dataset_root_path',
                                    'dataset_image_size',
                                    'dataloader_workers']
        # 构建安全的文件名，避免在文件名中包含字典、列表等特殊字符
        import hashlib, re

        def safe_str(s, max_len=64):
            # 将 s 转为字符串，移除不安全字符，只保留字母数字和少量符号，超长则截断
            s = str(s)
            s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
            if len(s) > max_len:
                s = s[:max_len]
            return s

        parts = []
        parts.append(f"client{self.args.client_number}")
        parts.append(f"rate{self.args.client_number_per_round}")
        parts.append(f"round{self.args.global_round}")
        parts.append(f"{safe_str(self.args.model)}")
        parts.append(f"bs{self.args.bs}")
        parts.append(f"alpha{safe_str(self.args.partition_alpha)}")
        parts.append(f"seed{self.args.seed}")
        parts.append(f"trainseed{self.args.train_seed}")

        # algorithms: 对于简单类型直接放入；对复杂类型使用短哈希
        for key, value in self.args.algorithms.items():
            if key == 'algorithm_name' or key in algorithm_not_contain_list:
                continue
            if isinstance(value, (int, float, bool)):
                parts.append(f"{safe_str(key)}{safe_str(value)}")
            elif isinstance(value, str):
                # 字符串保留（但清理）
                parts.append(f"{safe_str(key)}{safe_str(value, max_len=32)}")
            else:
                # 复杂类型（list/dict等）用短哈希替代
                h = hashlib.sha1(repr(value).encode()).hexdigest()[:8]
                parts.append(f"{safe_str(key)}h{h}")

        # dataset name
        if hasattr(self.args, 'datasets') and 'dataset_name' in self.args.datasets:
            parts.append(f"dataset{safe_str(self.args.datasets.dataset_name)}")

        safe_filename = "_".join(parts)
        if not safe_filename.endswith('.pth'):
            safe_filename = safe_filename + '.pth'

        # Primary: use simple filename equal to algorithm name
        algo_name = safe_str(self.args.algorithms.algorithm_name) if hasattr(self.args, 'algorithms') and hasattr(self.args.algorithms, 'algorithm_name') else 'model'
        primary_name = f"{algo_name}.pth"
        save_path = os.path.join(base_path, primary_name)

        model_to_save = self.get_model_params()
        try:
            torch.save(model_to_save, save_path)
            logging.info(f"Model saved to {save_path}")
            return
        except Exception as e:
            logging.warning(f"Failed to save using primary filename {save_path}: {e}; switching to short hashed filename.")

        # Fallback: use a short hashed filename and write metadata mapping
        import hashlib, json, time
        short_hash = hashlib.sha1(primary_name.encode()).hexdigest()[:12]
        short_name = f"model_{short_hash}.pth"
        short_path = os.path.join(base_path, short_name)
        # write model
        torch.save(model_to_save, short_path)
        # write metadata for mapping
        meta = {
            'original_filename': primary_name,
            'short_filename': short_name,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        }
        meta_path = os.path.join(base_path, f"model_{short_hash}.meta.json")
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as me:
            logging.warning(f"Failed to write metadata file {meta_path}: {me}")
        logging.info(f"Model saved to {short_path} (metadata at {meta_path})")
        
    
    
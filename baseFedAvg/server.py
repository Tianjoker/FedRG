import copy
import logging
import warnings
from collections import OrderedDict
import numpy as np
import torch
import os

from data_preprocessing.own_transforms import get_stransform
from data_preprocessing.personalized_dataset import Dataset_Normal
from  baseFedAvg import strategies

from utils.MetricTracker import MetricTracker

from utils.record import record

from model.model_manager import get_model

class Server(object):
    def __init__(self, 
                 args, 
                 device, 
                 index, 
                 X_train, 
                 y_train, 
                 X_test, 
                 y_test) -> None:
        
        self.role='server'

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

    def _build_model(self, model_name=None, num_classes=None, input_channels=None):
        return get_model(self.args, model_name, num_classes, input_channels)

    def _forward(self, inputs):
        feat, outputs = self.model(inputs)
        return feat, outputs
    
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

        total_mean = 0
        total_std = 0
        param_count = 0

        for name, param in self.model.named_parameters():
            mean = param.data.mean().item()
            std = param.data.std().item()
            param_count += 1
            total_mean += mean
            total_std += std
            # print(f"[Round {round_idx}] Param: {name}, Mean: {mean:.6f}, Std: {std:.6f}")
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

    def download_info(self, round):
        down_info = dict()
        down_info['GLOBAL_MODEL_PARAM'] = self.get_params(self.model,
                                                            type='model',
                                                            with_nograd=self.args.share_with_nongrad)
        if not hasattr(self, 'global_update'):
            down_info['GLOBAL_UPDATE_PARAM'] = None
            down_info['GLOBAL_EMA_UPDATE_PARAM'] = None
        else:
            down_info['GLOBAL_UPDATE_PARAM'] = self.global_update
            down_info['GLOBAL_EMA_UPDATE_PARAM'] = self.global_ema_update
        return down_info

    def get_params(self, model, type='model', with_nograd=True):
        if type == 'model':
            if with_nograd:
                return {k: copy.deepcopy(val.cpu())
                        for k, val in model.state_dict().items()}
            else:
                return {k: copy.deepcopy(val.cpu())
                        for k, val in model.named_parameters()}
        elif type == 'param':
            return {k: copy.deepcopy(val.cpu())
                    for k, val in model.items()}
        else:
            raise ValueError('Invalid type: {}'.format(type))

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
        clients_model_deltas = self._reorganize_uploded_info("LOCAL_MODEL_DELTA_PARAM")
        clients_params = self._reorganize_uploded_info("LOCAL_MODEL_PARAM")

        if len(self.clients_uploaded_message) == 0:
            warnings.warn("At aggregation stage, there is no client model")
            return

        logging.info(f"Aggregation round {round}: method = {agg}")

        if agg == 'general' :
            model_params = strategies.federated_averaging_by_params(clients_params, data_num_weighted)
            if model_params is None:
                logging.warning("No client model to aggregate (param).")
                return
            self.set_model_params(model_params)
            self.model_differences = self.calculate_model_update_direction(model_params)
            logging.info("✅ Global model updated using parameter averaging.")

        elif agg == 'direction':
            agg_model_delta = strategies.federated_averaging_by_params(clients_model_deltas, data_num_weighted)
            self.global_update = copy.deepcopy(agg_model_delta)
            # EMA
            if not hasattr(self, 'global_ema_update'):
                self.global_ema_update = copy.deepcopy(self.global_update)
            else:
                for name, param in self.global_ema_update.items():
                    param.data.copy_(param.data * self.args.algorithms.ema_decay + self.global_update[name].data * (
                                1 - self.args.algorithms.ema_decay))

            new_model_params = copy.deepcopy(self.model.state_dict())
            for name, param in agg_model_delta.items():
                if new_model_params[name].dtype != param.data.dtype:
                    new_model_params[name] = new_model_params[name].to(param.data.dtype)
                new_model_params[name] += param.data
            self.model.load_state_dict(new_model_params)

            self.model_differences = self.calculate_model_update_direction(self.get_params(self.model))
            logging.info("✅ Global model updated using direction aggregation.")
        else:
            raise NotImplementedError(f"Aggregation method '{agg}' not supported.")

        self.clients_uploaded_message.clear()


    def get_model_params(self):
        return {k: copy.deepcopy(val.cpu())
                for k, val in self.model.state_dict().items()}

    
    def set_model_params(self, params):
        self.model.load_state_dict(params)
        self.model.cpu()

    
    def calculate_model_update_direction(self, model_params):
        server_params = self.get_model_params()
        model_update_direction = OrderedDict()
        for key in server_params.keys():
            model_update_direction[key] = server_params[key] - model_params[key]
        return model_update_direction
    
    def save_model(self):

        base_path = os.path.join("checkpoints", self.args.algorithms.algorithm_name)
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
        # 构建文件名，使用相关参数
        filename = f"client{self.args.client_number}_rate{self.args.client_number_per_round}_round{self.args.global_round}"
        filename += f"_{self.args.model}_bs{self.args.bs}_alpha{self.args.partition_alpha}_seed{self.args.seed}_trainseed{self.args.train_seed}"
        for key, value in self.args.algorithms.items():
            if key == 'algorithm_name':
                continue
            if type(value) == str:
                continue
            if key in algorithm_not_contain_list:
                continue
            filename += f"{key}{value}_"
        for key, value in self.args.optim.items():
            if key in ['wd']:
                continue
            filename += f"{key}{value}_"
        for key, value in self.args.datasets.items():
            if key == 'dataset_name':
                filename += f"{key}{value}"
        filename += f".pth"
            
        save_path = os.path.join(base_path, filename)
        
        # 保存模型
        torch.save(self.best_model, save_path)
        logging.info(f"Model saved to {save_path}")
        
    
    
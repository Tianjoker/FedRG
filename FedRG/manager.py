import logging
import copy
import math
import time
from typing import Dict
import threading
import multiprocessing as mp
import torch
import numpy as np
from tqdm import tqdm
from utils.utils import set_random
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numbers
from utils.save_client_selection import _save_client_selection,load_client_selection_cached

from data_preprocessing.load import (
    split_data_into_clients
)
import concurrent.futures

from sklearn.mixture import GaussianMixture

from FedRG.server import TCFNLLServer
from FedRG.client import TCFNLLClient

class TCFNLLManager(object):
    def __init__(self, args) -> None:
        self.args = args
        self._initial_setup()

    def _initial_setup(self):
        self._setup_device()  # {client_idx:gpu_idx}
        self._setup_dataset()
        self._setup_server()
        if 'nl' in self.args.imperfect_scenario.type and self.args.imperfect_scenario.noise_type == 'random':
            self._random_noise_simulation()
        self._setup_clients()

    def _random_noise_simulation(self):
        level_n_system, level_n_lowerb = self.args.imperfect_scenario.random_nl_ratio
        gamma_s = np.random.binomial(1, level_n_system, self.args.client_number) # system level noise
        gamma_c_initial = np.random.rand(self.args.client_number)
        gamma_c_initial = (1 - level_n_lowerb) * gamma_c_initial + level_n_lowerb
        self.random_nl_system_level_list = gamma_s * gamma_c_initial
        logging.info(self.random_nl_system_level_list)
    def _setup_device(self):
        self.client_device_dict = {} # {client_idx:gpu_idx}
        device_num = len(self.args.gpu_list)
        for i in range(self.args.client_number):
            self.client_device_dict[i] =  torch.device("cuda:" + str(self.args.gpu_list[i%device_num]) if torch.cuda.is_available() else "cpu")
        self.server_device_dict = {'server':torch.device("cuda:" + str(self.args.gpu_list[0]) if torch.cuda.is_available() else "cpu")} 

    def _setup_dataset(self):
        if self.args.long_tail:
            self.client_X_train_dict, self.client_y_train_dict, self.X_test, self.y_test_np=split_data_into_clients(
                                                                            dataset=self.args.datasets.dataset_name,
                                                                            datadir=self.args.datasets.data_dir,
                                                                            client_number=self.args.client_number,
                                                                            partition_method=self.args.partition_method,
                                                                            long_tail=self.args.long_tail,
                                                                            class_num=self.args.datasets.num_classes,
                                                                            partition_alpha=self.args.partition_alpha,
                                                                            seed=self.args.seed,
                                                                            long_tail_imb_factor=self.args.long_tail_imb_factor
                                                                            )
        else:
            self.client_X_train_dict, self.client_y_train_dict, self.X_test, self.y_test_np, self.X_train_total, self.y_train_total=split_data_into_clients(
                                                                                        dataset=self.args.datasets.dataset_name,
                                                                                        datadir=self.args.datasets.data_dir,
                                                                                        client_number=self.args.client_number,
                                                                                        partition_method=self.args.partition_method,
                                                                                        long_tail=self.args.long_tail,
                                                                                        class_num=self.args.datasets.num_classes,
                                                                                        partition_alpha=self.args.partition_alpha,
                                                                                        seed=self.args.seed
                                                                                        )
    def _setup_server(self):
        self.server = TCFNLLServer(args=self.args,
                             device=self.server_device_dict['server'],
                             index=1,
                             X_train=None,
                             y_train=None,
                             X_test=self.X_test,
                             y_test=self.y_test_np)
        
    def _setup_clients(self):
        self.clients_dict: Dict[int:TCFNLLClient] = {}
        for i in self.client_X_train_dict.keys():
            if 'nl' in self.args.imperfect_scenario.type and self.args.imperfect_scenario.noise_type == 'random':
                self.clients_dict[i] = TCFNLLClient(args=self.args,
                                              device=self.client_device_dict[i],
                                              index=i,
                                              X_train=self.client_X_train_dict[i],
                                              y_train=self.client_y_train_dict[i],
                                              X_test=self.X_test,
                                              y_test=self.y_test_np,
                                              random_noise_level=self.random_nl_system_level_list[i])
            else:
                self.clients_dict[i] = TCFNLLClient(args=self.args,
                                            device=self.client_device_dict[i],
                                            index=i,
                                            X_train=self.client_X_train_dict[i],
                                            y_train=self.client_y_train_dict[i],
                                            X_total=self.X_train_total,
                                            y_total=self.y_train_total,
                                            X_test=self.X_test,
                                            y_test=self.y_test_np)
        
    def update_clients_lr(self, lr, client_indexes):
        for i in client_indexes:
            self.clients_dict[i].update_lr(lr)
            
    def update_optimizer_lr(self, opt, lr):
        opt.param_groups[0]['lr'] = lr

    def client_sampling(self,  client_num_in_total, client_num_per_round):
        # logging.info("Client Sample Probability: %s "%(str(p)))
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            # make sure for each comparison, we are selecting the same clients each round
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.selected_clients = client_indexes

        return client_indexes
    
    def train(self):


        #原版的就只有一个set_random  改版的会在2 3 阶段新设置random以保证复现性
        set_random(self.args.train_seed)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        import os
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 当前文件上两级目录: IFNLL/
        warm_model_dir = os.path.join(BASE_DIR, "checkpoints-no-projector", self.args.algorithms.algorithm_name)
        # print(warm_model_dir)
        cotrain_model_dir = os.path.join(BASE_DIR, "checkpoints-250", self.args.algorithms.algorithm_name)
        # print(cotrain_model_dir)
        # simplified filename: just algorithm name
        warm_model_path = os.path.join(warm_model_dir, f"{self.args.algorithms.algorithm_name}.pth")
        cotrain_model_path = os.path.join(cotrain_model_dir, f"{self.args.algorithms.algorithm_name}.pth")

        client_selection_cache = load_client_selection_cached()
        print(client_selection_cache)

        logging.info("---------------------First Stage- 表征暖启----------------------")
        if self.args.algorithms.use_cache_warmmodel:
            logging.info(f"检测到warm阶段模型，直接加载并跳过warm阶段: {warm_model_path}")
            # 加载模型参数到server
            state_dict = torch.load(warm_model_path, map_location=self.server.device)
            self.server.set_model_params(state_dict)

            # 同步一下所有客户端的数据      正式实验时 这个最好放到限免 不在if里
            for num, client_index in enumerate(range(self.args.client_number)):
                download_info = self.server.download_info()
                # selected clients parallel running the following steps.
                client: TCFNLLClient = self.clients_dict[client_index]
                client.client_update_local_info(download_info,stage="gengxin")

        else:
            for round in tqdm(range(self.args.algorithms.warm_round), desc='Warmup Communication Round'):
                download_info = self.server.download_info()
                client_indexes = self.client_sampling(  # 每一轮Sample全部client
                    self.args.client_number,
                    self.args.algorithms.selected_num)
                if self.args.algorithms.save_selection:
                    _save_client_selection(stage="warmup", round_idx=round, client_indexes=client_indexes)

                logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
                self.train_warm_round(round, client_indexes, download_info)
                agg = self.args.algorithms.aggregator if hasattr(self.args.algorithms, 'aggregator') else "general"
                self.server.aggregation(round,agg=agg)
                self.server.test_server(round)
            # warm阶段结束后保存模型
            # self.server.save_model()




        # logging.info("---------------------单独训练分类头----------------------")
        # for round in tqdm(range(self.args.algorithms.classfier_round), desc='单独训练分类头 Communication Round'):
        #     download_info = self.server.download_info()
        #     # client_indexes = self.client_sampling(  # 每一轮Sample全部client
        #     #     self.args.client_number,
        #     #     self.args.algorithms.selected_num)
        #     client_indexes=[1]
        #     logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
        #     self.train_classfier_traing_round(round, client_indexes, download_info)
        #     agg = self.args.algorithms.aggregator if hasattr(self.args.algorithms, 'aggregator') else "general"
        #     self.server.aggregation(round, agg=agg)
        #     self.server.test_server(round)
        #
        # logging.info("---------------------簇中心优化 无状态转移矩阵----------------------")
        # for round in tqdm(range(self.args.algorithms.sce_round), desc='with out T'):
        #     download_info = self.server.download_info()
        #     client_indexes = self.client_sampling(  # 每一轮Sample全部client
        #         self.args.client_number,
        #         self.args.algorithms.selected_num)
        #     logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
        #     self.train_cuxin_round(round, client_indexes, download_info)
        #
        #     agg = self.args.algorithms.aggregator if hasattr(self.args.algorithms, 'aggregator') else "general"
        #     self.server.aggregation(round, agg=agg)
        #     self.server.test_server(round)

        # for round in tqdm(range(self.args.algorithms.sce_round), desc='all sce Round'):
        #     download_info = self.server.download_info()
        #     client_indexes = self.client_sampling(  # 每一轮Sample全部client
        #         self.args.client_number,
        #         self.args.algorithms.selected_num)
        #     logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
        #     self.train_sce_round(round, client_indexes, download_info)
        #     agg = self.args.algorithms.aggregator if hasattr(self.args.algorithms, 'aggregator') else "general"
        #     self.server.aggregation(round,agg=agg)
        #     self.server.test_server(round)

        # for round in tqdm(range(self.args.algorithms.cotraining_round), desc='单独训练分类头 Communication Round'):
        #     download_info = self.server.download_info()
        #     client_indexes = self.client_sampling(  # 每一轮Sample全部client
        #         self.args.client_number,
        #         self.args.algorithms.selected_num)
        #     logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
        #     self.train_classfier_traing_round(round, client_indexes, download_info)
        #     agg = self.args.algorithms.aggregator if hasattr(self.args.algorithms, 'aggregator') else "general"
        #     self.server.aggregation(round, agg=agg)
        #     self.server.test_server(round)

        # logging.info("---------------------不考虑对比学习的第一阶段 更新backbone和分类头150round----------------------")
        # for round in tqdm(range(self.args.algorithms.warm_round), desc='warm_without_cl Communication Round'):
        #     download_info = self.server.download_info()
        #     client_indexes = self.client_sampling(  # 每一轮Sample全部client
        #         self.args.client_number,
        #         self.args.algorithms.selected_num)
        #     logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
        #     self.train_cotraing_round_without_contrastive(round, client_indexes, download_info)
        #     agg = self.args.algorithms.aggregator if hasattr(self.args.algorithms, 'aggregator') else "general"
        #     self.server.aggregation(round, agg=agg)
        #     self.server.test_server(round)

        # logging.info("---------------------Second Stage- co-training----------------------")
        # if self.args.algorithms.use_cache_cotrainmodel:
        #     logging.info(f"检测到warm+cotraining阶段模型，直接加载并跳过: {cotrain_model_path}")
        #     # 加载模型参数到server
        #     state_dict = torch.load(cotrain_model_path, map_location=self.server.device)
        #     self.server.set_model_params(state_dict)
        #     # 同步一下所有客户端的数据      正式实验时 这个最好放到限免 不在if里
        #     for num, client_index in enumerate(range(self.args.client_number)):
        #         download_info = self.server.download_info()
        #         # selected clients parallel running the following steps.
        #         client: TCFNLLClient = self.clients_dict[client_index]
        #         client.client_update_local_info(download_info)
        # else:
        #     # set_random(self.args.train_seed + self.args.algorithms.warm_round)
        #     for round in tqdm(range(self.args.algorithms.cotraining_round), desc='CO-training Communication Round'):
        #         download_info = self.server.download_info()
        #         if not self.args.algorithms.use_cache_warmmodel:
        #             client_indexes = self.client_sampling(  # 每一轮Sample全部client
        #                 self.args.client_number,
        #                 self.args.algorithms.selected_num)
        #         else:
        #             client_indexes = client_selection_cache.get("co-training", {}).get(f"round_{round}", [])
        #         # if self.args.algorithms.save_selection:
        #         #     _save_client_selection(stage="co-training", round_idx=round, client_indexes=client_indexes)
        #         logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
        #         self.train_cotraing_round(round, client_indexes, download_info)
        #         agg = self.args.algorithms.aggregator if hasattr(self.args.algorithms, 'aggregator') else "general"
        #         self.server.aggregation(round, agg=agg)
        #         self.server.test_server(round)
        #     # self.server.save_model()


        # logging.info("---------------------relabel----------------------")
        # for round in tqdm(range(1), desc='relable'):
        #     client_indexes = self.client_sampling(  # 每一轮Sample全部client
        #         self.args.client_number,
        #         self.args.client_number)
        #     logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
        #     self.relable_round(round, client_indexes)

        logging.info("---------------------Third Stage- fine-tune----------------------")
        # set_random(self.args.train_seed + self.args.algorithms.warm_round+self.args.algorithms.cotraining_round)
        for round in tqdm(range(self.args.algorithms.finetune_round), desc='Fine-tune Communication Round'):
            download_info = self.server.download_info()
            if not self.args.algorithms.use_cache_warmmodel:
                client_indexes = self.client_sampling(  # 每一轮Sample全部client
                    self.args.client_number,
                    self.args.algorithms.selected_num)
            else:
                client_indexes = client_selection_cache.get("fine-tune", {}).get(f"round_{round}", [])
            if self.args.algorithms.save_selection:
                _save_client_selection(stage="fine-tune", round_idx=round, client_indexes=client_indexes)
            logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
            self.train_finetune_round(round, client_indexes, download_info)
            agg = self.args.algorithms.aggregator if hasattr(self.args.algorithms, 'aggregator') else "general"
            self.server.aggregation(round, agg=agg)
            self.server.test_server(round)

            # if round == 50:
            #     client_indexes = self.client_sampling(  # 每一轮Sample全部client
            #         self.args.client_number,
            #         self.args.client_number)
            #     logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
            #     self.relable_round(round, client_indexes)


    def train_locally_per_round(self, round, selected_clients, download_info):
        upload_info = dict()

        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info)

            client.local_train(round)

            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))

    def train_warm_round(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info,stage="warm")
            client.local_warm_train(round)
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))

    def train_sce_round(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info,stage="sce")
            client.local_sce_train(round)
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))


    def train_cotraing_round(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info,stage="cotraining")
            client.local_cotraining_train(round)
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))

    def train_train_visualization(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info,stage="cotraining")
            client.local_cotraining_train(round)
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))

    def train_cotraing_round_without_contrastive(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            # 不考虑对比学习的第一阶段 更新backbone和分类头150round  这一阶段聚合和更新的内容包括backbone+分类头
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info,stage="finetune_agg_include_keys")
            client.local_warm_without_cl_train(round)
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))

    def train_classfier_traing_round(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info,stage="classfier")
            client.local_classfier_train(round)
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))

    def train_finetune_round(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info,stage="finetune")
            client.local_finetune_train(round)
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))

    def relable_round(self, round, selected_clients):
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.local_relable(round)
        logging.info("客户端进行本地relable,client_indexes = %s" % str(selected_clients))

    def train_finetune_with_T_round(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info,stage="finetune")
            client.local_finetune_train_with_T(round)
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))

    def train_cuxin_round(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: TCFNLLClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info, stage="cuxin")
            client.local_cuxin_train(round)
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))
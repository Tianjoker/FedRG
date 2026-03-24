import copy
import logging
import math
import time
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
from omegaconf import OmegaConf
from collections import Counter
from sklearn.mixture import GaussianMixture

from model.model_manager import get_model
from sklearn.metrics.pairwise import cosine_similarity
from baseFedAvg.client import Client
import os

from utils.save_client_selection import _save_client_selection
from model.utils import (
    consistency_loss_fixmatch,
    consistency_loss_freematch,
    info_nce_loss,
    build_SemiFed_model,
    dino_loss_fn,
    vicreg_loss,
    eval_dimensional_collapse
)
from loss.SymCE_loss import SCELoss

from utils.averager import AverageMeter
from utils.metric import (
    accuracy,
    top_one_accuracy
)
from utils.MetricTracker import MetricTracker
from data_preprocessing.personalized_dataset import (
    Dataset_Normal,
    Dataset_WeakStrong, Dataset_WeakStrong_true, Dataset_MultiCrop, Dataset_with_WeakStrong_SampleIndex
)
from data_preprocessing.utils import record_y_distribution
from data_preprocessing.own_transforms import get_stransform, MultiCropTransform
from utils.record import record
from utils.imp_utils import (
    noisification,
    noisification_in_client
)

from loss.loss_manager import LossManager

class TCFNLLClient(object):

    def __init__(self,
                 args,
                 device,
                 index,
                 X_train,
                 y_train,
                 X_total,
                 y_total,
                 X_test,
                 y_test,
                 **kwargs) -> None:

        self.args = args
        self.device = device
        self.index = index

        self.X_train_total=X_total
        self.y_train_total=y_total
        self.y_train_total_noise = copy.deepcopy(self.y_train_total)

        self.X_train = X_train
        self.y_true_train = y_train
        self.y_train = copy.deepcopy(self.y_true_train)
        self.X_test = X_test
        self.y_test = y_test
        self.relabeled = False
        self.pseudo_labels_memory = None
        if 'random_nozise_level' in kwargs:
            self.random_noise_level = kwargs['random_noise_level']

        self.role = 'TCFNLLClient'
        self.sample_num = len(self.y_true_train)
        self._local_data_stats()
        self._initial_setup()

    def _initial_setup(self):
        self._imperfect_data_environment_simulation()
        self.forward_time = 0
        self.backward_time = 0
        self._build_data_loader()
        self.loss_manager = LossManager(self.args, self.device)
        self.model = self._build_model()
        self.opt = self._build_optimizer(self.model.parameters())

    def update_noise_labels(self, is_noise):
        self.noise_labels = is_noise

    def _build_data_loader(self):
        if self.args.algorithms.semi_model == "DINO":
            train_trans = MultiCropTransform(global_crops_number=2, local_crops_number=4)
            if 'nl' in self.args.imperfect_scenario.type and 'pl' not in self.args.imperfect_scenario.type:
                self.local_train_lb_ds = Dataset_MultiCrop(data=self.X_train,
                                                                 targets=self.y_train,
                                                                 ulb=False,
                                                                 targets_true=self.y_true_train,
                                                                 dataset=self.args.datasets.dataset_name,
                                                                 transform=train_trans)
                self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
                                                                     batch_size=self.args.bs, shuffle=True,
                                                                     drop_last=False)
                # global
                self.local_train_lb_ds_global = Dataset_MultiCrop(data=self.X_train_total,
                                                                        targets=self.y_train_total_noise,
                                                                        ulb=False,
                                                                        targets_true=self.y_train_total,
                                                                        dataset=self.args.datasets.dataset_name,
                                                                        transform=train_trans)
                self.local_train_lb_dl_global = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds_global,
                                                                            batch_size=self.args.bs, shuffle=True,
                                                                            drop_last=False)

        else:
            train_trans = get_stransform(self.args.datasets.dataset_name, train=True)
            if 'nl' in self.args.imperfect_scenario.type and 'pl' not in self.args.imperfect_scenario.type:
                self.local_train_lb_ds = Dataset_with_WeakStrong_SampleIndex(data=self.X_train,
                                                            targets=self.y_train,
                                                            ulb=False,
                                                            targets_true=self.y_true_train,
                                                            dataset=self.args.datasets.dataset_name,
                                                            transform=train_trans)
                self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
                                                                     batch_size=self.args.bs, shuffle=True,
                                                                     drop_last=False)
                #global
                self.local_train_lb_ds_global = Dataset_WeakStrong_true(data=self.X_train_total,
                                                            targets=self.y_train_total_noise,
                                                            ulb=False,
                                                            targets_true=self.y_train_total,
                                                            dataset=self.args.datasets.dataset_name,
                                                            transform=train_trans)
                self.local_train_lb_dl_global = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds_global,
                                                                     batch_size=self.args.bs, shuffle=True,
                                                                     drop_last=False)
            elif 'pl' in self.args.imperfect_scenario.type:
                if self.lb_num > 0:
                    self.local_train_lb_ds = Dataset_WeakStrong_true(data=self.X_train_lb,
                                                                targets=self.y_train_lb,
                                                                ulb=False,
                                                                targets_true=self.y_true_train,
                                                                dataset=self.args.datasets.dataset_name,
                                                                transform=train_trans)
                    self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
                                                                         batch_size=self.args.bs, shuffle=True,
                                                                         drop_last=False)
                if self.ulb_num > 0:
                    self.local_train_ulb_ds = Dataset_WeakStrong_true(data=self.X_train_ulb,
                                                                 targets=self.y_train_ulb,
                                                                 ulb=True,
                                                                 targets_true=self.y_true_train,
                                                                 dataset=self.args.datasets.dataset_name,
                                                                 transform=train_trans)
                    self.local_train_ulb_dl = torch.utils.data.DataLoader(dataset=self.local_train_ulb_ds,
                                                                          batch_size=self.args.bs, shuffle=True,
                                                                          drop_last=False)
            elif 'general' in self.args.imperfect_scenario.type:
                self.local_train_lb_ds = Dataset_WeakStrong_true(data=self.X_train,
                                                            targets=self.y_true_train,
                                                            ulb=False,
                                                            targets_true=self.y_true_train,
                                                            dataset=self.args.datasets.dataset_name,
                                                            transform=train_trans)
                self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
                                                                     batch_size=self.args.bs, shuffle=True,
                                                                     drop_last=False)

    def _forward(self, inputs=None):
        x_w, x_s = inputs
        if self.args.algorithms.semi_model == 'SimSiam':
            self.forward_time += 2
            unsup_f, sup_f, unsup_loss, sup_logits = self.model(x_w, x_s)
            return unsup_f, sup_f, unsup_loss, sup_logits
        if self.args.algorithms.semi_model == 'SimCLR':
            self.forward_time += 2
            unsup_f, sup_f, unsup_logits, sup_logits = self.model(x_w, x_s)

            # [2*bs], [2*bs], [2*bs], [2*bs]
            return unsup_f, sup_f, unsup_logits, sup_logits

        if self.args.algorithms.semi_model == 'BYOL':
            self.forward_time += 2
            unsup_f_one, unsup_f_two, sup_f, unsup_loss, sup_logits = self.model(x_w, x_s)
            return unsup_f_one, unsup_f_two, sup_f, unsup_loss, sup_logits
        else:
            raise NotImplementedError


    def _backward(self, loss=None, mode='general', **kwargs):
        self.backward_time += 1
        loss.backward()
        return

    def get_model_params(self):
        return {k: copy.deepcopy(val.cpu())
                for k, val in self.model.state_dict().items()}

    def _local_data_stats(self):
        self.cls_map = dict()
        self.total_cls_map = dict()
        self.cls_arr, self.cls_cnt_arr = np.unique(self.y_true_train, return_counts=True)
        #new add
        self.total_cls_arr,self.total_cls_cnt_arr = np.unique(self.y_train_total, return_counts=True)

        for i in range(len(self.cls_arr)):
            self.cls_map[self.cls_arr[i]] = np.where(self.y_true_train == self.cls_arr[i])[0]

        for i in range(len(self.total_cls_arr)):
            self.total_cls_map[self.total_cls_arr[i]] = np.where(self.y_train_total == self.total_cls_arr[i])[0]

    def _imperfect_data_environment_simulation(self):
        # partial_label
        # imperfect environment for label, including noisy label (nl) and partial label(pl)

        if 'nl' in self.args.imperfect_scenario.type:
            if 'symmetric' in self.args.imperfect_scenario.noise_type or 'pairflip' in self.args.imperfect_scenario.noise_type:
                for cls in self.cls_map.keys():
                    noise_num = int(len(self.cls_map[cls]) * self.args.imperfect_scenario.noise_ratio)
                    noise_index = np.random.choice(self.cls_map[cls], noise_num, replace=False)
                    # if self.args.imperfect_scenario.noise_inclient:
                    #     logging.info("Client {} simulation in_client noise".format(self.index))
                    # else:
                    #     logging.info("Client {} simulation general noise".format(self.index))
                    for i in range(len(noise_index)):
                        # logging.info('label noise use symmetric noise or pairflip noise')
                        true_label = self.y_true_train[noise_index[i]]
                        if len(self.cls_arr) > 1 and self.args.imperfect_scenario.noise_inclient:
                            self.y_train[noise_index[i]] = noisification_in_client(true_label=true_label,
                                                                                   cls_arr=self.cls_arr,
                                                                                   noise_type=self.args.imperfect_scenario.noise_type)
                        else:
                            self.y_train[noise_index[i]] = noisification(true_label=true_label,
                                                                         num_classes=self.args.datasets.num_classes,
                                                                         noise_type=self.args.imperfect_scenario.noise_type)

                for cls in self.total_cls_map.keys():
                    total_noise_num = int(len(self.total_cls_map[cls]) * self.args.imperfect_scenario.noise_ratio)
                    noise_index = np.random.choice(self.total_cls_map[cls], total_noise_num, replace=False)
                    for i in range(len(noise_index)):
                        true_label = self.y_train_total[noise_index[i]]
                        if len(self.total_cls_arr) > 1 and self.args.imperfect_scenario.noise_inclient:
                            self.y_train_total_noise[noise_index[i]] = noisification_in_client(true_label=true_label,
                                                                                   cls_arr=self.total_cls_arr,
                                                                                   noise_type=self.args.imperfect_scenario.noise_type)
                        else:
                            self.y_train_total_noise[noise_index[i]] = noisification(true_label=true_label,
                                                                         num_classes=self.args.datasets.num_classes,
                                                                         noise_type=self.args.imperfect_scenario.noise_type)

            elif 'random' in self.args.imperfect_scenario.noise_type and self.random_noise_level > 0:
                logging.info(
                    "Random Noise Simulation Process, this client {} has noise with level {}".format(self.index,
                                                                                                     self.random_noise_level))
                sample_idx = np.array(range(self.sample_num))
                prob = np.random.rand(self.sample_num)
                noisy_idx = np.where(prob <= self.random_noise_level)[0]
                self.y_train[sample_idx[noisy_idx]] = np.random.randint(0, self.args.datasets.num_classes,
                                                                        len(noisy_idx))

            noise_num = np.sum(self.y_train != self.y_true_train)
            noise_ratio = np.mean(self.y_train != self.y_true_train)
            self.local_real_noise_ratio = noise_ratio
            if 'random' in self.args.imperfect_scenario.noise_type:
                logging.info("Client %d, noise level: %.4f, real noise ratio: %.4f, real noise num: %.4f" % (
                    self.index, self.random_noise_level, self.local_real_noise_ratio, noise_num))
            else:
                logging.info("Client %d,  real noise ratio: %.4f, real noise num: %.4f" % (
                    self.index, noise_ratio, noise_num))

        if 'pl' in self.args.imperfect_scenario.type:
            logging.info('use partial label')

            self.ulb_ratio = self.args.imperfect_scenario.clients_ulb_ratio[self.index]
            self.ulb_num = int(self.ulb_ratio * self.sample_num)
            self.lb_num = self.sample_num - self.ulb_num

            ulb_index = np.random.choice(range(self.sample_num), self.ulb_num, replace=False)
            lb_index = np.delete(range(self.sample_num), ulb_index)

            self.X_train_lb = self.X_train[lb_index]
            self.X_train_ulb = self.X_train[ulb_index]

            # label noise version if 'nl' in self.args.imperfect_scenario.type or is the ground truth label only in 'pl' scenario
            self.y_train_lb = self.y_train[lb_index]
            # use this to contraste the difference betweent noised label acc and true label acc
            self.y_train_ulb = self.y_train[ulb_index]
            logging.info("Client {} simulation partial label with unlabel ratio {}: unlabel num: {}".format(self.index,
                                                                                                            self.ulb_ratio,
                                                                                                            len(self.y_train_ulb)))
            if 'nl' in self.args.imperfect_scenario.type:
                # just memory the ground truth label of data when the label have noise
                self.y_train_true_ulb = self.y_true_train[ulb_index]
                self.y_train_true_lb = self.y_true_train[lb_index]


        elif self.args.imperfect_scenario.type is None:
            pass

    def set_model_params(self, params):
        self.model.load_state_dict(params,strict=False)
        self.model.cpu()

    def set_model_params_wo_stat_info(self, params):
        for idx, val in enumerate(self.model.parameters()):
            val.data.copy_(params[idx])

    def filter_state_dict(self, state_dict, include_keys):

        if include_keys is None:
            return state_dict
        filtered = {k: v for k, v in state_dict.items()
                    if any(p in k for p in include_keys)}
        return filtered

    def client_update_local_info(self, download_info, stage="general"):
        if 'GLOBAL_MODEL_PARAM' in download_info:
            global_params = download_info['GLOBAL_MODEL_PARAM']

            if stage == "warm":
                include_keys = self.args.algorithms.warm_agg_include_keys
            elif stage == "finetune":
                include_keys = self.args.algorithms.finetune_agg_include_keys
            elif stage == "gengxin":
                include_keys = ["unsup_model"]
            else:
                include_keys = None  # 默认不过滤，更新全部

            # 过滤，只保留当前阶段需要更新的参数
            download_info['GLOBAL_MODEL_PARAM'] = self.filter_state_dict(global_params, include_keys)

            # 调试输出
            print(f"[{stage}]保留的keys:{include_keys} ,聚合参数 keys: {list(download_info['GLOBAL_MODEL_PARAM'].keys())}")

            logging.info("Client {} update local model with global model".format(self.index))
            self.global_model_params = copy.deepcopy(download_info['GLOBAL_MODEL_PARAM'])
            self.set_model_params(copy.deepcopy(download_info['GLOBAL_MODEL_PARAM']))
        else:
            raise ValueError('Client does not receive global model')

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


    def _build_optimizer(self,
                         params_to_optimizer,
                         lr=None,
                         weight_decay=None,
                         momentum=None,**kwargs):
        if lr is None:
            lr = self.args.optim.lr
        if weight_decay is None:
            weight_decay = self.args.optim.wd
        if momentum is None:
            momentum = self.args.optim.momentum

        if self.args.optim.optim_name == 'SGD':
            if momentum < 1.0:
                optimizer = torch.optim.SGD(params_to_optimizer,
                                            lr=lr,
                                            weight_decay=weight_decay,
                                            momentum=momentum,
                                            nesterov=self.args.optim.nesterov)
            else:
                optimizer = torch.optim.SGD(params_to_optimizer,
                                            lr=lr,
                                            weight_decay=weight_decay,
                                            nesterov=self.args.optim.nesterov)
        elif self.args.optim.optim_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                params_to_optimizer,
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        elif self.args.algorithms.algorithm_name == 'FedProx':
            pass
        else:
            NotImplementedError

        return optimizer


    def test(self,
             round_idx,
             testloader):
        self.model.eval()
        self.model.to(self.device)

        loss_avg = AverageMeter()
        acc_avg = AverageMeter()
        total_num = 0
        correct = 0

        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                bs = inputs.size(0)

                _, outputs = self._forward(inputs=inputs, mode='test')
                loss = criterion(outputs, targets)
                prec1, _, _ = accuracy(outputs.data, targets)
                correct_num, total_bs_num = top_one_accuracy(outputs.data, targets)

                correct += correct_num
                total_num += total_bs_num
                loss_avg.update(loss.item(), bs)
                acc_avg.update(prec1, bs)
        acc = 100. * correct / total_num

        scalar_dict = {'{role}:{index} old averager acc'.format(role=self.role, index=self.index): acc_avg.avg.item(),
                       '{role}:{index} new real acc'.format(role=self.role, index=self.index): acc,
                       '{role}:{index} loss'.format(role=self.role, index=self.index): loss_avg.avg,
                       'Test Round': round_idx}
        if self.args.record:
            record(record_tool=self.args.record_tool, scalar=scalar_dict, step=round_idx)

        return acc

    def train(self,
              round,
              train_epoch,
              dataloader,
              **kwargs):

        logging.info("-------------------local train-------------------")
        self.model.to(self.device)
        self.model.train()

        for epoch in range(train_epoch):
            logging.info('=> Training Epoch #%d, LR=%.4f' % (epoch, self.opt.param_groups[0]['lr']))
            metric_tracker = MetricTracker(device=self.device)
            loss_avgs = {loss_name: AverageMeter() for loss_name in self.loss_manager.loss_functions.keys()}

            for batch_idx, (X_train, _, y_train) in enumerate(dataloader):
                X_train, labels = X_train.to(self.device), y_train.to(self.device)
                bs = X_train.size(0)
                self.opt.zero_grad()
                feats, outputs = self._forward(inputs=X_train)
                total_loss, individual_losses = self.loss_manager.compute_loss(
                    feats=feats,
                    outputs=outputs,
                    labels=labels,
                    local_params=self.model,
                    global_params=self.global_model_params
                )
                self._backward(loss=total_loss)
                self.opt.step()

                metric_tracker.update(outputs=outputs, targets=labels, loss=total_loss)

                for loss_name, loss_value in individual_losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        loss_avgs[loss_name].update(loss_value.item(), bs)
                    else:
                        loss_avgs[loss_name].update(loss_value, bs)

            results = metric_tracker.compute()

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

            for loss_name, loss_avg in loss_avgs.items():
                scalar_dict['{role}:{index} {loss_name} loss'.format(role=self.role, index=self.index,
                                                                     loss_name=loss_name)] = loss_avg.avg
            print(scalar_dict)
            if self.args.record:
                record(record_tool=self.args.record_tool, scalar=scalar_dict, step=self.forward_time)

        self.model.cpu()

    def warm_train(self,
              round,
              train_epoch,
              dataloader,
              **kwargs):
        logging.info("-------------------warm local train-------------------")
        self.model.to(self.device)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        other_params_before = {name: param.clone() for name, param in self.model.named_parameters() if
                               'unsup_model' not in name}
        self.model.use_projector = True
        self.model.set_projector_trainable(True)
        for epoch in range(train_epoch):
            logging.info('=> Training Epoch #%d, LR=%.4f' % (epoch, self.opt.param_groups[0]['lr']))
            warm_up_loss_avg = AverageMeter()
            instanceDis_loss_avg = AverageMeter()
            vicreg_loss_avg = AverageMeter()
            sim_loss_avg= AverageMeter()
            var_loss_avg= AverageMeter()
            cov_loss_avg= AverageMeter()

            PR_ratio_avg=AverageMeter()
            EffRank_avg=AverageMeter()
            Top10_avg=AverageMeter()
            Cond_avg=AverageMeter()

            error_num=0
            for batch_idx, (X_train, X_aug_train, y_train, y_train_true,idx) in enumerate(dataloader):
                X_train, X_aug_train, labels, labels_true= X_train.to(self.device), X_aug_train.to(self.device) ,y_train.to(self.device) ,y_train_true.to(self.device)
                bs = X_train.size(0)
                error_num+= (labels != labels_true).sum()
                self.opt.zero_grad()
                unsup_f, unsup_logits = self.model(X_train, X_aug_train)

                if batch_idx % 10 == 0:
                    z1, z2 = torch.split(unsup_logits, bs, dim=0)
                    stats = eval_dimensional_collapse(torch.cat([z1, z2], dim=0))
                    # logging.info(f"[Epoch {epoch} | Batch {batch_idx}] "
                    #              f"PR={stats['participation_ratio']:.2f}, "
                    #              f"EffRank={stats['effective_rank']:.2f}, "
                    #              f"Top10%={stats['topk_ratio']:.2f}, "
                    #              f"Cond={stats['condition_number']:.1f}")
                    PR_ratio_avg.update(stats['participation_ratio'], 1)
                    EffRank_avg.update(stats['effective_rank'],1)
                    Top10_avg.update(stats['topk_ratio'],1)
                    Cond_avg.update(stats['condition_number'],1)

                logits, labels = info_nce_loss(unsup_logits, bs, self.device)
                instanceDis_loss = criterion(logits, labels)

                z1, z2 = torch.split(unsup_logits, bs, dim=0)
                vic_loss, loss_pack = vicreg_loss(z1, z2)
                loss=instanceDis_loss
                self._backward(loss=loss)
                self.opt.step()
                warm_up_loss_avg.update(loss.data.item(),bs)
                instanceDis_loss_avg.update(instanceDis_loss.data.item(), bs)
                vicreg_loss_avg.update(vic_loss.data.item(),bs)
                sim_loss_avg.update(loss_pack[0].data.item(),bs)
                var_loss_avg.update(loss_pack[1].data.item(),bs)
                cov_loss_avg.update(loss_pack[2].data.item(),bs)
        scalar_dict = {
            f'{self.role}:{self.index} instanceDis loss': instanceDis_loss_avg.avg,
            f'{self.role}:{self.index} vicreg loss': vicreg_loss_avg.avg,
            f'{self.role}:{self.index} warm up loss': warm_up_loss_avg.avg,
            f'{self.role}:{self.index} sim loss': sim_loss_avg.avg,
            f'{self.role}:{self.index} var up loss': var_loss_avg.avg,
            f'{self.role}:{self.index} cov up loss': cov_loss_avg.avg,
            f'{self.role}:{self.index} PR_ratio': PR_ratio_avg.avg,
            f'{self.role}:{self.index} EffRank_avg': EffRank_avg.avg,
            f'{self.role}:{self.index} Top10_avg': Top10_avg.avg,
            f'{self.role}:{self.index} Cond_avg': Cond_avg.avg,
        }

        if self.args.record:
            record(record_tool=self.args.record_tool, scalar=scalar_dict, step=round)

        print('SemiFed:{}_{}_unlabeled_instanceDis_loss{}'.format(self.role, self.index,instanceDis_loss_avg.avg))
        self.model.cpu()

    def entropy(self,p, eps=1e-12):
        return -(p.clamp_min(eps) * (p.clamp_min(eps)).log()).sum(dim=1)  # [B]

    def js_divergence(self,p, q, eps=1e-12):
        m = 0.5 * (p + q)
        kl_pm = (p.clamp_min(eps) * (p.clamp_min(eps) / m.clamp_min(eps)).log()).sum(dim=1)
        kl_qm = (q.clamp_min(eps) * (q.clamp_min(eps) / m.clamp_min(eps)).log()).sum(dim=1)
        return 0.5 * (kl_pm + kl_qm)  # [B]

    def batch_quantile(self, t: torch.Tensor, q: float):
        k = max(1, int(q * t.numel()))
        vals, _ = torch.topk(t, k, largest=False)
        return vals.max()

    def finetune_train_VMF_ori(self,
                       round,
                       train_epoch,
                       dataloader,
                       **kwargs):


        logging.info("-------------------Representation Geometry Priority Principle for training-------------------")
        device = self.device
        self.model.to(device)
        self.model.train()
        self.model.unsup_model.train()
        self.model.use_projector = False
        for m in self.model.unsup_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train()
        ce = nn.CrossEntropyLoss(reduction='none')  # per-sample
        sce = SCELoss(device=device)  #  per-sample
        nll=nn.NLLLoss(reduction='none')
        self.model.set_projector_trainable(False)
        enc_params = [p for n, p in self.model.unsup_model.named_parameters() if not n.startswith('projector.')]
        for p in self.model.unsup_model.projector.parameters():
            p.requires_grad = False
        for p in self.model.classifier_head.parameters(): p.requires_grad = True

        optimizer = torch.optim.SGD([
            {'params': enc_params, 'weight_decay': 5e-4},
            {'params': self.model.classifier_head.parameters(), 'weight_decay': 5e-4},
            {'params': [self.model.T], 'weight_decay': 0.0}
        ], lr=0.01, momentum=0.9)

        # optimizer = torch.optim.SGD([
        #     {'params': self.model.unsup_model.parameters(), 'weight_decay': 5e-4},
        #     {'params': self.model.classifier_head.parameters(), 'weight_decay': 5e-4},
        #     {'params': [self.model.T], 'weight_decay': 0.0}
        # ], lr=0.01, momentum=0.9)

        # optimizer = torch.optim.SGD([
        #     {'params': self.model.unsup_model.parameters(), 'lr': 0.005, 'weight_decay': 5e-4},
        #     {'params': self.model.classifier_head.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
        #     {'params': [self.model.T], 'lr': 0.01, 'weight_decay': 0.0}
        # ], momentum=0.9)

        ce_meter = AverageMeter()  #
        clean_loss_meter = AverageMeter()

        pclean_meter = AverageMeter()
        total_clean_count = 0
        total_noisy_count = 0

        def l2n(x, dim=1, eps=1e-8):
            return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

        def Ainv_from_R(R, d):
            R = torch.clamp(R, 1e-6, 1 - 1e-6)
            return R * (d - R ** 2) / (1 - R ** 2)

        def vmf_logits(z, MU, KAPPA):
            cos = torch.matmul(z, MU.t())  # [B, G]
            return cos * KAPPA.unsqueeze(0)  # [B, G]

        num_samples = getattr(self, 'sample_num', 0) or len(self.y_true_train)
        num_classes = self.args.datasets.num_classes

        if not hasattr(self, '_sbrf_d'):
            self._sbrf_d = self.model.unsup_model.feature_dim
        d = self._sbrf_d


        L = getattr(self.args.algorithms, 'trunc_L', self.args.datasets.num_classes)


        if not hasattr(self, '_sbrf_mix'):
            MU = torch.randn(L, d, device=device)
            MU = l2n(MU, dim=1)
            KAPPA = torch.full((L,), 5.0, device=device)
            BETA = torch.full((L,), 1.0 / L, device=device)
            PI0 = torch.tensor(0.05, device=device)


            M_map = torch.full((num_classes, L), 1.0 / num_classes, device=device)

            alpha_dp = torch.tensor(getattr(self.args.algorithms, 'dp_alpha', 5.0), device=device)
            alpha_M = torch.tensor(getattr(self.args.algorithms, 'dirichlet_smooth', 0.5),
                                   device=device)


            buf = {
                'sum_vec': torch.zeros(L, d, device=device),
                'count': torch.zeros(L, device=device),
                'bg_count': torch.tensor(0.0, device=device),
                'N_cg': torch.zeros(num_classes, L, device=device),  # 估计 M 的软计数
            }


            self._sbrf_mix = {
                'MU': MU, 'KAPPA': KAPPA, 'BETA': BETA, 'PI0': PI0,
                'M': M_map, 'alpha_dp': alpha_dp, 'alpha_M': alpha_M,
                'buf': buf
            }

        mix = self._sbrf_mix
        MU, KAPPA, BETA, PI0 = mix['MU'], mix['KAPPA'], mix['BETA'], mix['PI0']
        M_map, alpha_dp, alpha_M = mix['M'], mix['alpha_dp'], mix['alpha_M']
        buf = mix['buf']

        # Lyapunov 温度 / 窗口（尺度参数，不是阈值）
        # beta_temp = getattr(self.args.algorithms, 'lyap_beta', 5.0)
        # lyap_momentum = getattr(self.args.algorithms, 'lyap_momentum', 0.6)  # 对 λ 的 EMA
        proto_momentum = getattr(self.args.algorithms, 'proto_momentum', 0.1)  # 对 MU/M 的 EMA
        # ----------------------- 训练循环 -----------------------
        for epoch in range(train_epoch):
            logging.info('=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
            offset = 0
            C = self.model.T.size(0)
            T_true = torch.zeros(C, C, dtype=torch.float32, device=device)

            for batch_idx, (X_train, X_aug, y_obs, y_true,idx) in enumerate(dataloader):
                X_train = X_train.to(device)
                X_aug = X_aug.to(device)
                y_obs = y_obs.to(device)
                bs = X_train.size(0)
                if bs==1:
                    continue
                idx_tensor = torch.arange(offset, offset + bs, device=device)
                offset += bs
                optimizer.zero_grad()

                h1 = self.model.unsup_model.encoder_forward(X_train)  # [B, 512]
                h2 = self.model.unsup_model.encoder_forward(X_aug)
                z1 = l2n(h1, dim=1)
                z2 = l2n(h2, dim=1)
                z = z1
                logits = self.model.classifier_head(h1)
                p = torch.softmax(logits, dim=1)


                logit_g = vmf_logits(z, MU, KAPPA) + torch.log(BETA + 1e-12)  # [B, L]
                logit_bg = torch.full((bs, 1), torch.log(PI0 + 1e-12), device=device)  # 背景
                logits_mix = torch.cat([logit_bg, logit_g], dim=1)  # [B, 1+L]
                post = torch.softmax(logits_mix/2.0, dim=1)  # [B, 1+L]
                gamma_bg = post[:, :1]  # [B, 1]
                gamma = post[:, 1:]  # [B, L]  几何责任

                # ------- (B) 增强稳定性：观测精度 r_i （无阈值） -------
                R = ((z1 + z2) / 2.0).norm(p=2, dim=1).clamp(1e-6, 1 - 1e-6)
                kappa_aug = Ainv_from_R(R, d)  # [B]
                with torch.no_grad():
                    if not hasattr(self, '_sbrf_kappa_median'):
                        self._sbrf_kappa_median = torch.median(kappa_aug).detach()
                    else:
                        self._sbrf_kappa_median = 0.9 * self._sbrf_kappa_median + 0.1 * torch.median(kappa_aug).detach()
                r_i = torch.clamp(kappa_aug / (self._sbrf_kappa_median + 1e-8), 0.2, 2.0)  # [B]


                logit_g_tilde = (vmf_logits(z, MU, KAPPA) * r_i.unsqueeze(1)) + torch.log(BETA + 1e-12)
                logits_mix_tilde = torch.cat([logit_bg, logit_g_tilde], dim=1)
                post_tilde = torch.softmax(logits_mix_tilde/2.0, dim=1)
                gamma_bg_tilde = post_tilde[:, :1]  # [B,1]
                gamma_tilde = post_tilde[:, 1:]  # [B,L]

                M_y = M_map[y_obs]  # [B, L]
                q_tilde_y = torch.sum(M_y * gamma_tilde, dim=1)  # [B] ∈ (0,1)

                one_minus_My = 1.0 - M_y  # [B, L]
                rest_sum = torch.sum(one_minus_My * gamma_tilde, dim=1)  # [B]
                denom =q_tilde_y + rest_sum + gamma_bg_tilde.squeeze(1) + 1e-12
                P_clean = (q_tilde_y) / denom  # [B] ∈ (0,1)
                pclean_meter.update(P_clean.mean().item(), bs)

                from sklearn.mixture import GaussianMixture
                w_noisy = (1.0 - P_clean).detach()
                w_np = w_noisy.detach().cpu().view(-1, 1).numpy()
                if not hasattr(self, "_gmm_noise_split"):
                    self._gmm_noise_split = GaussianMixture(
                        n_components=2, covariance_type="full", reg_covar=1e-6, max_iter=100, random_state=0
                    )
                gmm = self._gmm_noise_split
                gmm.fit(w_np)
                probs = gmm.predict_proba(w_np)  # [B, 2]
                means = gmm.means_.flatten()
                clean_comp = np.argmin(means)
                P_clean_gmm = torch.from_numpy(probs[:, clean_comp]).to(self.device).float().detach()


                clean_mask = (P_clean_gmm >= self.args.algorithms.gmm_thread)
                noisy_mask = ~clean_mask
                m_clean = clean_mask.float().detach()
                m_noisy = noisy_mask.float().detach()

                num_clean = int(m_clean.sum().item())
                num_noisy = int(m_noisy.sum().item())

                total_clean_count += num_clean
                total_noisy_count += num_noisy
                sce_per = sce.forward(logits, y_obs).mean()  # [B]
                loss_clean = sce_per


                T_dev = self.model.T
                p_noisy = torch.matmul(p, T_dev)  # [B, C]
                loss_noisy_per = nll(torch.log(p_noisy + 1e-12), y_obs)  # [B]
                loss_noisy = (loss_noisy_per * m_noisy).sum() / (m_noisy.sum() + 1e-8)

                loss = (self.args.algorithms.loss_weight.clean_loss * loss_clean
                        + self.args.algorithms.loss_weight.noise_loss * loss_noisy)
                loss.backward()
                optimizer.step()


                with torch.no_grad():
                    self.model.T.data = self.model.T.data.clamp_min(1e-6)
                    self.model.T.data /= self.model.T.data.sum(dim=1, keepdim=True)

                with torch.no_grad():
                    ent = -(gamma_tilde * (gamma_tilde + 1e-12).log()).sum(dim=1).mean()
                    if (batch_idx + 1) % 10 == 0:
                        print(f"[diag] entropy(gamma_tilde)={ent.item():.3f}, "
                              f"beta min/max={BETA.min():.4f}/{BETA.max():.4f}, "
                              f"kappa min/med/max={KAPPA.min():.2f}/{KAPPA.median():.2f}/{KAPPA.max():.2f}, "
                              f"pi0={PI0.item():.4f}")


                    buf['sum_vec'] += torch.matmul(gamma_tilde.t(), z)  # [L,d] += [L,B]@[B,d]
                    buf['count'] += torch.sum(gamma_tilde, dim=0)  # [L]
                    buf['bg_count'] += torch.sum(gamma_bg_tilde)


                    if m_clean.sum() > 0:
                        N_cg_batch = torch.zeros_like(M_map)  # [C,L]
                        gamma_clean = gamma_tilde * m_clean.unsqueeze(1)
                        N_cg_batch.index_add_(0, y_obs, gamma_clean)
                        buf['N_cg'] += N_cg_batch


                    if (batch_idx + 1) % getattr(self.args.algorithms, 'mix_update_interval', 1) == 0:
                        # MU
                        new_mu = l2n(buf['sum_vec'].clone(), dim=1)  # [L,d]
                        MU.mul_(1 - proto_momentum).add_(proto_momentum * new_mu)
                        MU.copy_(l2n(MU, dim=1))


                        R_g = (buf['sum_vec'].norm(p=2, dim=1) / (buf['count'] + 1e-8)).clamp(1e-6, 1 - 1e-6)
                        new_kappa = Ainv_from_R(R_g, d)

                        new_kappa = new_kappa.clamp(0.5, 100.0)
                        KAPPA.mul_(1 - proto_momentum).add_(proto_momentum * new_kappa)

                        counts = buf['count'].clone()
                        total = counts.sum() + buf['bg_count']
                        prior = alpha_dp / L
                        new_beta = (counts + prior) / (counts.sum() + prior * L + 1e-12)
                        new_pi0 = (buf['bg_count'] + (alpha_dp * 0.1)) / (total + alpha_dp * 0.1 + 1e-12)
                        BETA.mul_(1 - proto_momentum).add_(proto_momentum * new_beta)
                        PI0.mul_(1 - proto_momentum).add_(proto_momentum * new_pi0)
                        PI0.clamp_(1e-3, 0.2)

                        N_cg = buf['N_cg']
                        if N_cg.sum().item() > 0:
                            M_new = (N_cg + alpha_M) / (torch.sum(N_cg + alpha_M, dim=1, keepdim=True) + 1e-12)
                            M_map.mul_(1 - proto_momentum).add_(proto_momentum * M_new)
                            M_map /= (M_map.sum(dim=1, keepdim=True) + 1e-12)

                        # 清空本轮统计
                        buf['sum_vec'].zero_()
                        buf['count'].zero_()
                        buf['bg_count'].zero_()
                        buf['N_cg'].zero_()

                    for i in range(bs):
                        T_true[y_true[i], y_obs[i]] += 1

                # ---- 记录 loss meter ----
                clean_loss_meter.update(loss_clean.item(), bs)
                ce_meter.update(loss.item(), bs)

            with torch.no_grad():
                self.model.T.data.clamp_(min=1e-6)
                self.model.T.data /= self.model.T.data.sum(dim=1, keepdim=True)

        print("map{}".format(M_map))
        scalar_dict = {
            f'{self.role}:{self.index} finetune loss': ce_meter.avg,
            f'{self.role}:{self.index} clean loss': clean_loss_meter.avg        }
        if self.args.record:
            record(record_tool=self.args.record_tool, scalar=scalar_dict, step=round)

        self._sbrf_mix = {
            'MU': MU.detach(), 'KAPPA': KAPPA.detach(), 'BETA': BETA.detach(), 'PI0': PI0.detach(),
            'M': M_map.detach(), 'alpha_dp': alpha_dp, 'alpha_M': alpha_M,
            'buf': buf
        }
        self.model.cpu()



    def local_train(self, round_idx, **kwargs):
        self.train(round_idx, self.args.algorithms.local_epochs, self.local_train_lb_dl)

    def local_warm_train(self, round_idx, **kwargs):
        self.warm_train(round_idx, self.args.algorithms.local_epochs, self.local_train_lb_dl)

    def local_finetune_train(self, round_idx, **kwargs):
        self.finetune_train_VMF_ori(round_idx, self.args.algorithms.local_epochs, self.local_train_lb_dl)

    def upload_info_account(self, upload_info):
        # compute the communication cost
        total_params = 0
        for info_key in upload_info.keys():
            if 'PARAM' in info_key:
                total_params += sum(upload_info[info_key][key].numel() for key in upload_info[info_key].keys())
        logging.info("The total information size is {} sent to server of client {}".format(total_params, self.index))
        return total_params

    def upload_info(self):
        # compute the communication cost
        upload_info = {"ROLE": self.role,
                       "SAMPLE_NUM": self.sample_num}
        upload_info["LOCAL_MODEL_PARAM"] = self.get_model_params()

        upload_info['UPLOAD_AMOUNT'] = self.upload_info_account(upload_info)
        return upload_info

    def get_index(self):
        return self.index

    def update_lr(self, lr):
        self.opt.param_groups[0]['lr'] = lr

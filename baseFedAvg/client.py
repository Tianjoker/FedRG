import copy
import logging
import math
from collections import OrderedDict

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import cv2
from model.model_manager import get_model

from model.ResNet import ResNet18, ResNet50, ResNet34
from utils.averager import AverageMeter
from utils.metric import (
    accuracy,
    top_one_accuracy
)
from utils.MetricTracker import MetricTracker
from data_preprocessing.personalized_dataset import (
    Dataset_Normal,
    Dataset_WeakStrong
)
from data_preprocessing.utils import record_y_distribution
from data_preprocessing.own_transforms import get_stransform
from utils.record import record
from utils.imp_utils import (
    noisification,
    noisification_in_client
)
from utils.utils import proximal_loss
from loss.loss_manager import LossManager


class Client(object):

    def __init__(self,
                 args,
                 device,
                 index,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 **kwargs) -> None:

        self.args = args
        self.device = device
        self.index = index

        self.X_train = X_train
        self.y_true_train = y_train
        self.y_train = copy.deepcopy(self.y_true_train)
        self.X_test = X_test
        self.y_test = y_test
        if 'random_nozise_level' in kwargs:
            self.random_noise_level = kwargs['random_noise_level']

        self.role = 'client'
        self.sample_num = len(self.y_true_train)
        self._local_data_stats()
        self._initial_setup()
        self._imperfect_data_environment_simulation()
        self.forward_time = 0
        self.backward_time = 0
        self._build_data_loader()
        self.loss_manager = LossManager(self.args, self.device)
        self.model_delta = OrderedDict()


    def _initial_setup(self):
        self.model = self._build_model()
        self.opt = self._build_optimizer(self.model.parameters())

    def _build_data_loader(self):
        train_trans = get_stransform(self.args.datasets.dataset_name, train=True)
        if 'nl' in self.args.imperfect_scenario.type and 'pl' not in self.args.imperfect_scenario.type:
            self.local_train_lb_ds = Dataset_WeakStrong(data=self.X_train,
                                                        targets=self.y_train,
                                                        ulb=False,
                                                        dataset=self.args.datasets.dataset_name,
                                                        transform=train_trans)
            # self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
            #                                                      batch_size=self.args.bs, shuffle=True,
            #                                                      drop_last=False)
            self.local_train_lb_dl = torch.utils.data.DataLoader(
                dataset=self.local_train_lb_ds, batch_size=self.args.bs, shuffle=True, drop_last=False, num_workers=1,
                pin_memory=True)
        elif 'pl' in self.args.imperfect_scenario.type:
            if self.lb_num > 0:
                self.local_train_lb_ds = Dataset_WeakStrong(data=self.X_train_lb,
                                                            targets=self.y_train_lb,
                                                            ulb=False,
                                                            dataset=self.args.datasets.dataset_name,
                                                            transform=train_trans)
                self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
                                                                     batch_size=self.args.bs, shuffle=True,
                                                                     drop_last=False)

            if self.ulb_num > 0:
                self.local_train_ulb_ds = Dataset_WeakStrong(data=self.X_train_ulb,
                                                             targets=self.y_train_ulb,
                                                             ulb=True,
                                                             dataset=self.args.datasets.dataset_name,
                                                             transform=train_trans)
                self.local_train_ulb_dl = torch.utils.data.DataLoader(dataset=self.local_train_ulb_ds,
                                                                      batch_size=self.args.bs, shuffle=True,
                                                                      drop_last=False)
        elif 'general' in self.args.imperfect_scenario.type:
            self.local_train_lb_ds = Dataset_WeakStrong(data=self.X_train,
                                                        targets=self.y_true_train,
                                                        ulb=False,
                                                        dataset=self.args.datasets.dataset_name,
                                                        transform=train_trans)
            self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
                                                                 batch_size=self.args.bs, shuffle=True,
                                                                 drop_last=False)

    def _forward(self, inputs=None, mode='general', **kwargs):
        feat, outputs = self.model(inputs)
        if mode == 'general':
            self.forward_time += 1
        elif mode == 'test':
            pass
        else:
            raise NotImplementedError
        return feat, outputs

    def _backward(self, loss=None, mode='general', **kwargs):
        self.backward_time += 1
        loss.backward()
        return

    def get_data_distribution(self):
        return record_y_distribution(self.y_true_train)

    def get_model_params(self):
        return {k: copy.deepcopy(val.cpu())
                for k, val in self.model.state_dict().items()}

    def get_model_params_wo_stat_info(self):
        return [copy.deepcopy(val.cpu())
                for val in self.model.parameters()]

    def _local_data_stats(self):

        self.cls_map = dict()
        self.cls_arr, self.cls_cnt_arr = np.unique(self.y_true_train, return_counts=True)
        for i in range(len(self.cls_arr)):
            self.cls_map[self.cls_arr[i]] = np.where(self.y_true_train == self.cls_arr[i])[0]

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

    def get_labeled_data_distribution(self):
        if self.lb_num > 0:
            return record_y_distribution(self.y_train_lb)
        else:
            return None

    def set_model_params(self, params):
        self.model.load_state_dict(params)

    def set_model_params_wo_stat_info(self, params):
        for idx, val in enumerate(self.model.parameters()):
            val.data.copy_(params[idx])

    def client_update_local_info(self, download_info):

        if 'GLOBAL_MODEL_PARAM' in download_info:
            logging.info("Client {} update local model with global model".format(self.index))

            # FedInit use Relaxed Initialization
            init_method = self.args.algorithms.ini_method if hasattr(self.args.algorithms,
                                                                     'ini_method') else 'general'

            self.set_model_params(copy.deepcopy(download_info['GLOBAL_MODEL_PARAM']))

            self.global_model_params = copy.deepcopy(download_info['GLOBAL_MODEL_PARAM'])
        else:
            raise ValueError('Client does not receive global model')

        if 'GLOBAL_UPDATE_PARAM' in download_info:
            logging.info("Client {} update global update model locally".format(self.index))
            self.global_update = copy.deepcopy(download_info['GLOBAL_UPDATE_PARAM'])
        else:
            raise ValueError('Client does not receive global update model')
        if 'GLOBAL_EMA_UPDATE_PARAM' in download_info:
            logging.info("Client {} update global ema update model locally".format(self.index))
            self.global_ema_update = copy.deepcopy(download_info['GLOBAL_EMA_UPDATE_PARAM'])
        else:
            raise ValueError('Client does not receive global ema update model')


    def _build_model(self, model_name=None, num_classes=None, input_channels=None):
        return get_model(self.args, model_name, num_classes, input_channels)


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

    def get_train_batch_data(self, train_local):
        try:
            train_batch_data = next(self.train_local_iter)
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
        except:
            self.train_local_iter = iter(train_local)
            train_batch_data = next(self.train_local_iter)
        return train_batch_data

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
            # model_before_train = copy.deepcopy(self.model.state_dict())

            metric_tracker = MetricTracker(device=self.device)

            loss_avgs = {loss_name: AverageMeter() for loss_name in self.loss_manager.loss_functions.keys()}

            for batch_idx, (X_train, _, y_train) in enumerate(dataloader):
                X_train, labels = X_train.to(self.device,non_blocking=True), y_train.to(self.device,non_blocking=True)
                # print("first batch shape:", X_train.shape);
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
                if self.args.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.args.max_norm)
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
        self.post_train()


    def local_train(self, round_idx, **kwargs):
        self.train(round_idx, self.args.algorithms.local_epochs, self.local_train_lb_dl)

    def train_using_lb_data(self, round_idx, **kwargs):
        self.train(round_idx, self.args.algorithms.local_epochs, self.local_train_lb_dl)

    def has_labeled_data(self):
        return True if self.lb_num > 0 else False

    def upload_info_account(self, upload_info):
        # compute the communication cost
        total_params = 0
        for info_key in upload_info.keys():
            if 'PARAM' in info_key:
                total_params += sum(upload_info[info_key][key].numel() for key in upload_info[info_key].keys())
        logging.info("The total information size is {} sent to server of client {}".format(total_params, self.index))
        return total_params
    #
    # def upload_info(self):
    #     # compute the communication cost
    #     upload_info = {"ROLE": self.role,
    #                    "SAMPLE_NUM": self.sample_num}
    #     upload_info["LOCAL_MODEL_PARAM"] = self.get_model_params()
    #
    #     upload_info['UPLOAD_AMOUNT'] = self.upload_info_account(upload_info)
    #     return upload_info
    def post_train(self):
        self.model_delta.clear()
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                self.model_delta[name] = param.data - self.global_model_params[name].data

    def upload_info(self):
        upload_info = {"ROLE": self.role,
                    "LOCAL_MODEL_DELTA_PARAM": copy.deepcopy(self.model_delta),
                    "SAMPLE_NUM": self.sample_num,
                    "LOCAL_MODEL_PARAM": self.get_model_params()}
        upload_info['UPLOAD_AMOUNT'] = self.upload_info_account(upload_info)
        return upload_info

    def get_index(self):
        return self.index

    def update_lr(self, lr):
        self.opt.param_groups[0]['lr'] = lr

    def labeded_data_training(self, round, train_epoch, **kwargs):
        logging.info("-------------------trainining only use labeled data-------------------")
        self.model.to(self.device)
        self.model.train()

        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(train_epoch):
            logging.info('=> Training Epoch #%d, LR=%.4f' % (epoch, self.opt.param_groups[0]['lr']))
            train_acc_avg = AverageMeter()
            train_loss_avg = AverageMeter()
            correct = 0
            total_num = 0
            for batch_idx, (X_train, _, y_train) in enumerate(self.local_train_lb_dl):

                X_train, labels = X_train.to(self.device), y_train.to(self.device)
                bs = X_train.size(0)
                self.opt.zero_grad()

                feats, outputs = self._forward(inputs=X_train)

                loss = criterion(outputs, labels)

                self._backward(loss=loss)

                if self.args.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.args.max_norm)

                self.opt.step()

                prec1, _, _ = accuracy(outputs.data, labels)
                correct_num, total_bs_num = top_one_accuracy(outputs.data, labels)
                correct += correct_num
                total_num += total_bs_num

                train_loss_avg.update(loss.item(), bs)
                train_acc_avg.update(prec1, bs)
            acc = 100. * correct / total_num

            scalar_dict = {
                '{role}:{index} old averager acc'.format(role=self.role, index=self.index): train_acc_avg.avg,
                '{role}:{index} new real acc'.format(role=self.role, index=self.index): acc,
                '{role}:{index} loss'.format(role=self.role, index=self.index): train_loss_avg.avg}
            if self.args.record:
                record(record_tool=self.args.record_tool, scalar=scalar_dict, step=round)

        self.model.cpu()

    def calculate_pert_grad(self, pert_grad, grad_after_pert):
        pert_grad = pert_grad.to(self.device).view(-1)
        grad_after_pert = grad_after_pert.to(self.device).view(-1)
        cos_sim = torch.nn.functional.cosine_similarity(pert_grad, grad_after_pert, dim=0)
        angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180 / torch.pi
        gv = grad_after_pert - (torch.norm(grad_after_pert) * cos_sim * pert_grad / torch.norm(pert_grad))

        return cos_sim, angle, torch.norm(gv)

    def calculate_local_update_direction(self, model_before_train, model_after_train):
        local_update_direction = {}
        for key in model_before_train.keys():
            local_update_direction[key] = model_after_train[key] - model_before_train[key]
        return local_update_direction
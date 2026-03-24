import logging
import numpy as np
import pickle
import copy
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
from torchvision.datasets import (
     CIFAR10, 
     CIFAR100, 
     SVHN, 
     FashionMNIST
     )
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from data_preprocessing.utils import (
    partition_data,
    record_y_distribution,
    classify_label,
    train_long_tail
    )

full_data_obj_dict = {
        "CIFAR10": CIFAR10,
        "CIFAR100": CIFAR100,
        "SVHN": SVHN,
        "FMNIST": FashionMNIST,
}


def long_tail_simulation(X_train, y_train, num_classes, imb_factor):
    list_label2indices = classify_label(y_train, num_classes)
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), num_classes,
                                                    imb_factor=imb_factor, imb_type='exp')
    idx = list(itertools.chain(*list_label2indices_train_new))
    return X_train[idx], y_train[idx]


def load_full_data(dataset,  
                   datadir):
        if dataset == "MCND":
            train_image_paths, y_train_np, test_image_paths, y_test_np = load_MCND_data(datadir)
            try:
                y_dis = record_y_distribution(y_train_np)
                logging.info("The total label distribution of MCND dataset loaded in full:")
                logging.info(y_dis)
            except NameError:
                pass
            return train_image_paths, y_train_np, test_image_paths, y_test_np

        elif dataset == "SVHN":
            train_ds = full_data_obj_dict[dataset](datadir,  "train", download=True, transform=transforms.Compose([transforms.ToTensor()]), target_transform=None)
            test_ds = full_data_obj_dict[dataset](datadir,  "test", download=True, transform=transforms.Compose([transforms.ToTensor()]), target_transform=None)
        else:
            train_ds = full_data_obj_dict[dataset](datadir,  train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            test_ds = full_data_obj_dict[dataset](datadir,  train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

        X_train = train_ds.data
        X_test = test_ds.data

        if dataset in ["fmnist"]:
            y_train = train_ds.targets.data
            y_test = test_ds.targets.data
        elif dataset in ["SVHN"]:
            y_train = train_ds.labels
            y_test = test_ds.labels
        else:
            y_train = train_ds.targets
            y_test = test_ds.targets
        
        y_train_np = np.array(y_train)
        y_dis = record_y_distribution(y_train_np)
        y_test_np = np.array(y_test)

        return X_train, y_train_np, X_test, y_test_np

def split_data_into_clients(dataset,
                            datadir,
                            client_number, 
                            partition_method, 
                            long_tail=False,
                            class_num=10,
                            partition_alpha=None,
                            seed=0,
                            **kwargs):
    if dataset in ['CIFAR100', 'CIFAR10', 'SVHN']:
        data_split_save_dict={}
        X_train_total, y_train_total, X_test, y_test_np = load_full_data(dataset, datadir)
        if long_tail:
            X_train_total, y_train_total = long_tail_simulation(X_train_total, y_train_total, class_num, imb_factor=kwargs['long_tail_imb_factor'])
        y_dis = record_y_distribution(y_train_total)
        logging.info("The total label distribution of simulated dataset")
        logging.info(y_dis)
        X_test = np.array(X_test)

        client_dataidx_map, traindata_cls_counts = partition_data(y_train_total, 
                                                                  client_number, 
                                                                  partition_method, 
                                                                  class_num=class_num, 
                                                                  partition_alpha=partition_alpha,
                                                                  seed=seed)
        client_X_train_dict = {}
        client_y_train_dict = {}
        client_sample_dict = {}
        for client_num in client_dataidx_map.keys():
            client_X_train_dict[client_num] = np.array([X_train_total[i] for i in client_dataidx_map[client_num]])
            client_y_train_dict[client_num] = np.array([y_train_total[i] for i in client_dataidx_map[client_num]])
            client_sample_dict[client_num] = len(client_y_train_dict[client_num])
        data_split_save_dict['client_unsup_map'] = client_dataidx_map

        # with open("{}_{}_{}_{}_{}.pkl".format(dataset,partition_method,partition_alpha,client_number,seed), "wb") as file:
        #     pickle.dump(data_split_save_dict, file)
        logging.info('Data statistics' )
        for client_num in traindata_cls_counts.keys():
            logging.info('Client {}, data count = {}, detial class num = {}'.format(client_num, client_sample_dict[client_num], traindata_cls_counts[client_num]))
        return client_X_train_dict, client_y_train_dict, X_test, y_test_np, X_train_total, y_train_total
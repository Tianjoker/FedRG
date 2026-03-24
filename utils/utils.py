import copy
import numpy as np
import random
import torch 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
import wandb
from data_preprocessing.own_transforms import get_stransform
from data_preprocessing.personalized_dataset import Dataset_Relabel


def proximal_loss(global_params, local_model, device):
    proximal_term = 0.0
    for name, local_param in local_model.named_parameters():
        global_param = global_params[name].to(device).to(torch.float32)
        proximal_term += (global_param - local_param).norm(2)
    return proximal_term

def compare_state_dicts(sd1, sd2):
    for key in sd1:
        if key not in sd2:
            return False
        if not torch.equal(copy.deepcopy(sd1[key]).cpu(), copy.deepcopy(sd2[key]).cpu(),):
            return False
    for key in sd2:
        if key not in sd1:
            return False
    return True

def pseudo_label_batch_wise(model, data, device):
    pass


def compute_relabel_class_wise_acc(num_classes, relabel_labels, relabel_gt_labels):
    """
    Args:
        num_classes (`int`):
             the number of classes
        relabel_labels (`np.arrat/torch.Tensor` of shape `(num_samples,)`):
            all relabeled data samples
        relabel_gt_labels (`np.arrat/torch.Tensor` of shape `(num_samples,)`):
            its corresponding ground truth label
    Return:
        class_acc (`np.arrat/torch.Tensor` of shape `(num_samples,)`)
    """
    sum_correct = (relabel_labels == relabel_gt_labels).sum().item()
    acc = sum_correct / len(relabel_labels)
    class_acc = { cls:0 for cls in range(num_classes)}
    for idx in range(len(relabel_labels)):
        if relabel_labels[idx] == relabel_gt_labels[idx]:
            class_acc[relabel_labels[idx]] += 1
        
    return acc, class_acc

def pseudo_label_dataset_wise(ulb_data, 
                              gt_labels, 
                              model, 
                              dataset_name, 
                              device, 
                              num_classes=10, 
                              threshold_type='all', 
                              threshold=0.9):
    """
    Args:
        ulb_data (`torch.Tensor` of shape `(num_samples, C, H, W)`):
            the unlabeled data
        gt_labels (`torch.Tensor` of shape `(num_samples,)`):
            the ground truth labels of the unlabeled data
        model (`torch.nn.Module`):
            the model to be used
        dataset_name (`str`):
            the name of the dataset
        device (`torch.device`):
            the device to be used
        num_classes (`int`):
            the number of classes
        threshold_type (`str`):
            the type of threshold
        threshold (`float`):
            the threshold
    Return:
        relabeled_data (`torch.Tensor` of shape `(num_samples, C, H, W)`):
            the relabeled data
        relabeled_labels (`torch.Tensor` of shape `(num_samples,)`):
            the relabeled labels
        acc (`float`):
            the accuracy
        class_acc (`dict`):
            the class accuracy
    """
    train_trans = get_stransform(dataset_name, train=True)
    label_ds = Dataset_Relabel(data=ulb_data,
                                  targets=gt_labels,
                                  ulb=False,
                                  dataset=dataset_name,
                                  transform=train_trans)
    label_dl = torch.utils.data.DataLoader(dataset=label_ds,
                                           batch_size=64, shuffle=False,
                                          drop_last=False)

    model.eval()
    model.to(device)

    collect_confident_data = []
    collect_confident_labels = []
    collect_gt_confident_labels = []
    
    if threshold_type == 'all':
        with torch.no_grad():
            for batch_idx, (ori_data, weak_data, labels) in enumerate(label_dl):
                weak_data = weak_data.to(device)
                logits = model(weak_data)
                logits = logits.detach()
                probs = F.softmax(logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                confident_idx = torch.where(max_probs >= threshold)[0]
                
                confident_data = ori_data[confident_idx]
                gt_confident_labels = labels[confident_idx]
                confident_labels = preds[confident_idx]

                collect_confident_data.append(confident_data)
                collect_confident_labels.append(confident_labels)
                collect_gt_confident_labels.append(gt_confident_labels)
            relabeled_data = torch.cat(collect_confident_data) if collect_confident_data else torch.empty(0)
            relabeled_labels = torch.cat(collect_confident_labels) if collect_confident_labels else torch.empty(0)
            relabeled_gt_labels = torch.cat(collect_gt_confident_labels) if collect_confident_labels else torch.empty(0)
            acc, class_acc = compute_relabel_class_wise_acc(num_classes=num_classes,
                                                       relabel_labels=relabeled_labels,
                                                       relabel_gt_labels=relabeled_gt_labels)

        return relabeled_data, relabeled_labels, acc, class_acc
    elif threshold_type == 'class_wise':
        with torch.no_grad():
            for batch_idx, (ori_data, weak_data, labels) in enumerate(label_dl):
                weak_data = weak_data.to(device)
                logits = model(weak_data)
                logits = logits.detach()
                probs = F.softmax(logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                sample_wise_confident_threshold = torch.zeros(preds.shape)

                # sample_wise_confident_threshold `(batch_size, )`
                for i in range(len(preds)):
                    sample_wise_confident_threshold[i] =  threshold[preds[i]]
                
                confident_idx = torch.where(max_probs >= sample_wise_confident_threshold)[0]

                confident_data = ori_data[confident_idx]
                gt_confident_labels = labels[confident_idx]
                confident_labels = preds[confident_idx]

                collect_confident_data.append(confident_data)
                collect_confident_labels.append(confident_labels)
                collect_gt_confident_labels.append(gt_confident_labels)
            relabeled_data = torch.cat(collect_confident_data) if collect_confident_data else torch.empty(0)
            relabeled_labels = torch.cat(collect_confident_labels) if collect_confident_labels else torch.empty(0)
            relabeled_gt_labels = torch.cat(collect_gt_confident_labels) if collect_confident_labels else torch.empty(0)
            acc, class_acc = compute_relabel_class_wise_acc(num_classes=num_classes,
                                                       relabel_labels=relabeled_labels,
                                                       relabel_gt_labels=relabeled_gt_labels)

        return relabeled_data, relabeled_labels, acc, class_acc

def get_data_distribution(labels, num_classes):
    """
    Get the distribution of the labels
    Args:
        labels (`np.array` of shape `(N,)`):
            the labels of the data
        num_classes (`int`):
            the number of classes
    Return:
        distribution (`np.array` of shape `(num_classes,)`):
            the distribution of the labels
    """
    distribution = np.zeros(num_classes)
    for i in labels:
        distribution[i] = (labels == i).float().sum()
    return distribution


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
    
def zero_weights(model, type='model'):
    with torch.no_grad():
        if type == 'model':
            # 处理参数
            for name, param in model.named_parameters():
                param.data.zero_()
            # 处理缓冲区（包括 running_mean 和 running_var）
            for name, buffer in model.named_buffers():
                buffer.zero_()
        elif type == 'params':
            for param in model.values():
                param.data.zero_()

def check_remove_batch_norm_stats_info(model_params:dict):
    if model_params is not None:
        new_model_params = copy.deepcopy(model_params)
        for name, param in new_model_params.items():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                model_params.pop(name)
        return model_params

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def set_random(seed):
    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)         
    random.seed(seed)           
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
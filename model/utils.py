import numpy as np
import logging
import torch
import torch.nn.functional as F
from model.ssl_model import SimCLRModel, BYOLModel, SimSiamModel,DINO
from model.ResNet import ResNet18,Custom_ResNet50,ResNet34
from model.SemiFed import SemiFed

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def dino_loss_fn(student_output, teacher_output, center, teacher_temp, student_temp=0.1):

    student_out = torch.cat(student_output, dim=0)   # [B*n_crops, out_dim]
    student_out = student_out / student_temp
    student_out = student_out.chunk(len(student_output))

    # Teacher 输出 softmax with temperature + centering
    teacher_out = [(t - center) / teacher_temp for t in teacher_output]
    teacher_out = [F.softmax(t, dim=-1) for t in teacher_out]
    teacher_out = torch.cat(teacher_out, dim=0).detach()
    teacher_out_chunks = teacher_out.chunk(len(teacher_output))

    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_out_chunks):
        for v in range(len(student_out)):
            if v == iq:
                continue
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms

    batch_center = torch.cat(teacher_output, dim=0).mean(dim=0, keepdim=True)
    new_center = center * 0.9 + batch_center * 0.1  # momentum=0.9

    return total_loss, new_center

import torch
import torch.nn.functional as F


def compute_covariance(z):
    # z: (batch, dim)
    b, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    # unbiased: divide by (b - 1)
    cov = (z.t() @ z) / (b - 1)
    return cov

def eval_dimensional_collapse(z, eps=1e-12):
    """
    输入:
      z: torch.Tensor, shape (batch, dim)
    返回 dict 包含多项指标
    """
    b, d = z.shape
    # center
    zc = z - z.mean(dim=0, keepdim=True)

    # per-dim std/var
    var = zc.var(dim=0, unbiased=True)  # (dim,)
    std = torch.sqrt(var + eps)

    # covariance matrix
    cov = compute_covariance(z)

    # eigenvalues (covariance symmetric PSD)
    # use torch.symeig / torch.linalg.eigvalsh
    eigs = torch.linalg.eigvalsh(cov)  # ascending order
    eigs = torch.clamp(eigs, min=0.0)
    eigs_desc = eigs.flip(0)  # descending

    total_var = eigs_desc.sum().item()
    top1 = eigs_desc[0].item()
    topk = 10
    topk_var = eigs_desc[:topk].sum().item()
    topk_ratio = topk_var / (total_var + eps)

    # participation ratio
    pr = (eigs_desc.sum() ** 2) / ( (eigs_desc ** 2).sum() + eps )
    # effective rank via entropy
    p = eigs_desc / (eigs_desc.sum() + eps)
    entropy = -(p * (p + eps).log()).sum()
    eff_rank = torch.exp(entropy)

    # off-diagonal energy
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0)
    off_energy = (off_diag ** 2).sum().item() / d

    # condition number
    cond = (eigs_desc[0] / (eigs_desc[-1] + eps)).item()

    # num dims above threshold
    threshold = total_var * 1e-3  # 可调
    dims_used = (eigs_desc > threshold).sum().item()

    stats = {
        "dim": d,
        "batch": b,
        "per_dim_std_mean": std.mean().item(),
        "per_dim_std_median": std.median().item(),
        "total_variance": total_var,
        "top1_var": top1,
        "topk_ratio": topk_ratio,
        "participation_ratio": pr.item(),
        "effective_rank": eff_rank.item(),
        "off_diagonal_energy": off_energy,
        "condition_number": cond,
        "dims_used_threshold": dims_used,
        "eigenvalues_desc": eigs_desc.detach().cpu().numpy(),  # for plotting if needed
    }
    return stats



def vicreg_loss(z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4, var_target=1.0):
    assert z1.shape == z2.shape
    b, d = z1.shape

    sim_loss = F.mse_loss(z1, z2, reduction='mean')

    def variance_term(z):
        std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
        hinge = F.relu(var_target - std)
        return hinge.mean()

    var_loss = 0.5 * (variance_term(z1) + variance_term(z2))

    def covariance_term(z):
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.t() @ z) / (b - 1)
        diag = torch.diag(cov)
        cov = cov - torch.diag_embed(diag)
        return (cov ** 2).sum() / d

    cov_loss = 0.5 * (covariance_term(z1) + covariance_term(z2))

    loss = sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss

    return loss, [sim_loss,var_loss,cov_loss]


def info_nce_loss(features, batch_size, device, n_views=2, temperature=0.07):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     n_views * self.conf.batch_size, n_views * self.conf.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels



def build_SemiFed_model(args, 
                        num_classes, 
                        model_input_channels, 
                        semi_model, 
                        base_model):
    if base_model == 'resnet18':
        unsup_net = ResNet18(args=args, num_classes=num_classes,
                    input_channels=model_input_channels)
    elif base_model == 'resnet50_ct':
        unsup_net = Custom_ResNet50(args=args, num_classes=num_classes,
                    input_channels=model_input_channels)
    elif base_model == 'resnet34':
        unsup_net = ResNet34(args=args, num_classes=num_classes,
                             input_channels=model_input_channels)

    if semi_model == 'SimCLR':
        unsup_model = SimCLRModel(unsup_net)
    elif semi_model == 'DINO':
        unsup_model = DINO(unsup_net)
    
    model = SemiFed(unsup_model,
                    semi_model=semi_model, 
                    input_channel=model_input_channels, 
                    num_class=num_classes)
    return model


def consistency_loss_fixmatch(logits_w, logits_s, T=1.0, p_cutoff=0.0, use_hard_labels=True):
    
    logits_w = logits_w.detach()
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(p_cutoff).float()
    select = max_probs.ge(p_cutoff).long()

    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
    else:
        pseudo_label = torch.softmax(logits_w / T, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask

    return masked_loss.mean(), mask.mean(), select, max_idx.long()

def consistency_loss_freematch(dataset, logits_w, logits_s, time_p, p_model, use_hard_labels=True):

    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    p_cutoff = time_p
    p_model_cutoff = p_model / torch.max(p_model,dim=-1)[0]
    threshold = p_cutoff * p_model_cutoff[max_idx]
    if dataset == 'SVHN':
        threshold = torch.clamp(threshold, min=0.9, max=0.95)
    mask = max_probs.ge(threshold)
    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask.float()
    else:
        pseudo_label = torch.softmax(logits_w / 0.5, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask.float()
    return masked_loss.mean(), mask


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
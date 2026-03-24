import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class MetricTracker:
    def __init__(self, device='cpu'):
        self.device = device
        self.reset()

    def reset(self):
        self.metrics = {

            'loss_list': [],
            'top1_correct': 0,
            'top5_correct': 0,
            'total_samples': 0,


            'old_top1_sum': 0.0,
            'old_top5_sum': 0.0,
            'batch_count': 0,
            'total_batch_size': 0,


            'all_preds': [],
            'all_targets': []
        }

    def update(self, outputs, targets, loss=None):
        """
        outputs : logits [bs, num_classes]
        targets : [bs]
        loss    : Tensor 或 float
        """
        bs = targets.size(0)
        targets = targets.to(outputs.device)

        # -------- Top‑k --------
        _, pred_top1 = outputs.topk(1, dim=1, largest=True, sorted=True)  # [bs,1]
        _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)  # [bs,5]

        # === 1) real Acc===
        self.metrics['top1_correct'] += (pred_top1.view(-1) == targets).sum().item()
        self.metrics['top5_correct'] += sum([targets[i] in pred_top5[i] for i in range(bs)])
        self.metrics['total_samples'] += bs

        # === 2) old Acc===
        batch_top1 = (pred_top1.view(-1) == targets).float().mean().item() * 100.0
        batch_top5 = sum([targets[i] in pred_top5[i] for i in range(bs)]) / bs * 100.0
        self.metrics['old_top1_sum'] += batch_top1 * bs
        self.metrics['old_top5_sum'] += batch_top5 * bs
        self.metrics['batch_count'] += 1
        self.metrics['total_batch_size'] += bs

        # -------- Loss --------
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self.metrics['loss_list'].append(loss * bs)

        # -------- for precision / recall / f1 --------
        self.metrics['all_preds'].extend(pred_top1.view(-1).detach().cpu().tolist())
        self.metrics['all_targets'].extend(targets.view(-1).detach().cpu().tolist())

    def compute(self):
        # --- real Acc ---
        real_top1 = 100.0 * self.metrics['top1_correct'] / max(1, self.metrics['total_samples'])
        real_top5 = 100.0 * self.metrics['top5_correct'] / max(1, self.metrics['total_samples'])

        # --- old Acc ---
        old_top1 = self.metrics['old_top1_sum'] / max(1, self.metrics['total_batch_size'])
        old_top5 = self.metrics['old_top5_sum'] / max(1, self.metrics['total_batch_size'])

        # --- Loss ---
        avg_loss = (np.sum(self.metrics['loss_list']) / self.metrics['total_samples']) if self.metrics['loss_list'] else 0.0

        # --- Precision / Recall / F1 (macro) ---
        precision = precision_score(self.metrics['all_targets'], self.metrics['all_preds'],
                                    average='macro', zero_division=0) * 100
        recall    = recall_score(self.metrics['all_targets'], self.metrics['all_preds'],
                                 average='macro', zero_division=0) * 100
        f1        = f1_score(self.metrics['all_targets'], self.metrics['all_preds'],
                             average='macro', zero_division=0) * 100

        return {

            'real_top1_acc': real_top1,
            'real_top5_acc': real_top5,


            'old_top1_acc': old_top1,
            'old_top5_acc': old_top5,


            'avg_loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

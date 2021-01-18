import torch
import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss

@LOSSES.register_module()
class KLDivergenceLoss(BaseWeightedLoss):
    def __init__(self, loss_weight, num_classes):
        super().__init__(loss_weight)
        self.num_classes = num_classes
    def _forward(self, cls_score, p, **kwargs):
        #KL(p∥q)=∫p(x)log (p(x)/q(x))dx
        log_q = torch.log_softmax(cls_score,dim=-1)
        loss_kl = (p*p.log()-p*log_q).sum()
        return loss_kl
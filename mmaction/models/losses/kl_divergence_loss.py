import torch
import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss

@LOSSES.register_module()
class KLDivergenceLoss(BaseWeightedLoss):
    """ Forward KL-divergence loss
    """
    def __init__(self,loss_weight=1.):
        super().__init__(loss_weight)
    def _forward(self, cls_score, p, eps=1e-8,**kwargs):
        """Calculate the forward KL-divergence between soft labels (p) and class scores.
            q = exp(cls_score)/∫exp(cls_score)
            KL(p∥q)=∫p(x)log(p(x)/q(x))dx
        Args:
                cls_score (torch.Tensor): Predicted scores for labels (energy).
                p (torch.Tensor): Ground truth soft-labels.
                eps (float): Epsilon for small value. Default: 1e-8.

        Returns:
                torch.Tensor: Mean forward KL-divergence over the batch .
        """
        #
        log_q = cls_score.log_softmax(dim=-1)
        loss_kl = (p*(p+eps).log()-p*log_q).sum(dim=-1).mean() #TODO: make reduction method optional
        return loss_kl
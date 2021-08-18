import torch
import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss

@LOSSES.register_module()
class L1RegressionLoss(BaseWeightedLoss):
    """ Forward KL-divergence loss
    """
    def __init__(self,loss_weight=1.,class_weight=None):
        super().__init__(loss_weight)
        self.class_weight = class_weight
    def _forward(self, x_hat, p, eps=1e-8,**kwargs):
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
        B,C = p.shape
        x = (p*torch.linspace(3,15,C,device=p.device)).sum(dim=-1) # B,
        loss_l1 = (x-x_hat).abs().mean()
        return loss_l1
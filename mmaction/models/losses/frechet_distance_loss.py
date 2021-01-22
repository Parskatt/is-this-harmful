import torch
import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss

@LOSSES.register_module()
class FrechetDistanceLoss(BaseWeightedLoss):
    """ Frechet Distance Loss
    """
    def __init__(self,loss_weight=1.):
        super().__init__(loss_weight)
        
    def _forward(self, cls_score, p, eps=1e-8,**kwargs):
        """Calculate the Frechet distance between between soft labels (p) and class scores under the assumption that they are Gaussian.
        Args:
                cls_score (torch.Tensor): Predicted scores for labels (energy).
                p (torch.Tensor): Ground truth soft-labels.

        Returns:
                torch.Tensor: Mean forward KL-divergence over the batch .
        """
        x = torch.linspace(0,1,num=p.shape[-1])[None,:]
        mu1 = (x*p).sum()
        sigma1 = torch.sqrt((p*(x-mu1)**2).sum())

        q = cls_score.softmax(dim=-1)
        mu2 = (x*q).sum()
        sigma2 = torch.sqrt((q*(x-mu2)**2).sum())
        frechet_loss = torch.sqrt((mu1-mu2)**2+(sigma1-sigma2)**2)
        return frechet_loss
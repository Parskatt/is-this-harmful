import torch
import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss

@LOSSES.register_module()
class EMDLoss(BaseWeightedLoss):
    """ Forward KL-divergence loss
    """
    def __init__(self,C,loss_weight=1.):
        super().__init__(loss_weight)
        self.register_buffer('C',torch.tensor(C))
        
    def _forward(self, cls_score, p,**kwargs):
        """Calculate the forward KL-divergence between soft labels (p) and class scores.
            q = exp(cls_score)/∫exp(cls_score)
            KL(p∥q)=∫p(x)log(p(x)/q(x))dx
        Args:
                cls_score (torch.Tensor): Predicted scores for labels (energy).
                p (torch.Tensor): Ground truth soft-labels.

        Returns:
                torch.Tensor: Mean forward KL-divergence over the batch .
        """
        #q = cls_score.softmax()#F.softplus(cls_score)
        loss = EMDLoss.emd_1d(cls_score,p,self.C[None,...].expand(len(cls_score),-1,-1))
        return loss

    @staticmethod
    def emd_1d(e,b,C):
        """
        Calculate EMD under the assumption that the distributions a,b are sorted and that the cost function behaves nicely.
        C is the associated cost matrix.
        If a is of shape (b,n), then the function is reduced over the batched dimension (mean reduction)
        """

        if e.dim() > 1:
            return sum(EMDLoss.emd_1d(e[k],b[k],C[k]) for k in range(len(e)))/len(e)
        a = e.softmax(dim=-1)
        log_a = e.log_softmax(dim=-1)


        N,M,zero = len(a),len(b), torch.zeros(1,device=a.device)
        F,G = torch.cumsum(torch.cat((zero,a)),0),torch.cumsum(torch.cat((zero,b)),0)
        j,cost = 0,0.
        #print(log_a)
        for i in range(N):
            for j in range(j,M):
                delta = min(F[i+1],G[j+1])-max(F[i],G[j])
                cost += delta*C[i,j]*(-log_a[j])
                if G[j+1] >= F[i+1]: break
        return cost

    
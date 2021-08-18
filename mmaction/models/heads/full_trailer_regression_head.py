import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init
from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class FullTrailerRegressionHead(BaseHead):
    """Classification head for TSN on audio.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.4,
                 num_layers = 3,
                 channels = 64,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)
        self.age_factor = nn.Parameter(torch.zeros(1))
        self.age_bias = nn.Parameter(torch.zeros(1))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        B,N,C = x.shape
        ages = (x*torch.linspace(3,15,C,device=x.device)).sum(dim=-1) # B,N
        max_age = ages.max(dim=-1).values # B,
        pred = max_age*self.age_factor.exp()+self.age_bias # B,
        return pred #B

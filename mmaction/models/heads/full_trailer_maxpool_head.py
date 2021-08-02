import torch.nn as nn
from mmcv.cnn import kaiming_init

from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class FullTrailerMaxPoolHead(BaseHead):
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
        self.fc_cls = nn.Conv1d(in_channels, self.num_classes,1,1,0)

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
        cls_score = self.fc_cls(x)
        cls_score = cls_score.max(dim=-1).values
        return cls_score

import torch.nn as nn
from mmcv.cnn import kaiming_init

from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class FullTrailerCNNHead(BaseHead):
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

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool1d((1,))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        create_conv_block = lambda in_channels,out_channels, stride: nn.Sequential(nn.Conv1d(in_channels, out_channels,3,stride,1),nn.ReLU(True),nn.GroupNorm(4,out_channels))
        self.layer1 = nn.Sequential(create_conv_block(self.in_channels,channels,2),
                                    create_conv_block(channels,channels,2),
                                    create_conv_block(channels,channels,1))
        self.fc_cls = nn.Conv1d(channels, self.num_classes,1,1,0)

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
        x = self.layer1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc_cls(x)
        if self.avg_pool:
            cls_score = self.avg_pool(cls_score)
        return cls_score

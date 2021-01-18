import copy
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..core import mean_average_precision
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class SweTrailersDataset(BaseDataset):
    def __init__(self):
        pass
    
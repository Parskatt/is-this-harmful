import copy
import os.path as osp
import json
from os.path import join
from collections import defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..core import mean_average_precision
from .swe_trailers_dataset import SweTrailersDataset
from .registry import DATASETS

from ..core import (mean_class_accuracy, top_k_accuracy, confusion_matrix,
                    wasserstein_1_distance, KL, euclidean_distance, 
                    class_euclidean_distance,mean_class_euclidean_distance)
import random

@DATASETS.register_module()
class SweTrailersFusionDataset(SweTrailersDataset):
    """Swedish Movie Trailer Dataset for prediction fusion

    Loads .json files with the annotations

    Args:
        ann_file (str): Path to the annotation file like
            ``swe_trailers.json``.
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        label_as_distribution (bool): if the label should be converted into a distribution 
    """

    def __init__(self, ann_file, *preds, pipeline, data_prefix=None, num_classes=4, label_as_distribution=True, sample_by_class=False, **kwargs):
        self.preds = preds
        super().__init__(ann_file, pipeline, data_prefix=data_prefix, num_classes=num_classes, label_as_distribution=label_as_distribution, sample_by_class=sample_by_class, **kwargs)
    def label_to_ind(self, lbl):
        return self._label_to_ind[lbl]

    def load_annotations(self):
        """Load annotation file to get video information."""
        video_infos = json.load(open(self.ann_file, "r"))
        audio_preds = json.load(open(self.audio_preds,"r"))
        video_preds = json.load(open(self.video_preds,"r"))
        for idx,clip in enumerate(video_infos):
            clip["orig_label"] = clip["label"].copy()
            if self.label_as_distribution:
                p = np.zeros(self.num_classes)
                c = self.label_to_ind(clip["label"])
                p[c] = 1
                clip["label"] = p
            else:
                lbl = clip["label"]#[0]
                clip["label"] = self.label_to_ind(lbl)
            clip["audio_path"] = join(
                self.data_prefix, clip["filename"], clip["filename"]+".npy")
            clip["preds"] = [self.preds[k][idx] for k in range(len(self.preds))]
        return video_infos


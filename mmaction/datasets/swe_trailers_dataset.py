import copy
import os.path as osp
import json
from os.path import join

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..core import mean_average_precision
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class SweTrailersDataset(BaseDataset):
    """Swedish Movie Trailer Dataset

    Loads .json files with the annotations

    Args:
        ann_file (str): Path to the annotation file like
            ``swe_trailers.json``.
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        label_as_distribution (bool): if the label should be converted into a distribution 
    """
    def __init__(self, ann_file, pipeline,data_prefix=None, num_classes = 4,label_as_distribution=True, **kwargs):
        self.label_as_distribution = label_as_distribution
        self.data_prefix = data_prefix
        self._label_to_ind = {"bt":0,"7":1,"11":2,"15":3}
        super().__init__(ann_file, pipeline, num_classes=num_classes,data_prefix=data_prefix, **kwargs)
        
    def label_to_ind(self,lbl):
        return self._label_to_ind[lbl]

    def load_annotations(self):
        """Load annotation file to get video information."""
        video_infos = json.load(open(self.ann_file,"r"))
        for clip in video_infos:
            if self.label_as_distribution:
                p = np.zeros(self.num_classes)
                for lbl in clip["label"]:
                    c = self.label_to_ind(lbl)
                    p[c] += 1
                p /= np.sum(p)
                clip["label"] = p
            else:
                lbl = clip["label"][0]
                clip["label"] = self._label_to_ind(lbl)
        return video_infos
    
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['filename_tmpl'] = results["filename"]+'_{:05}.jpg'
        results['frame_dir'] = join(self.data_prefix,results["filename"])
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['filename_tmpl'] = results["filename"]+'_{:05}.jpg'
        results['frame_dir'] = join(self.data_prefix,results["filename"])
        return self.pipeline(results)
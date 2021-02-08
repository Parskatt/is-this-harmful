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

from ..core import (mean_class_accuracy,top_k_accuracy,confusion_matrix)



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
                clip["label"] = self.label_to_ind(lbl)
        return video_infos
    
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['filename_tmpl'] = results["filename"]+'_{}.jpg'
        results['frame_dir'] = join(self.data_prefix,results["filename"])
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['filename_tmpl'] = results["filename"]+'_{}.jpg'
        results['frame_dir'] = join(self.data_prefix,results["filename"])
        return self.pipeline(results)

    def evaluate(self,
                results,
                metrics='confusion_matrix',
                metric_options=dict(top_k_accuracy=dict(topk=(1,))),
                logger=None,
                **deprecated_kwargs):
        """Perform evaluation for common datasets.
        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1,)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'confusion_matrix', 'mean_class_accuracy'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        if not self.label_as_distribution:
            gt_labels = [ann['label'] for ann in self.video_infos]
        else:
            gt_labels = [np.argmax(ann['label']) for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                    {}).setdefault(
                                                        'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
            elif metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nMean class accuracy: \n {mean_acc:.4f}'
                print_log(log_msg, logger=logger)            
            elif metric == 'confusion_matrix':
                result_argmax = np.argmax(results,axis=1)
                confusion_mat = confusion_matrix(result_argmax,gt_labels)
                eval_results['confusion_matrix'] = confusion_mat
                log_msg = f'\nConfusion Matrix:\n{confusion_mat}'
                print_log(log_msg, logger=logger)
        return eval_results

import argparse
import os
import os.path as osp
import warnings

import numpy as np
import mmcv
from mmcv import load
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.utils import get_logger
from mmaction.apis import multi_gpu_test, single_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Combine Visual and Audio modalities')
    parser.add_argument('results',nargs='+', help='result file(s)')
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    if os.path.isdir(args.results[0]):
        results = [os.path.join(args.results[0],result) for result in os.listdir(args.results[0])]
    else:
        results = args.results
    outs = np.array([load(result) for result in results])
    print(outs[0])

    outs = np.exp((np.log(outs)).sum(0))
    outs /= outs.sum(1,keepdims=True)
    print(outs)


if __name__ == '__main__':
    main()

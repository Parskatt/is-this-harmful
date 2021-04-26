import argparse
import os
import os.path as osp
import warnings

import numpy as np
import mmcv
from mmcv import load, dump
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
    parser.add_argument('video', help='result file')
    parser.add_argument('audio', help='result file')
    parser.add_argument('--fusetype',default='indep',help='type of fusion, mean or indep')
    parser.add_argument('--out',help='out-name')
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    vid = np.array(load(args.video))
    audio = np.array(load(args.audio))
    both = np.stack((vid,audio))
    x = np.linspace(0,1,4)[None,None]
    if args.fusetype == 'indep':
        outs = np.exp((np.log(both)).sum(0))
    elif args.fusetype == 'mean':
        outs = both.mean(0)
    elif args.fusetype == 'max':
        mu = (x*both).sum(-1,keepdims=True)
        mask = mu > mu.mean(0,keepdims=True) 
        outs = (both*mask).sum(0)
    else:
        raise ValueError(f'{args.fusetype} not recognized')
    outs /= outs.sum(1,keepdims=True)
    dump(outs,args.out)


if __name__ == '__main__':
    main()

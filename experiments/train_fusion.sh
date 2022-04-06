#!/bin/bash
for RUN in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/refined/fusion/fuse_MLP_${RUN}.py
done
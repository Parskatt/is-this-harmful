#!/bin/bash
for RUN in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/refined/fusion/fuse_MLP_${RUN}.py \
    work_dirs/refined_fusion_mlp_${RUN}/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean confusion_matrix --out
done
#!/bin/bash
for RUN in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=1 python tools/train_evaluate.py configs/is-this-harmful/refined/slowfast_swe_trailers_class_balanced_refined.py work_dirs/refined_train_video_1/slowfast_swe_trailers_class_balanced_refined/epoch_2.pth --eval class_euclidean mean_class_euclidean --out
done
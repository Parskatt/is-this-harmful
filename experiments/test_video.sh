#!/bin/bash
#2 3
for RUN in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/refined/slowfast_swe_trailers_class_balanced_refined.py \
    work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_class_balanced_refined/epoch_2.pth \
    --eval class_euclidean mean_class_euclidean mean_class_accuracy --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/refined/slowfast_swe_trailers_refined.py \
    work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_refined/epoch_2.pth \
    --eval class_euclidean mean_class_euclidean mean_class_accuracy --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/refined/slowfast_swe_trailers_class_balanced_refined_no_pt.py \
    work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_class_balanced_refined_no_pt/epoch_2.pth \
    --eval class_euclidean mean_class_euclidean mean_class_accuracy --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/refined/slowfast_swe_trailers_refined_no_pt.py \
    work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_refined_no_pt/epoch_2.pth \
    --eval class_euclidean mean_class_euclidean mean_class_accuracy --out
done
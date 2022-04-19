#!/bin/bash
# 
for RUN in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0 python tools/test_precalculated.py configs/datasets/swe_trailers_refined.py \
    work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_class_balanced_refined_no_pt/test_preds.json \
    --eval class_euclidean mean_class_euclidean mean_class_precision mean_class_recall
    CUDA_VISIBLE_DEVICES=0 python tools/test_precalculated.py configs/datasets/swe_trailers_refined.py \
    work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_refined/test_preds.json \
    --eval class_euclidean mean_class_euclidean mean_class_precision mean_class_recall
    CUDA_VISIBLE_DEVICES=0 python tools/test_precalculated.py configs/datasets/swe_trailers_refined.py \
    work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_refined_no_pt/test_preds.json \
    --eval class_euclidean mean_class_euclidean mean_class_precision mean_class_recall
done
#!/bin/bash
for RUN in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=1 python tools/val_evaluate.py configs/is-this-harmful/refined/tsn_r18_swe_trailers_audio_feature_class_balanced_refined.py \
    work_dirs/refined_train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_refined/epoch_5.pth --eval class_euclidean mean_class_euclidean --out
done
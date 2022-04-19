#!/bin/bash
for RUN in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/refined/tsn_r18_swe_trailers_audio_feature_class_balanced_refined.py \
    work_dirs/refined_train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_refined/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean mean_class_accuracy --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/refined/tsn_r18_swe_trailers_audio_feature_class_balanced_refined_no_pt.py \
    work_dirs/refined_train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_refined_no_pt/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean mean_class_accuracy --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/refined/tsn_r18_swe_trailers_audio_feature_refined.py \
    work_dirs/refined_train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_refined/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean mean_class_accuracy --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/refined/tsn_r18_swe_trailers_audio_feature_refined_no_pt.py \
    work_dirs/refined_train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_refined_no_pt/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean mean_class_accuracy --out
done

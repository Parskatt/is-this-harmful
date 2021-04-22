#!/bin/bash
for RUN in 1 2 3
do
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/cb/tsn_r18_swe_trailers_audio_feature_class_balanced_KL.py \
    work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_KL/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/cb/tsn_r18_swe_trailers_audio_feature_class_balanced_KL_no_pretrain.py \
    work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_KL_no_pretrain/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/pretrain/tsn_r18_swe_trailers_audio_feature_KL.py \
    work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_KL/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/pretrain/tsn_r18_swe_trailers_audio_feature_KL_no_pretrain.py \
    work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_KL_no_pretrain/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean --out
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/is-this-harmful/sampled_label/tsn_r18_swe_trailers_audio_feature_class_balanced_CE.py \
    work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_CE/epoch_5.pth \
    --eval class_euclidean mean_class_euclidean --out
done

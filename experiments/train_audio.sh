#!/bin/bash
for RUN in 1 2 3
do
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/cb/tsn_r18_swe_trailers_audio_feature_class_balanced_KL.py --work-dir work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_KL
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/cb/tsn_r18_swe_trailers_audio_feature_class_balanced_KL_no_pretrain.py --work-dir work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_KL_no_pretrain
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/pretrain/tsn_r18_swe_trailers_audio_feature_KL.py --work-dir work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_KL
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/pretrain/tsn_r18_swe_trailers_audio_feature_KL_no_pretrain.py --work-dir work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_KL_no_pretrain
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/sampled_label/tsn_r18_swe_trailers_audio_feature_class_balanced_CE.py --work-dir work_dirs/train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_CE
done
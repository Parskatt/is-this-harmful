#!/bin/bash
for RUN in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/refined/tsn_r18_swe_trailers_audio_feature_class_balanced_refined.py --work-dir work_dirs/refined_train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_refined
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/refined/tsn_r18_swe_trailers_audio_feature_class_balanced_refined_no_pt.py --work-dir work_dirs/refined_train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_class_balanced_refined_no_pt
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/refined/tsn_r18_swe_trailers_audio_feature_refined.py --work-dir work_dirs/refined_train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_refined
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/refined/tsn_r18_swe_trailers_audio_feature_refined_no_pt.py --work-dir work_dirs/refined_train_audio_${RUN}/tsn_r18_swe_trailers_audio_feature_refined_no_pt

done
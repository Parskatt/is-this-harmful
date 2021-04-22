#!/bin/bash
for RUN in 1 2 3
do
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/cb/slowfast_swe_trailers_class_balanced_KL.py --work-dir work_dirs/train_video_${RUN}/slowfast_swe_trailers_class_balanced_KL
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/cb/slowfast_swe_trailers_class_balanced_KL_no_pretrain.py --work-dir work_dirs/train_video_${RUN}/slowfast_swe_trailers_class_balanced_KL_no_pretrain
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/pretrain/slowfast_swe_trailers_KL.py --work-dir work_dirs/train_video_${RUN}/slowfast_swe_trailers_KL
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/pretrain/slowfast_swe_trailers_KL_no_pretrain.py --work-dir work_dirs/train_video_${RUN}/slowfast_swe_trailers_KL_no_pretrain
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/sampled_label/slowfast_swe_trailers_class_balanced_CE.py --work-dir work_dirs/train_video_${RUN}/slowfast_swe_trailers_class_balanced_CE
done
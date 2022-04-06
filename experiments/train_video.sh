#!/bin/bash
for RUN in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/refined/slowfast_swe_trailers_class_balanced_refined.py --work-dir work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_class_balanced_refined
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/refined/slowfast_swe_trailers_class_balanced_refined_no_pt.py --work-dir work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_class_balanced_refined_no_pt
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/refined/slowfast_swe_trailers_refined.py --work-dir work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_refined
    CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/is-this-harmful/refined/slowfast_swe_trailers_refined_no_pt.py --work-dir work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_refined_no_pt
done
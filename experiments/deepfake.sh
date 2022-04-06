
for RUN in 2 3 4 5
do
    python tools/test.py configs/is-this-harmful/refined/deepfake/slowfast_swe_trailers_class_balanced_refined.py work_dirs/refined_train_video_${RUN}/slowfast_swe_trailers_class_balanced_refined/epoch_2.pth --eval class_euclidean mean_class_euclidean --out
done
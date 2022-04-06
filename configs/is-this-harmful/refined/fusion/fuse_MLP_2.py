label_as_distribution = False
# model settings
model = dict(
    type='FusionModel',
    backbone=None,
    cls_head=dict(
        type='FusionHead',
        num_classes=4,
        in_channels=8,
        dropout_ratio=0.5,
        channels=64,
        num_layers=1,
        label_as_distribution=label_as_distribution,
    loss_cls=dict(type='CrossEntropyLoss')))

# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'SweTrailersFusionDataset'
data_root = 'data/swe_trailers/data'
data_root_val = 'data/swe_trailers/data'
data_root_test = 'data/swe_trailers/data'
ann_file_train = 'data/swe_trailers/refined_train.json'
ann_file_val = 'data/swe_trailers/refined_val.json'
ann_file_test = 'data/swe_trailers/refined_test.json'
train_pipeline = [
    dict(type='Collect', keys=['video_pred', 'audio_pred', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['video_pred','audio_pred'])
]
val_pipeline = [
    dict(type='Collect', keys=['video_pred', 'audio_pred', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['video_pred','audio_pred'])
]
test_pipeline = [
    dict(type='Collect', keys=['video_pred', 'audio_pred', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['video_pred','audio_pred'])
]
train_video_preds = "/home/johed950/is-this-harmful/work_dirs/refined_train_video_2/slowfast_swe_trailers_class_balanced_refined/train_preds.json"
train_audio_preds = "/home/johed950/is-this-harmful/work_dirs/refined_train_audio_2/tsn_r18_swe_trailers_audio_feature_class_balanced_refined/train_preds.json"
val_video_preds = "/home/johed950/is-this-harmful/work_dirs/refined_train_video_2/slowfast_swe_trailers_class_balanced_refined/val_preds.json"
val_audio_preds = "/home/johed950/is-this-harmful/work_dirs/refined_train_audio_2/tsn_r18_swe_trailers_audio_feature_class_balanced_refined/val_preds.json"
test_video_preds = "/home/johed950/is-this-harmful/work_dirs/refined_train_video_2/slowfast_swe_trailers_class_balanced_refined/test_preds.json"
test_audio_preds = "/home/johed950/is-this-harmful/work_dirs/refined_train_audio_2/tsn_r18_swe_trailers_audio_feature_class_balanced_refined/test_preds.json"

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        video_preds=train_video_preds,
        audio_preds = train_audio_preds,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        label_as_distribution=label_as_distribution,
        sample_by_class=True),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        video_preds=val_video_preds,
        audio_preds=val_audio_preds,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        label_as_distribution=label_as_distribution,
        sample_by_class=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        video_preds=test_video_preds,
        audio_preds = test_audio_preds,
        data_prefix=data_root_test,
        pipeline=test_pipeline,
        label_as_distribution=label_as_distribution))
# optimizer
optimizer = dict(
    type='SGD',momentum=0.9, lr=1e-3,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-3)
total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=10, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
exp_name = 'refined_fusion_mlp_2'
work_dir = './work_dirs/'+exp_name
load_from = None
resume_from = None
workflow = [('train', 1),('val',1)]

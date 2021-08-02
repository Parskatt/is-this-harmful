label_as_distribution = False
# model settings
model = dict(
    type='FullTrailerModel',
    backbone=None,
    cls_head=dict(
        type='FullTrailerMaxPoolHead',
        num_classes=4,
        in_channels=4,
        dropout_ratio=0.0,
        channels=64,
        num_layers=1,
        label_as_distribution=label_as_distribution,
    loss_cls=dict(type='CrossEntropyLoss')))

# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'SweFullTrailersDataset'
data_root = 'data/swe_trailers/data'
data_root_val = 'data/swe_trailers/data'
data_root_test = 'data/swe_trailers/data'
ann_file_train = 'data/swe_trailers/ft_train.json'
ann_file_val = 'data/swe_trailers/ft_val.json'
ann_file_test = 'data/swe_trailers/ft_test.json'
train_pipeline = [
    dict(type='Collect', keys=['preds', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['preds'])
]
val_pipeline = [
    dict(type='Collect', keys=['preds', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['preds'])
]
test_pipeline = [
    dict(type='Collect', keys=['preds', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['preds'])
]
#train_video_preds = "/home/johed950/is-this-harmful/work_dirs/train_video_1/slowfast_swe_trailers_class_balanced_KL/train_preds_with_id.json"
train_audio_preds = "/home/johed950/is-this-harmful/work_dirs/train_audio_1/tsn_r18_swe_trailers_audio_feature_class_balanced_KL/train_preds_with_id.json"
#val_video_preds = "/home/johed950/is-this-harmful/work_dirs/train_video_1/slowfast_swe_trailers_class_balanced_KL/val_preds.json"
val_audio_preds = "/home/johed950/is-this-harmful/work_dirs/train_audio_1/tsn_r18_swe_trailers_audio_feature_class_balanced_KL/val_preds_with_id.json"
#test_video_preds = "/home/johed950/is-this-harmful/work_dirs/train_video_1/slowfast_swe_trailers_class_balanced_KL/test_preds.json"
test_audio_preds = "/home/johed950/is-this-harmful/work_dirs/train_audio_1/tsn_r18_swe_trailers_audio_feature_class_balanced_KL/test_preds_with_id.json"

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        preds=[train_audio_preds],
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        label_as_distribution=label_as_distribution,
        sample_by_class=False),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        preds=[val_audio_preds],
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        label_as_distribution=label_as_distribution,
        sample_by_class=False),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        preds=[test_audio_preds],
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
total_epochs = 50
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
exp_name = 'full_trailer_maxpool'
work_dir = './work_dirs/'+exp_name
load_from = None
resume_from = None
workflow = [('train', 1),('val',1)]

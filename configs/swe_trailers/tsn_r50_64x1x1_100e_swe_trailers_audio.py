label_as_distribution = False
# model settings
model = dict(
    type='AudioRecognizer',
    backbone=dict(type='ResNet', depth=50, in_channels=1, norm_eval=False),
    cls_head=dict(
        type='AudioTSNHead',
        num_classes=4,
        in_channels=2048,
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'SweTrailersDataset'
data_root = 'data/swe_trailers/data'
data_root_val = 'data/swe_trailers/data'
data_root_test = 'data/swe_trailers/data'
ann_file_train = 'data/swe_trailers/train.json'
ann_file_val = 'data/swe_trailers/val.json'
ann_file_test = 'data/swe_trailers/test.json'
train_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(type='SampleFrames', clip_len=200, frame_interval=1, num_clips=1),
    dict(type='AudioDecode'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
val_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(
        type='SampleFrames',
        clip_len=200,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioDecode'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
test_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(
        type='SampleFrames',
        clip_len=200,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioDecodeInit'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        label_as_distribution=label_as_distribution),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        label_as_distribution=label_as_distribution),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        label_as_distribution=label_as_distribution))
# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 100
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tsn_r50_64x1x1_100e_kinetics400_audio/'
load_from = None
resume_from = None
workflow = [('train', 1)]

label_as_distribution = True
# model settings
model = dict(
    type='AudioRecognizer',
    backbone=dict(type='ResNet', depth=18, in_channels=1, norm_eval=False),
    cls_head=dict(
        type='AudioTSNHead',
        num_classes=4,
        in_channels=512,
        dropout_ratio=0.5,
        init_std=0.01,
        label_as_distribution=label_as_distribution,
    loss_cls=dict(type='KLDivergenceLoss')))

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
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=128, frame_interval=1, num_clips=1),
    dict(type='AudioFeatureSelector',fixed_length=300),
    dict(type='AudioNormalize'),
    dict(type='SpecAugment'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
val_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(
        type='SampleFrames',
        clip_len=200,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioFeatureSelector',fixed_length=500),
    dict(type='AudioNormalize'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
test_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(
        type='SampleFrames',
        clip_len=240,
        frame_interval=1,
        num_clips=2,
        test_mode=True),
    dict(type='AudioFeatureSelector',fixed_length=600),
    dict(type='AudioNormalize'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        label_as_distribution=label_as_distribution,
        sample_by_class=False),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        label_as_distribution=label_as_distribution,
        sample_by_class=False),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        label_as_distribution=label_as_distribution))
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-3)
total_epochs = 5
checkpoint_config = dict(interval=5)
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
exp_name = 'tsn_r18_swe_trailers_audio_feature_KL'
work_dir = './work_dirs/'+exp_name
load_from = "https://download.openmmlab.com/mmaction/recognition/audio_recognition/tsn_r18_64x1x1_100e_kinetics400_audio_feature/tsn_r18_64x1x1_100e_kinetics400_audio_feature_20201012-bf34df6c.pth"
resume_from = None
workflow = [('train', 1),('val',1)]

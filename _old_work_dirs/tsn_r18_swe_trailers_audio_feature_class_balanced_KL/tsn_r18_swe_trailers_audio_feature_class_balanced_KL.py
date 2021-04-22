label_as_distribution = True
model = dict(
    type='AudioRecognizer',
    backbone=dict(type='ResNet', depth=18, in_channels=1, norm_eval=False),
    cls_head=dict(
        type='AudioTSNHead',
        num_classes=4,
        in_channels=512,
        dropout_ratio=0.5,
        init_std=0.01,
        label_as_distribution=True,
        loss_cls=dict(type='KLDivergenceLoss')))
train_cfg = None
test_cfg = dict(average_clips='prob')
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
    dict(type='AudioFeatureSelector', fixed_length=300),
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
    dict(type='AudioFeatureSelector', fixed_length=500),
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
    dict(type='AudioFeatureSelector', fixed_length=600),
    dict(type='AudioNormalize'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='SweTrailersDataset',
        ann_file='data/swe_trailers/train.json',
        data_prefix='data/swe_trailers/data',
        pipeline=[
            dict(type='LoadAudioFeature'),
            dict(
                type='SampleFrames',
                clip_len=128,
                frame_interval=1,
                num_clips=1),
            dict(type='AudioFeatureSelector', fixed_length=300),
            dict(type='AudioNormalize'),
            dict(type='SpecAugment'),
            dict(type='FormatAudioShape', input_format='NCTF'),
            dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['audios'])
        ],
        label_as_distribution=True,
        sample_by_class=True),
    val=dict(
        type='SweTrailersDataset',
        ann_file='data/swe_trailers/val.json',
        data_prefix='data/swe_trailers/data',
        pipeline=[
            dict(type='LoadAudioFeature'),
            dict(
                type='SampleFrames',
                clip_len=200,
                frame_interval=1,
                num_clips=1,
                test_mode=True),
            dict(type='AudioFeatureSelector', fixed_length=500),
            dict(type='AudioNormalize'),
            dict(type='FormatAudioShape', input_format='NCTF'),
            dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['audios'])
        ],
        label_as_distribution=True,
        sample_by_class=True),
    test=dict(
        type='SweTrailersDataset',
        ann_file='data/swe_trailers/test.json',
        data_prefix='data/swe_trailers/data',
        pipeline=[
            dict(type='LoadAudioFeature'),
            dict(
                type='SampleFrames',
                clip_len=240,
                frame_interval=1,
                num_clips=2,
                test_mode=True),
            dict(type='AudioFeatureSelector', fixed_length=600),
            dict(type='AudioNormalize'),
            dict(type='FormatAudioShape', input_format='NCTF'),
            dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['audios'])
        ],
        label_as_distribution=True))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr=0.001)
total_epochs = 10
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=10, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tsn_r18_swe_trailers_audio_feature_class_balanced_KL/'
load_from = 'https://download.openmmlab.com/mmaction/recognition/audio_recognition/tsn_r18_64x1x1_100e_kinetics400_audio_feature/tsn_r18_64x1x1_100e_kinetics400_audio_feature_20201012-bf34df6c.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
gpu_ids = range(0, 1)
omnisource = False

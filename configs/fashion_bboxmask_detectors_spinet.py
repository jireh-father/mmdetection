_base_ = [
    './_base_/models/cascade_mask_rcnn_r50_fpn_fashion_spinet.py',
    # './_base_/schedules/schedule_1x.py',
]
norm_cfg = dict(type='SyncBN', momentum=0.01, eps=1e-3, requires_grad=True)
model = dict(
    backbone=dict(
        type='SpineNet',
        arch="190",
        norm_cfg=norm_cfg),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')))


# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# # optimizer
# # optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# # lr_config = dict(
# #     policy='step',
# #     warmup='linear',
# #     warmup_iters=500,
# #     warmup_ratio=0.001,
# #     step=[20, 23])
# # total_epochs = 24
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[38, 42])
# total_epochs = 48

# checkpoint_config = dict(interval=1)
# # yapf:disable
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ])
# # yapf:enable
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None
# resume_from = None


dataset_type = 'FashionDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(1280, 1280),
        ratio_range=(0.1, 1.9),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1280, 1280)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(1280, 1280)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'fashion/train_total.json',
        img_prefix=data_root + 'fashion/train_images',
        # classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'fashion/val_split.json',
        img_prefix=data_root + 'fashion/train_images',
        # classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'fashion/test_pubilc.json',
        img_prefix=data_root + 'fashion/test_images',
        # classes=classes,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.07,
    momentum=0.9,
    weight_decay=4e-5,
    paramwise_options=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=8000,
    warmup_ratio=0.1,
    step=[320, 340])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 50
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/spinenet_190_B/'
load_from = None
resume_from = None
workflow = [('train', 1)]

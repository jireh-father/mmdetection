_base_ = './_base_/models/cascade_mask_rcnn_r50_fpn.py'
classes = ('ectropion', )
#data = dict(
#    train=dict(classes=classes),
#    val=dict(classes=classes),
#    test=dict(classes=classes))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_ectropion_anno.json',
        img_prefix=data_root + 'ectropion_images/',
        classes=classes, pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_ectropion_anno.json',
        img_prefix=data_root + 'ectropion_images/',
        classes=classes, pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
#        ann_file=None,#data_root + 'annotations/val_ectropion_anno.json',
        img_prefix='../aiforpet_dog_eye_disease_recognition/resource/ectropion_binary_dataset_v3/val/1_ectropion',
        classes=classes, pipeline=test_pipeline))

log_level = 'INFO'
evaluation = dict(interval=1, metric=['bbox', 'segm'])
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
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
total_epochs = 100
dist_params = dict(backend='nccl')
work_dir = './work_dirs/cascade_mask_rcnn_r50_fpn_1x_2'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpus = 0

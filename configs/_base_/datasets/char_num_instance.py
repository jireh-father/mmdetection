dataset_type = 'CharNumCocoDataset'
data_root = 'data/coco/'
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
classes=('0','1','2','3','4','5','6','7','8','9')


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'char_num/train_coco.json',
        img_prefix=data_root + 'char_num/images',
        classes=classes, pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'char_num/val_coco.json',
        img_prefix=data_root + 'char_num/images',
        classes=classes, pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'char_num/test_coco.json',
        img_prefix=data_root + 'char_num/images',
        classes=classes, pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])

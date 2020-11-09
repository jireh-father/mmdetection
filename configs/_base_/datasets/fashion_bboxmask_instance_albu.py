dataset_type = 'FashionDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(800, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
#

train_transforms = [
    dict(type='RandomResizedCrop',
         height=800,
         width=800,
         scale=[0.4, 1.0],
         interpolation=0,
         p=0.5),
    dict(type='Resize', height=800, width=800, p=1.0),
    # dict(type='OneOf',
    #      transforms=[
    #          dict(type='ShiftScaleRotate', border_mode=0,
    #               shift_limit=[-0.2, 0.2], scale_limit=[-0.2, 0.2], rotate_limit=[-20, 20], p=0.9),
    #          dict(type='OpticalDistortion', border_mode=0,
    #               distort_limit=[-0.5, 0.5], shift_limit=[-0.5, 0.5], p=0.9),
    #          dict(type='GridDistortion', num_steps=5, distort_limit=[-0., 0.3], border_mode=0, p=0.9),
    #          dict(type='ElasticTransform', border_mode=0, p=0.9),
    #          dict(type='IAAPerspective', p=0.9),
    #      ], p=0.01),
    # dict(type='Rotate', limit=[-25, 25], border_mode=0, p=0.1),
    # dict(type='OneOf',
    #      transforms=[
    #          dict(type='HueSaturationValue',
    #               hue_shift_limit=[-20, 20],
    #               sat_shift_limit=[-30, 30],
    #               val_shift_limit=[-20, 20], p=0.9),
    #          dict(type='RandomBrightnessContrast',
    #               brightness_limit=[-0.2, 0.2],
    #               contrast_limit=[-0.2, 0.2], p=0.9)
    #      ], p=0.1),
    # dict(type='OneOf',
    #      transforms=[
    #          dict(type='MotionBlur', blur_limit=[3, 7], p=0.9),
    #          dict(type='Blur', blur_limit=[3, 7], p=0.9),
    #          dict(type='MedianBlur', blur_limit=[3, 7], p=0.9),
    #          dict(type='GaussianBlur', blur_limit=[3, 7], p=0.9),
    #      ], p=0.05),
    # dict(type='OneOf',
    #      transforms=[
    #          dict(type='RandomGamma', gamma_limit=[30, 150], p=0.9),
    #          dict(type='RGBShift', p=0.9),
    #          dict(type='CLAHE', clip_limit=[1, 15], p=0.9),
    #          dict(type='ChannelShuffle', p=0.9),
    #          dict(type='InvertImg', p=0.9),
    #      ], p=0.1),
    # dict(type='OneOf',
    #      transforms=[
    #          dict(type='RandomSnow', p=0.9),
    #          dict(type='RandomRain', p=0.9),
    #          dict(type='RandomFog', p=0.9),
    #          dict(type='RandomSunFlare', num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=110, p=0.9),
    #          dict(type='RandomShadow', p=0.9),
    #      ], p=0.1),
    # dict(type='OneOf',
    #      transforms=[
    #          dict(type='GaussNoise', var_limit=[10, 20], p=0.9),
    #          dict(type='ISONoise', color_shift=[0, 15], p=0.9),
    #          dict(type='MultiplicativeNoise', p=0.9),
    #      ], p=0.1),
    # dict(type='ToSepia', p=0.05),
    # dict(type='Solarize', p=0.05),
    # dict(type='Equalize', p=0.05),
    # dict(type='Posterize', p=0.05),
    # dict(type='FancyPCA', p=0.05),
    # dict(type='HorizontalFlip', p=0.2),
    # dict(type='VerticalFlip', p=0.2),
    # dict(type='GridDropout', p=0.05),
    # # dict(type='ChannelDropout', p=0.05),
    # dict(type='Cutout', num_holes=30, max_h_size=37, max_w_size=37, fill_value=0, p=0.05),
    # dict(type='Downscale', scale_min=0.5, scale_max=0.8, p=0.05),
    # dict(type='ImageCompression', quality_lower=60, p=0.05),

]

val_transforms = [
    dict(type='Resize', height=800, width=800, p=1.0)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Albu',
        transforms=train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='coco',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# classes = ('top',
#            'blouse',
#            't-shirt',
#            'Knitted fabric',
#            'shirt',
#            'bra top',
#            'hood',
#            'blue jeans',
#            'pants',
#            'skirt',
#            'leggings',
#            'jogger pants',
#            'coat',
#            'jacket',
#            'jumper',
#            'padding jacket',
#            'best',
#            'kadigan',
#            'zip up',
#            'dress',
#            'jumpsuit')

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
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
evaluation = dict(metric=['bbox', 'segm'])

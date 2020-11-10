_base_ = [
    './_base_/models/cascade_mask_rcnn_r50_fpn_fashion.py',
    './_base_/datasets/fashion_bboxmask_instance_albu.py',
    # './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
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

work_dir = './work_dirs/fashion_detectors'

# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# total_epochs = 12

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20, 23])
total_epochs = 24

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

_base_ = 'mmdet3d::pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py'

custom_imports = dict(imports=['projects.TC2D3D.tc2d3d'])

# model settings
model = dict(
    backbone=dict(
        frozen_stages=1,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    neck=dict(start_level=1, num_outs=5),
    bbox_head=dict(
        type='TC2D3DHead',
        strides=(8, 16, 32, 64, 128),
        regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384), (384, 1e8)),
        use_depth_classifier=False,
        weight_dim=-1))

backend_args = None

tta_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='mmdet.Resize', scale=scale, keep_ratio=True)
            for scale in [(828, 250), (1242, 375), (1863, 563)]
        ], [dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5)],
                    [dict(type='Pack3DDetInputs', keys=['img'])]])
]

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)

train_cfg = dict(val_interval=2)
auto_scale_lr = dict(enable=True, base_batch_size=12)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(
    type='Det3DTTAModel',
    tta_cfg=dict(nms_pre=100, nms_thr=0.05, score_thr=0.001, max_per_img=20))

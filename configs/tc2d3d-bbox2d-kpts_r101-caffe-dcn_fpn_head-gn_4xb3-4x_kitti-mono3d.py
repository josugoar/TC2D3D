_base_ = './fcos3d-bbox2d-kpts_r101-caffe-dcn_fpn_head-gn_4xb3-4x_kitti-mono3d.py'  # noqa: E501

# model settings
model = dict(
    bbox_head=dict(
        reg_branch=(
            (),  # offset
            (),  # depth
            (256, ),  # size
            (256, ),  # rot
            (256, ),  # kpts
            (256, )  # bbox2d
        ),
        use_tight_constraint=True),
    train_cfg=dict(code_weight=[
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0
    ]))

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize3D', scale=(1242, 375), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='BBoxes3DToBBoxes'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='Resize3D', scale=(1242, 375), keep_ratio=True),
    dict(type='Pack3DDetInputs', keys=['img'])
]
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
            dict(type='Resize3D', scale=scale, keep_ratio=True)
            for scale in [(828, 250), (1242, 375), (1863, 563)]
        ], [dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5)],
                    [dict(type='Pack3DDetInputs', keys=['img'])]])
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

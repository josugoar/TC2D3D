_base_ = './fcos3d_r101-caffe-dcn_fpn_head-gn_4xb3-4x_kitti-mono3d.py'

custom_imports = dict(imports=['projects.TC2D3D.tc2d3d'])

# model settings
model = dict(
    bbox_head=dict(
        type='TC2D3DHead',
        group_reg_dims=(2, 1, 3, 1, 4),  # offset, depth, size, rot, bbox2d
        reg_branch=(
            (),       # offset
            (),       # depth
            (256, ),  # size
            (256, ),  # rot
            (256, )   # bbox2d
        )))

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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

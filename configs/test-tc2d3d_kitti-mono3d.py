_base_ = [
    'mmdet3d::_base_/datasets/kitti-mono3d.py',
    'mmdet3d::_base_/schedules/mmdet-schedule-1x.py',
    'mmdet3d::_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.TC2D3D.tc2d3d'])

model = dict(type='TC2D3DTest')

test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ])
]

test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

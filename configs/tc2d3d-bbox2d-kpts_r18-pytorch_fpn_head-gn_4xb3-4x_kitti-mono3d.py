_base_ = './tc2d3d-bbox2d-kpts_r101-caffe-dcn_fpn_head-gn_4xb3-4x_kitti-mono3d.py'  # noqa: E501

# model settings
model = dict(
    backbone=dict(
        depth=18,
        style='pytorch',
        init_cfg=dict(checkpoint='torchvision://resnet18'),
        dcn=None),
    neck=dict(in_channels=[64, 128, 256, 512]))

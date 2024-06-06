_base_ = './tc2d3d_r101-caffe-dcn_fpn_head-gn_4xb3-4x_kitti-mono3d.py'

# model settings
model = dict(
    backbone=dict(
        depth=18,
        style='pytorch',
        init_cfg=dict(checkpoint='torchvision://resnet18'),
        dcn=None),
    neck=dict(in_channels=(64, 128, 256, 512)))

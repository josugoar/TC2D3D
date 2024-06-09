from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import box3d_to_bbox, points_cam2img


@TRANSFORMS.register_module()
class BBoxes3DToBBoxes(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        bboxes_3d = input_dict['gt_bboxes_3d']
        cam2img = input_dict['cam2img']

        bboxes = box3d_to_bbox(bboxes_3d.tensor.numpy(force=True), cam2img)

        input_dict['gt_bboxes'] = bboxes

        return input_dict


@TRANSFORMS.register_module()
class BottomCenterToCenters2DWithDepth(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        bboxes_3d = input_dict['gt_bboxes_3d']
        cam2img = input_dict['cam2img']

        centers_2d_with_depth = points_cam2img(
            bboxes_3d.bottom_center, cam2img,
            with_depth=True).numpy(force=True)

        input_dict['centers_2d'] = centers_2d_with_depth[:, :2]
        input_dict['depths'] = centers_2d_with_depth[:, 2]

        return input_dict

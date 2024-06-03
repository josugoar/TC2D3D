from mmcv.transforms import BaseTransform

from mmdet3d.structures import box3d_to_bbox
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BBoxes3DToBBoxes(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        cam2img = input_dict['cam2img']

        gt_bboxes = box3d_to_bbox(gt_bboxes_3d, cam2img)

        input_dict['gt_bboxes'] = gt_bboxes

        return input_dict

from mmcv.transforms import BaseTransform

from mmdet3d.structures import box3d_to_bbox
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BBoxes3DToBBoxes(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        bboxes_3d = input_dict['gt_bboxes_3d']
        cam2img = input_dict['cam2img']

        bboxes = bboxes_3d.tensor.new_tensor(
            box3d_to_bbox(bboxes_3d.tensor.numpy(force=True), cam2img))

        input_dict['gt_bboxes'] = bboxes

        return input_dict

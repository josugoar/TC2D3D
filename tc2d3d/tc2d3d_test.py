import numpy as np
import torch
from mmengine.structures import InstanceData

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import box3d_to_bbox
from .utils import points_img2tc


@MODELS.register_module()
class TC2D3DTest(Base3DDetector):

    def loss(self, batch_inputs, batch_data_samples): ...

    def predict(self, batch_inputs, batch_data_samples):
        results = []
        for batch_data_sample in batch_data_samples:
            metainfo = batch_data_sample.metainfo
            box_type_3d = metainfo['box_type_3d']
            cam2img = metainfo['cam2img']

            eval_ann_info = batch_data_sample.eval_ann_info
            bboxes_3d = eval_ann_info['gt_bboxes_3d']
            labels_3d = bboxes_3d.tensor.new_tensor(eval_ann_info['gt_labels_3d'], dtype=torch.long)
            scores_3d = torch.ones_like(labels_3d)

            bboxes = bboxes_3d.tensor.new_tensor(
                box3d_to_bbox(bboxes_3d.tensor.numpy(force=True), cam2img))
            location = points_img2tc(bboxes,
                                     bboxes.new_tensor(np.array(cam2img)),
                                     bboxes_3d.dims,
                                     bboxes_3d.yaw,
                                     bboxes_3d.tensor.clone())
            bboxes_3d.tensor[:, :3] = location

            result = InstanceData()
            result.bboxes_3d = box_type_3d(bboxes_3d.tensor)
            result.labels_3d = labels_3d
            result.scores_3d = scores_3d
            results.append(result)

        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results)
        return batch_data_samples

    def _forward(self, batch_inputs, batch_data_samples=None): ...

    def extract_feat(self, batch_inputs): ...

import torch
from mmengine.structures import InstanceData

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import box3d_to_bbox
from .box_ops import bbox_to_box3d


@MODELS.register_module()
class TC2D3DTest(Base3DDetector):

    def loss(self, batch_inputs, batch_data_samples):
        raise NotImplementedError

    def predict(self, batch_inputs, batch_data_samples):
        result_list_3d = []

        for batch_data_sample in batch_data_samples:
            bboxes_3d = batch_data_sample.eval_ann_info['gt_bboxes_3d']
            labels_3d = bboxes_3d.tensor.new_tensor(
                batch_data_sample.eval_ann_info['gt_labels_3d'],
                dtype=torch.long)
            scores_3d = bboxes_3d.tensor.new_ones(labels_3d.shape)
            cam2img = batch_data_sample.metainfo['cam2img']

            bboxes = bboxes_3d.tensor.new_tensor(
                box3d_to_bbox(bboxes_3d.tensor.numpy(force=True), cam2img))
            bboxes_3d.tensor = bbox_to_box3d(
                bboxes,
                bboxes_3d.dims,
                bboxes_3d.yaw,
                cam2img,
                origin=(0.5, 1.0, 0.5))

            result = InstanceData()
            result.bboxes_3d = bboxes_3d
            result.labels_3d = labels_3d
            result.scores_3d = scores_3d
            result_list_3d.append(result)

        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  result_list_3d)

        return predictions

    def _forward(self, batch_inputs, batch_data_samples=None):
        raise NotImplementedError

    def extract_feat(self, batch_inputs):
        raise NotImplementedError

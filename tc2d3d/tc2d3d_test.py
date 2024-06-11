from typing import Tuple, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import box3d_to_bbox
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .box_ops import bbox_to_box3d

EPSILON = 1e-4


@MODELS.register_module()
class TC2D3DTest(Base3DDetector):

    def __init__(self,
                 *args,
                 origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 noise: bool = False,
                 noise_mean: float = 0.0,
                 noise_std: float = 1.0,
                 **kwargs):
        self.origin = origin
        self.noise = noise
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        super().__init__(*args, **kwargs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        raise NotImplementedError

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        results = []

        for batch_data_sample in batch_data_samples:
            cam2img = batch_data_sample.metainfo['cam2img']
            box_type_3d = batch_data_sample.metainfo['box_type_3d']

            bboxes_3d = batch_data_sample.eval_ann_info['gt_bboxes_3d']
            labels_3d = bboxes_3d.tensor.new_tensor(
                batch_data_sample.eval_ann_info['gt_labels_3d'],
                dtype=torch.long)
            scores_3d = bboxes_3d.tensor.new_ones(labels_3d.shape)

            bboxes = bboxes_3d.tensor.new_tensor(
                box3d_to_bbox(bboxes_3d.tensor.numpy(force=True), cam2img))
            if self.noise:
                bboxes += torch.normal(self.noise_mean, self.noise_std,
                                       bboxes.shape)

            bboxes_3d.tensor = bbox_to_box3d(
                bboxes,
                bboxes_3d.dims,
                bboxes_3d.yaw,
                cam2img,
                origin=self.origin)

            result = InstanceData()
            result.bboxes_3d = box_type_3d(
                bboxes_3d.tensor - EPSILON,
                box_dim=bboxes_3d.box_dim,
                with_yaw=bboxes_3d.with_yaw,
                origin=self.origin)
            result.labels_3d = labels_3d
            result.scores_3d = scores_3d
            results.append(result)

        predictions = self.add_pred_to_datasample(batch_data_samples, results)

        return predictions

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        raise NotImplementedError

    def extract_feat(self, batch_inputs: Tensor):
        raise NotImplementedError

from typing import List, Tuple

import torch
from mmengine.model import BaseTTAModel
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.structures import Det3DDataSample, xywhr2xyxyr


# TODO: bboxes2d TTA
@MODELS.register_module()
class Det3DTTAModel(BaseTTAModel):

    def __init__(self, tta_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.tta_cfg = tta_cfg

    def merge_aug_bboxes(self, aug_bboxes: List[Tensor],
                         aug_bboxes_for_nms: List[Tensor],
                         aug_scores: List[Tensor],
                         img_metas: List[str]) -> Tuple[Tensor, Tensor]:
        recovered_bboxes = []
        recovered_bboxes_for_nms = []
        for bboxes, bboxes_for_nms, img_info in zip(aug_bboxes,
                                                    aug_bboxes_for_nms,
                                                    img_metas):
            recovered_bboxes.append(bboxes)
            recovered_bboxes_for_nms.append(bboxes_for_nms)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        bboxes_for_nms = torch.cat(recovered_bboxes_for_nms, dim=0)
        scores = torch.cat(aug_scores, dim=0)
        return bboxes, bboxes_for_nms, scores

    def merge_preds(self, data_samples_list: List[List[Det3DDataSample]]):
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples

    def _merge_single_sample(
            self, data_samples: List[Det3DDataSample]) -> Det3DDataSample:
        aug_bboxes = []
        aug_bboxes_for_nms = []
        aug_scores = []
        img_metas = []
        for data_sample in data_samples:
            aug_bboxes.append(data_sample.pred_instances_3d.bboxes_3d)
            aug_bboxes_for_nms = xywhr2xyxyr(
                data_sample.metainfo['box_type_3d'](
                    aug_bboxes, box_dim=aug_bboxes.shape[1]).bev)
            aug_bboxes_for_nms.append(aug_bboxes_for_nms)
            aug_scores.append(data_sample.pred_instances_3d.scores_3d)
            img_metas.append(data_sample.metainfo)

        merged_bboxes, merged_bboxes_for_nms, merged_scores = \
            self.merge_aug_bboxes(aug_bboxes, aug_bboxes_for_nms, aug_scores,
                                  img_metas)

        if merged_bboxes.numel() == 0:
            return data_samples[0]

        det_bboxes, det_scores, det_labels = box3d_multiclass_nms(
            merged_bboxes, merged_bboxes_for_nms, merged_scores,
            self.tta_cfg.score_thr, self.tta_cfg.max_per_img, self.tta_cfg)

        results_3d = InstanceData()
        results_3d.bboxes_3d = det_bboxes
        results_3d.scores_3d = det_scores
        results_3d.labels_3d = det_labels
        det_results = data_samples[0]
        det_results.pred_instances_3d = results_3d
        return det_results

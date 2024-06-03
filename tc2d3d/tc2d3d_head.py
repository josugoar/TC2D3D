from typing import List

import numpy as np
import torch
from mmdet.structures.bbox import distance2bbox
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import PGDHead
from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import MODELS
from mmdet3d.structures import points_img2cam, xywhr2xyxyr
from mmdet3d.utils.typing_utils import ConfigType
from .utils import points_img2tc


@MODELS.register_module()
class TC2D3DHead(PGDHead):

    def __init__(self,
                 *args,
                 use_depth_classifier: bool = False,
                 weight_dim: int = -1,
                 **kwargs):
        super().__init__(
            *args,
            use_depth_classifier=use_depth_classifier,
            weight_dim=weight_dim,
            **kwargs)

    def loss_by_feat(self, *args, **kwarsg):
        loss_dict = super().loss_by_feat(*args, **kwarsg)

        del loss_dict['loss_offset']
        del loss_dict['loss_depth']
        if self.pred_keypoints:
            del loss_dict['loss_kpts']
        if self.pred_bbox2d:
            del loss_dict['loss_consistency']

        return loss_dict

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                dir_cls_pred_list: List[Tensor],
                                depth_cls_pred_list: List[Tensor],
                                weight_list: List[Tensor],
                                attr_pred_list: List[Tensor],
                                centerness_pred_list: List[Tensor],
                                mlvl_points: Tensor,
                                img_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = False) -> InstanceData:
        rescale = False
        view = np.array(img_meta['cam2img'])
        scale_factor = img_meta['scale_factor']
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []
        mlvl_depth_cls_scores = []
        mlvl_depth_uncertainty = []
        mlvl_bboxes2d = None
        if self.pred_bbox2d:
            mlvl_bboxes2d = []

        for cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
                attr_pred, centerness, points in zip(
                    cls_score_list, bbox_pred_list, dir_cls_pred_list,
                    depth_cls_pred_list, weight_list, attr_pred_list,
                    centerness_pred_list, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            depth_cls_pred = depth_cls_pred.permute(1, 2, 0).reshape(
                -1, self.num_depth_cls)
            depth_cls_score = F.softmax(
                depth_cls_pred, dim=-1).topk(
                    k=2, dim=-1)[0].mean(dim=-1)
            if self.weight_dim != -1:
                weight = weight.permute(1, 2, 0).reshape(-1, self.weight_dim)
            else:
                weight = weight.permute(1, 2, 0).reshape(-1, 1)
            depth_uncertainty = torch.exp(-weight[:, -1])
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred3d = bbox_pred[:, :self.bbox_coder.bbox_code_size]
            if self.pred_bbox2d:
                bbox_pred2d = bbox_pred[:, -4:]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                merged_scores = scores * centerness[:, None]
                if self.use_depth_classifier:
                    merged_scores *= depth_cls_score[:, None]
                    if self.weight_dim != -1:
                        merged_scores *= depth_uncertainty[:, None]
                max_scores, _ = merged_scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred3d = bbox_pred3d[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                depth_cls_pred = depth_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                depth_cls_score = depth_cls_score[topk_inds]
                depth_uncertainty = depth_uncertainty[topk_inds]
                attr_score = attr_score[topk_inds]
                if self.pred_bbox2d:
                    bbox_pred2d = bbox_pred2d[topk_inds, :]
            # change the offset to actual center predictions
            bbox_pred3d[:, :2] = points - bbox_pred3d[:, :2]
            if rescale:
                if self.pred_bbox2d:
                    bbox_pred2d /= bbox_pred2d.new_tensor(scale_factor[0])
            if self.use_depth_classifier:
                prob_depth_pred = self.bbox_coder.decode_prob_depth(
                    depth_cls_pred, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                bbox_pred3d[:, 2] = sig_alpha * bbox_pred3d[:, 2] + \
                    (1 - sig_alpha) * prob_depth_pred
            pred_center2d = bbox_pred3d[:, :3].clone()
            bbox_pred3d[:, :3] = points_img2cam(bbox_pred3d[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred3d)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_depth_cls_scores.append(depth_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            mlvl_depth_uncertainty.append(depth_uncertainty)
            if self.pred_bbox2d:
                bbox_pred2d = distance2bbox(
                    points, bbox_pred2d, max_shape=None)
                mlvl_bboxes2d.append(bbox_pred2d)
                pred_center2d = torch.cat([
                    (bbox_pred2d[:, 0] + bbox_pred3d[:, 2]) / 2,
                    (bbox_pred2d[:, 1] + bbox_pred3d[:, 3]) / 2], dim=1)
                mlvl_centers2d[-1] = pred_center2d

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.pred_bbox2d:
            mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)

        # change local yaw to global yaw for 3D nms
        cam2img = torch.eye(
            4, dtype=mlvl_centers2d.dtype, device=mlvl_centers2d.device)
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

        if self.pred_bbox2d:
            dims = mlvl_bboxes[:, 3:6]
            yaw = mlvl_bboxes[:, 6]
            locations = points_img2tc(mlvl_bboxes2d, cam2img, dims, yaw, mlvl_bboxes)
            mlvl_bboxes[:, :3] = locations

        mlvl_bboxes_for_nms = xywhr2xyxyr(img_meta['box_type_3d'](
            mlvl_bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 1.0, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # no scale_factors in box3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        if self.use_depth_classifier:  # multiply the depth confidence
            mlvl_depth_cls_scores = torch.cat(mlvl_depth_cls_scores)
            mlvl_nms_scores *= mlvl_depth_cls_scores[:, None]
            if self.weight_dim != -1:
                mlvl_depth_uncertainty = torch.cat(mlvl_depth_uncertainty)
                mlvl_nms_scores *= mlvl_depth_uncertainty[:, None]
        nms_results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                           mlvl_nms_scores, cfg.score_thr,
                                           cfg.max_per_img, cfg,
                                           mlvl_dir_scores, mlvl_attr_scores,
                                           mlvl_bboxes2d)
        bboxes, scores, labels, dir_scores, attrs = nms_results[0:5]
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = img_meta['box_type_3d'](
            bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 1.0, 0.5))
        if not self.pred_attrs:
            attrs = None

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        if attrs is not None:
            results.attr_labels = attrs

        results_2d = InstanceData()

        if self.pred_bbox2d:
            bboxes2d = nms_results[-1]
            results_2d.bboxes = bboxes2d
            results_2d.scores = scores
            results_2d.labels = labels

        results_2d = None

        return results, results_2d

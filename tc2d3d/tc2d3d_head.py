from typing import List, Optional, Tuple

import numpy as np
import torch
from mmdet.structures.bbox import distance2bbox
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import PGDHead
from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import MODELS
from mmdet3d.structures import points_cam2img, points_img2cam, xywhr2xyxyr
from mmdet3d.utils.typing_utils import ConfigType
from .box_ops import bbox_to_box3d


@MODELS.register_module()
class TC2D3DHead(PGDHead):

    def __init__(self, *args, use_tc: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_tc = use_tc

    def get_proj_bbox2d(self,
                        bbox_preds: List[Tensor],
                        pos_dir_cls_preds: List[Tensor],
                        labels_3d: List[Tensor],
                        bbox_targets_3d: List[Tensor],
                        pos_points: Tensor,
                        pos_inds: Tensor,
                        batch_img_metas: List[dict],
                        pos_depth_cls_preds: Optional[Tensor] = None,
                        pos_weights: Optional[Tensor] = None,
                        pos_cls_scores: Optional[Tensor] = None,
                        with_kpts: bool = False) -> Tuple[Tensor]:
        views = [np.array(img_meta['cam2img']) for img_meta in batch_img_metas]
        num_imgs = len(batch_img_metas)
        img_idx = []
        for label in labels_3d:
            for idx in range(num_imgs):
                img_idx.append(
                    labels_3d[0].new_ones(int(len(label) / num_imgs)) * idx)
        img_idx = torch.cat(img_idx)
        pos_img_idx = img_idx[pos_inds]

        flatten_strided_bbox_preds = []
        flatten_strided_bbox2d_preds = []
        flatten_bbox_targets_3d = []
        flatten_strides = []

        for stride_idx, bbox_pred in enumerate(bbox_preds):
            flatten_bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                -1, sum(self.group_reg_dims))
            if self.use_tc:
                flatten_bbox_pred[:, :2] = (flatten_bbox_pred[:, -4:-2] -
                                            flatten_bbox_pred[:, -2:]) / 2
            flatten_bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_bbox_pred[:, -4:] *= self.strides[stride_idx]
            flatten_strided_bbox_preds.append(
                flatten_bbox_pred[:, :self.bbox_coder.bbox_code_size])
            flatten_strided_bbox2d_preds.append(flatten_bbox_pred[:, -4:])

            bbox_target_3d = bbox_targets_3d[stride_idx].clone()
            bbox_target_3d[:, :2] *= self.strides[stride_idx]
            bbox_target_3d[:, -4:] *= self.strides[stride_idx]
            flatten_bbox_targets_3d.append(bbox_target_3d)

            flatten_stride = flatten_bbox_pred.new_ones(
                *flatten_bbox_pred.shape[:-1], 1) * self.strides[stride_idx]
            flatten_strides.append(flatten_stride)

        flatten_strided_bbox_preds = torch.cat(flatten_strided_bbox_preds)
        flatten_strided_bbox2d_preds = torch.cat(flatten_strided_bbox2d_preds)
        flatten_bbox_targets_3d = torch.cat(flatten_bbox_targets_3d)
        flatten_strides = torch.cat(flatten_strides)
        pos_strided_bbox_preds = flatten_strided_bbox_preds[pos_inds]
        pos_strided_bbox2d_preds = flatten_strided_bbox2d_preds[pos_inds]
        pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
        pos_strides = flatten_strides[pos_inds]

        pos_decoded_bbox2d_preds = distance2bbox(pos_points,
                                                 pos_strided_bbox2d_preds)

        pos_strided_bbox_preds[:, :2] = \
            pos_points - pos_strided_bbox_preds[:, :2]
        pos_bbox_targets_3d[:, :2] = \
            pos_points - pos_bbox_targets_3d[:, :2]

        if self.use_depth_classifier and (not self.use_onlyreg_proj):
            pos_prob_depth_preds = self.bbox_coder.decode_prob_depth(
                pos_depth_cls_preds, self.depth_range, self.depth_unit,
                self.division, self.num_depth_cls)
            sig_alpha = torch.sigmoid(self.fuse_lambda)
            pos_strided_bbox_preds[:, 2] = \
                sig_alpha * pos_strided_bbox_preds.clone()[:, 2] + \
                (1 - sig_alpha) * pos_prob_depth_preds

        box_corners_in_image = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))
        box_corners_in_image_gt = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))

        for idx in range(num_imgs):
            mask = (pos_img_idx == idx)
            if pos_strided_bbox_preds[mask].shape[0] == 0:
                continue
            cam2img = torch.eye(
                4,
                dtype=pos_strided_bbox_preds.dtype,
                device=pos_strided_bbox_preds.device)
            view_shape = views[idx].shape
            cam2img[:view_shape[0], :view_shape[1]] = \
                pos_strided_bbox_preds.new_tensor(views[idx])

            centers2d_preds = pos_strided_bbox_preds.clone()[mask, :2]
            centers2d_targets = pos_bbox_targets_3d.clone()[mask, :2]
            centers3d_targets = points_img2cam(pos_bbox_targets_3d[mask, :3],
                                               views[idx])

            # decode yaws
            if self.use_direction_classifier:
                pos_dir_cls_scores = torch.max(
                    pos_dir_cls_preds[mask], dim=-1)[1]
                pos_strided_bbox_preds[mask] = self.bbox_coder.decode_yaw(
                    pos_strided_bbox_preds[mask], centers2d_preds,
                    pos_dir_cls_scores, self.dir_offset, cam2img)
            pos_bbox_targets_3d[mask, 6] = torch.atan2(
                centers2d_targets[:, 0] - cam2img[0, 2],
                cam2img[0, 0]) + pos_bbox_targets_3d[mask, 6]

            if self.use_tc:
                pos_strided_bbox_preds[mask] = bbox_to_box3d(
                    pos_decoded_bbox2d_preds[mask],
                    pos_strided_bbox_preds[mask, 3:6],
                    pos_strided_bbox_preds[mask, 6], cam2img)
            else:
                # use predicted depth to re-project the 2.5D centers
                pos_strided_bbox_preds[mask, :3] = points_img2cam(
                    pos_strided_bbox_preds[mask, :3], views[idx])
            pos_bbox_targets_3d[mask, :3] = centers3d_targets

            # depth fixed when computing re-project 3D bboxes
            pos_strided_bbox_preds[mask, 2] = \
                pos_bbox_targets_3d.clone()[mask, 2]

            corners = batch_img_metas[0]['box_type_3d'](
                pos_strided_bbox_preds[mask],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image[mask] = points_cam2img(corners, cam2img)

            corners_gt = batch_img_metas[0]['box_type_3d'](
                pos_bbox_targets_3d[mask, :self.bbox_code_size],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image_gt[mask] = points_cam2img(corners_gt, cam2img)

        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        proj_bbox2d_preds = torch.cat([minxy, maxxy], dim=1)

        outputs = (proj_bbox2d_preds, pos_decoded_bbox2d_preds)

        if with_kpts:
            norm_strides = pos_strides * self.regress_ranges[0][1] / \
                self.strides[0]
            kpts_targets = box_corners_in_image_gt - pos_points[..., None, :]
            kpts_targets = kpts_targets.view(
                (*pos_strided_bbox_preds.shape[:-1], 16))
            kpts_targets /= norm_strides

            outputs += (kpts_targets, )

        return outputs

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
            if self.pred_bbox2d and self.use_tc:
                bbox_pred[:, :2] = (bbox_pred[:, -4:-2] -
                                    bbox_pred[:, -2:]) / 2
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
                if self.pred_bbox2d and not self.use_tc:
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
                    points,
                    bbox_pred2d,
                    max_shape=img_meta['img_shape']
                    if not self.use_tc else None)
                mlvl_bboxes2d.append(bbox_pred2d)

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

        if self.pred_bbox2d and self.use_tc:
            mlvl_bboxes = bbox_to_box3d(mlvl_bboxes2d, mlvl_bboxes[:, 3:6],
                                        mlvl_bboxes[:, 6], cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(img_meta['box_type_3d'](
            mlvl_bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

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
            origin=(0.5, 0.5, 0.5))
        if not self.pred_attrs:
            attrs = None

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        if attrs is not None:
            results.attr_labels = attrs

        results_2d = InstanceData()

        if self.pred_bbox2d and not self.use_tc:
            bboxes2d = nms_results[-1]
            results_2d.bboxes = bboxes2d
            results_2d.scores = scores
            results_2d.labels = labels

        return results, results_2d

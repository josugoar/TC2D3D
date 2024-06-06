import numpy as np
import torch
from mmdet.structures.bbox import bbox_overlaps

from mmdet3d.structures import box3d_to_bbox, center_to_corner_box3d
from mmdet3d.utils import array_converter

CORNERS = [
    (4, 0, 1, 6), (4, 0, 2, 6), (7, 0, 1, 6), (7, 0, 2, 6), (0, 1, 5, 7),
    (0, 1, 6, 7), (3, 1, 5, 7), (3, 1, 6, 7), (5, 4, 0, 2), (5, 4, 3, 2),
    (6, 4, 0, 2), (6, 4, 3, 2), (1, 5, 4, 3), (1, 5, 7, 3), (2, 5, 4, 3),
    (2, 5, 7, 3), (1, 0, 4, 3), (1, 0, 7, 3), (2, 0, 4, 3), (2, 0, 7, 3),
    (5, 1, 0, 2), (5, 1, 3, 2), (6, 1, 0, 2), (6, 1, 3, 2), (0, 4, 5, 7),
    (0, 4, 6, 7), (3, 4, 5, 7), (3, 4, 6, 7), (4, 5, 1, 6), (4, 5, 2, 6),
    (7, 5, 1, 6), (7, 5, 2, 6), (0, 1, 4, 3), (0, 5, 4, 3), (0, 1, 4, 7),
    (0, 5, 4, 7), (1, 4, 0, 2), (1, 5, 0, 2), (1, 4, 0, 3), (1, 5, 0, 3),
    (4, 0, 5, 6), (4, 1, 5, 6), (4, 0, 5, 7), (4, 1, 5, 7), (5, 0, 1, 2),
    (5, 4, 1, 2), (5, 0, 1, 6), (5, 4, 1, 6), (0, 0, 4, 3), (0, 0, 4, 7),
    (0, 4, 4, 3), (0, 4, 4, 7), (1, 1, 0, 2), (1, 1, 0, 2), (1, 0, 0, 3),
    (1, 0, 0, 3), (4, 4, 5, 6), (4, 4, 5, 6), (4, 5, 5, 7), (4, 5, 5, 7),
    (5, 5, 1, 2), (5, 5, 1, 2), (5, 1, 1, 6), (5, 1, 1, 6)
]  # 4x4x4=64


@array_converter(apply_to=('bboxes', 'dims', 'yaw', 'cam2img'))
def bbox_to_box3d(bboxes,
                  dims,
                  yaw,
                  cam2img,
                  use_iou_constraint=False,
                  origin=(0.5, 0.5, 0.5)):
    batch = bboxes.shape[0]

    proj_mat = torch.eye(
        4, dtype=bboxes.dtype,
        device=bboxes.device).reshape(1, 1, 4, 4).repeat(batch, 8, 1, 1)
    proj_mat[:, :, :3, 3] = bboxes.new_tensor(
        center_to_corner_box3d(
            np.zeros((batch, 3)),
            dims.numpy(force=True),
            yaw.numpy(force=True),
            origin=origin))
    proj_mat = cam2img @ proj_mat

    location_best = bboxes.new_zeros((batch, 3))
    threshold_best = bboxes.new_full((batch, 1), torch.inf)
    if use_iou_constraint:
        threshold_best *= -1

    for corners in CORNERS:
        A = proj_mat[:, corners,
                     [0, 1, 0, 1], :3] - proj_mat[:, corners,
                                                  2, :3] * bboxes[:, :, None]
        b = proj_mat[:, corners, 2, 3] * bboxes - proj_mat[:, corners,
                                                           [0, 1, 0, 1], 3]
        location, residual, _, _ = torch.linalg.lstsq(A, b, driver='gels')

        if use_iou_constraint:
            bboxes_3d = torch.cat([location, dims, yaw], dim=1)
            proj_bboxes = bboxes_3d.new_tensor(
                box3d_to_bbox(bboxes_3d.numpy(force=True), cam2img))
            iou = bbox_overlaps(proj_bboxes, bboxes, is_aligned=True)
            mask = iou > threshold_best
            location_best[mask] = location[mask]
            threshold_best[mask] = iou[mask]
        else:
            mask = (residual < threshold_best).squeeze(-1) & (
                location[:, 2] > 0)
            location_best[mask] = location[mask]
            threshold_best[mask] = residual[mask]

    return torch.cat([location_best, dims, yaw], dim=1)

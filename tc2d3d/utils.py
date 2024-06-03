from itertools import permutations

import torch
from mmdet.structures.bbox import bbox_overlaps

from mmdet3d.structures import box3d_to_bbox, corners_nd, rotation_3d_in_axis


def points_img2tc(bboxes, cam2img, dims, yaw, bboxes_3d):
    N = bboxes.shape[0]

    P = cam2img.unsqueeze(-3)
    X = bboxes.new_tensor(corners_nd(dims.numpy(force=True),
                                     origin=(0.5, 1.0, 0.5)))
    RX = torch.eye(4, dtype=bboxes.dtype, device=bboxes.device)
    RX = RX.reshape(1, 1, 4, 4).repeat(N, 8, 1, 1)
    RX[:, :, :3, 3] = rotation_3d_in_axis(X, yaw, axis=1)
    M = P @ RX

    T_best = bboxes.new_zeros((N, 3))
    residual_best = bboxes.new_full((N, 1), torch.inf)
    iou_best = bboxes.new_full((N,), -torch.inf)

    for idxs in combinations:

        A = M[:, idxs, [0, 1, 0, 1], :3] - M[:, idxs, 2, :3] * bboxes[:, :, None]
        b = M[:, idxs, 2, 3] * bboxes - M[:, idxs, [0, 1, 0, 1], 3]

        T, residual, _, _ = torch.linalg.lstsq(A, b, driver='gels')

        # TODO: project and check if inside image
        mask = (residual < residual_best).squeeze(-1) & (T[:, 2] > 0)
        T_best[mask] = T[mask]
        residual_best[mask] = residual[mask]

        # bboxes_3d[:, :3] = T
        # bboxes_2d = bboxes_3d.new_tensor(
        #     box3d_to_bbox(bboxes_3d.numpy(force=True), cam2img))
        # iou = bbox_overlaps(bboxes_2d, bboxes, is_aligned=True)
        # mask = iou > iou_best
        # T_best[mask] = T[mask]
        # iou_best[mask] = iou[mask]

    return T_best


combinations = (
    (4, 0, 1, 6),
    (4, 0, 2, 6),
    (7, 0, 1, 6),
    (7, 0, 2, 6),
    (0, 1, 5, 7),
    (0, 1, 6, 7),
    (3, 1, 5, 7),
    (3, 1, 6, 7),
    (5, 4, 0, 2),
    (5, 4, 3, 2),
    (6, 4, 0, 2),
    (6, 4, 3, 2),
    (1, 5, 4, 3),
    (1, 5, 7, 3),
    (2, 5, 4, 3),
    (2, 5, 7, 3),
    (1, 0, 4, 3),
    (1, 0, 7, 3),
    (2, 0, 4, 3),
    (2, 0, 7, 3),
    (5, 1, 0, 2),
    (5, 1, 3, 2),
    (6, 1, 0, 2),
    (6, 1, 3, 2),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
    (3, 4, 5, 7),
    (3, 4, 6, 7),
    (4, 5, 1, 6),
    (4, 5, 2, 6),
    (7, 5, 1, 6),
    (7, 5, 2, 6),
    (0, 1, 4, 3),
    (0, 5, 4, 3),
    (0, 1, 4, 7),
    (0, 5, 4, 7),
    (1, 4, 0, 2),
    (1, 5, 0, 2),
    (1, 4, 0, 3),
    (1, 5, 0, 3),
    (4, 0, 5, 6),
    (4, 1, 5, 6),
    (4, 0, 5, 7),
    (4, 1, 5, 7),
    (5, 0, 1, 2),
    (5, 4, 1, 2),
    (5, 0, 1, 6),
    (5, 4, 1, 6),
    (0, 0, 4, 3),
    (0, 0, 4, 7),
    (0, 4, 4, 3),
    (0, 4, 4, 7),
    (1, 1, 0, 2),
    (1, 1, 0, 2),
    (1, 0, 0, 3),
    (1, 0, 0, 3),
    (4, 4, 5, 6),
    (4, 4, 5, 6),
    (4, 5, 5, 7),
    (4, 5, 5, 7),
    (5, 5, 1, 2),
    (5, 5, 1, 2),
    (5, 1, 1, 6),
    (5, 1, 1, 6)
)

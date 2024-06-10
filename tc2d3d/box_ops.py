from typing import Tuple, Union
import numpy as np
import torch
from torch import Tensor

from mmdet3d.structures import center_to_corner_box3d
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
]


@array_converter(apply_to=('bboxes', 'dims', 'yaw', 'cam2img'))
def bbox_to_box3d(
    bboxes: Union[Tensor, np.ndarray],
    dims: Union[Tensor, np.ndarray],
    yaw: Union[Tensor, np.ndarray],
    cam2img: Union[Tensor, np.ndarray],
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> Union[Tensor, np.ndarray]:
    num_bboxes = bboxes.shape[0]

    proj_mat = torch.eye(
        4, dtype=bboxes.dtype,
        device=bboxes.device).reshape(1, 1, 4, 4).repeat(num_bboxes, 8, 1, 1)
    proj_mat[:, :, :3, 3] = bboxes.new_tensor(
        center_to_corner_box3d(
            np.zeros((num_bboxes, 3)),
            dims.numpy(force=True),
            yaw.numpy(force=True),
            origin=origin))
    proj_mat = cam2img @ proj_mat

    A = proj_mat[:, CORNERS,
                 [0, 1, 0, 1], :3] - proj_mat[:, CORNERS,
                                              2, :3] * bboxes.reshape(
                                                  num_bboxes, 1, 4, 1)
    b = proj_mat[:, CORNERS, 2, 3] * bboxes.reshape(
        num_bboxes, 1, 4) - proj_mat[:, CORNERS, [0, 1, 0, 1], 3]
    location, residual, _, _ = torch.linalg.lstsq(A, b, driver='gels')

    mask = residual.argmin(dim=1).squeeze(1)
    indices = mask.new_tensor(list(range(len(mask))))
    location = location[indices, mask]

    return torch.cat([location, dims, yaw.unsqueeze(1)], dim=1)

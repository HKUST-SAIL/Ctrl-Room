import torch
import numpy as np

from mmcv.ops import box_iou_rotated


def height_overlaps(boxes1, boxes2) -> torch.Tensor:
    """Calculate height overlaps of two boxes.

    Note:
        This function calculates the height overlaps between ``boxes1`` and
        ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

    Args:
        boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
        boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.

    Returns:
        Tensor: Calculated height overlap of the boxes.
    """
    # assert isinstance(boxes1, BaseInstance3DBoxes)
    # assert isinstance(boxes2, BaseInstance3DBoxes)
    # assert type(boxes1) == type(boxes2), \
    #     '"boxes1" and "boxes2" should be in the same type, ' \
    #     f'but got {type(boxes1)} and {type(boxes2)}.'

    boxes1_top_height = (boxes1[:, 2] + 0.5 * boxes1[:, 5]).view(-1, 1)
    boxes1_bottom_height = (boxes1[:, 2] - 0.5 * boxes1[:, 5]).view(-1, 1)
    boxes2_top_height = (boxes2[:, 2] + 0.5 * boxes2[:, 5]).view(1, -1)
    boxes2_bottom_height = (boxes2[:, 2] - 0.5 * boxes2[:, 5]).view(1, -1)

    heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
    lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
    overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
    return overlaps_h


def bdb3d_iou(boxes1_3d: torch.Tensor, boxes2_3d: torch.Tensor, b_aligned=False):
    """calculated 3d iou. assume the 3d bounding boxes are only rotated around z axis

    Args:
        boxes1_3d (torch.Tensor): (N, 3+3+1),  (x,y,z,w,h,l,alpha)
        boxes2_3d (torch.Tensor): (M, 3+3+1),  (x,y,z,w,h,l,alpha)
    """

    rows = len(boxes1_3d)
    cols = len(boxes2_3d)
    if rows * cols == 0:
        return torch.zeros((rows, cols), dtype=torch.float32, device=boxes1_3d.device)

    # height overlap
    overlaps_h = height_overlaps(boxes1_3d, boxes2_3d)

    # Restrict the min values of W and H to avoid memory overflow in
    # ``box_iou_rotated``.
    boxes1_bev = boxes1_3d[:, [0, 1, 3, 4, 6]]  # 2d box
    boxes2_bev = boxes2_3d[:, [0, 1, 3, 4, 6]]
    # boxes1_bev[:, 2:4] = boxes1_bev[:, 2:4].clamp(min=1e-4)
    # boxes2_bev[:, 2:4] = boxes2_bev[:, 2:4].clamp(min=1e-4)

    # bev overlap
    iou2d = box_iou_rotated(boxes1_bev, boxes2_bev, mode='iou', aligned=b_aligned)
    areas1 = (boxes1_bev[:, 2] * boxes1_bev[:, 3]).unsqueeze(1).expand(rows, cols)
    areas2 = (boxes2_bev[:, 2] * boxes2_bev[:, 3]).unsqueeze(0).expand(rows, cols)
    overlaps_bev = iou2d * (areas1 + areas2) / (1 + iou2d)

    # 3d overlaps
    overlaps_3d = overlaps_bev.to(boxes1_3d.device) * overlaps_h

    volume1 = (boxes1_3d[:, 3] * boxes1_3d[:, 4] * boxes1_3d[:, 5]).view(-1, 1)
    volume2 = (boxes2_3d[:, 3] * boxes2_3d[:, 4] * boxes2_3d[:, 5]).view(1, -1)

    # the clamp func is used to avoid division of 0
    iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-8)

    return iou3d

"""Geometry and filtering helpers for multitask inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    box_xyxy: Tuple[int, int, int, int]
    score: float


def nms_ars(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.5) -> Tensor:
    """Class-agnostic hard NMS used as the current ARS filtering step."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    order = scores.argsort(descending=True)
    kept: List[Tensor] = []
    while order.numel():
        current = order[0]
        reference = boxes[current]
        kept.append(reference)
        if order.numel() == 1:
            break
        rest = order[1:]
        inter_left = torch.maximum(reference[0], boxes[rest, 0])
        inter_top = torch.maximum(reference[1], boxes[rest, 1])
        inter_right = torch.minimum(reference[2], boxes[rest, 2])
        inter_bottom = torch.minimum(reference[3], boxes[rest, 3])
        inter = (inter_right - inter_left).clamp_min(0) * (inter_bottom - inter_top).clamp_min(0)
        area_ref = (reference[2] - reference[0]).clamp_min(0) * (reference[3] - reference[1]).clamp_min(0)
        area_rest = (boxes[rest, 2] - boxes[rest, 0]).clamp_min(0) * (boxes[rest, 3] - boxes[rest, 1]).clamp_min(0)
        iou = inter / (area_ref + area_rest - inter).clamp_min(1e-6)
        order = rest[iou <= iou_threshold]
    return torch.stack(kept)


def box_ioa(cell: Sequence[float], box: Sequence[float]) -> float:
    """Intersection over cell area, matching the requested Cell-to-box IoA."""
    left = max(cell[0], box[0])
    top = max(cell[1], box[1])
    right = min(cell[2], box[2])
    bottom = min(cell[3], box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = max(0.0, cell[2] - cell[0]) * max(0.0, cell[3] - cell[1])
    return intersection / area if area else 0.0


def select_cells(
    grid_prob: Tensor,
    boxes_xyxy: Tensor,
    image_width: int,
    image_height: int,
    threshold: float = 0.5,
    ioa_threshold: float = 0.1,
) -> List[GridCell]:
    """Select positive grid cells that overlap at least one detection box."""
    if grid_prob.ndim != 2:
        raise ValueError("grid_prob must have shape (grid_h, grid_w)")
    gh, gw = grid_prob.shape
    cells: List[GridCell] = []
    for row, col in (grid_prob >= threshold).nonzero(as_tuple=False).tolist():
        x1 = round(col * image_width / gw)
        y1 = round(row * image_height / gh)
        x2 = round((col + 1) * image_width / gw)
        y2 = round((row + 1) * image_height / gh)
        cell = (x1, y1, x2, y2)
        if boxes_xyxy.numel() and max(box_ioa(cell, box.tolist()) for box in boxes_xyxy) >= ioa_threshold:
            cells.append(GridCell(row, col, cell, float(grid_prob[row, col])))
    return cells


def restore_grid_from_letterbox(
    grid_prob: Tensor,
    original_size: Tuple[int, int],
    image_size: int,
    ratio: float,
    pad_x: int,
    pad_y: int,
) -> Tensor:
    """Remove LetterBox padding and resize the valid region to grid size."""
    height, width = original_size
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    left = round(pad_x * grid_prob.shape[1] / image_size)
    top = round(pad_y * grid_prob.shape[0] / image_size)
    right = min(
        grid_prob.shape[1],
        left + max(1, round(resized_width * grid_prob.shape[1] / image_size)),
    )
    bottom = min(
        grid_prob.shape[0],
        top + max(1, round(resized_height * grid_prob.shape[0] / image_size)),
    )
    valid = grid_prob[top:bottom, left:right]
    if valid.numel() == 0:
        raise ValueError("LetterBox valid region does not overlap grid map")
    return torch.nn.functional.interpolate(
        valid[None, None],
        size=grid_prob.shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]

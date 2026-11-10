"""End-to-end grid filtering, DeepLab patch segmentation, and PCI output."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from multitask.dataset_multitask import letterbox_image
from multitask.post_process import nms_ars, select_cells
from multitask.post_process import restore_grid_from_letterbox


def calculate_pci(grid_prob: torch.Tensor, cell_area: float = 1.0) -> float:
    """Return a simple distress-density PCI proxy from the binary grid result."""
    distress_ratio = float((grid_prob >= 0.5).float().mean())
    return max(0.0, 100.0 - 100.0 * distress_ratio * cell_area)


class InferencePipeline:
    def __init__(
        self,
        multitask_model: torch.nn.Module,
        deeplab_model: torch.nn.Module,
        device: str | torch.device = "cpu",
        grid_threshold: float = 0.5,
        ioa_threshold: float = 0.1,
        segmentation_size: int = 256,
    ) -> None:
        self.device = torch.device(device)
        self.multitask_model = multitask_model.to(self.device).eval()
        self.deeplab_model = deeplab_model.to(self.device).eval()
        self.grid_threshold = grid_threshold
        self.ioa_threshold = ioa_threshold
        self.segmentation_size = segmentation_size

    @torch.inference_mode()
    def __call__(self, image_path: str | Path) -> Dict[str, object]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        letterboxed, ratio, pad_x, pad_y, _ = letterbox_image(image, 640)
        array = np.asarray(letterboxed, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        result = self.multitask_model(input_tensor)
        grid_prob = restore_grid_from_letterbox(
            result["grid_logits"].sigmoid()[0].cpu(),
            (height, width),
            640,
            ratio,
            pad_x,
            pad_y,
        )
        boxes = self._decode_boxes(
            result["detection"], width, height, ratio, pad_x, pad_y
        )
        cells = select_cells(
            grid_prob, boxes, width, height, self.grid_threshold, self.ioa_threshold
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        for cell in cells:
            patch = image.crop(cell.box_xyxy)
            patch_tensor = self._prepare_patch(patch).to(self.device)
            prediction = self.deeplab_model(patch_tensor)
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            patch_mask = prediction.argmax(1)[0].cpu().numpy().astype(np.uint8)
            patch_mask = np.asarray(
                Image.fromarray(patch_mask).resize(
                    (cell.box_xyxy[2] - cell.box_xyxy[0], cell.box_xyxy[3] - cell.box_xyxy[1]),
                    Image.Resampling.NEAREST,
                )
            )
            x1, y1, x2, y2 = cell.box_xyxy
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], patch_mask)
        return {"grid": grid_prob, "cells": cells, "mask": mask, "pci": calculate_pci(grid_prob)}

    def _prepare_patch(self, patch: Image.Image) -> torch.Tensor:
        patch = patch.resize((self.segmentation_size, self.segmentation_size), Image.Resampling.BILINEAR)
        array = np.asarray(patch, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _decode_boxes(
        detection: object,
        width: int,
        height: int,
        ratio: float = 1.0,
        pad_x: int = 0,
        pad_y: int = 0,
    ) -> torch.Tensor:
        if isinstance(detection, tuple):
            detection = detection[0]
        if not isinstance(detection, torch.Tensor) or detection.ndim != 3:
            return torch.zeros((0, 4))
        # Ultralytics Detect inference is (B, 4 + classes, anchors), with
        # decoded xywh boxes in input pixels and sigmoid class scores.
        values = detection[0].detach().cpu()
        if values.shape[0] < 5:
            return torch.zeros((0, 4))
        boxes = values[:4].transpose(0, 1)
        scores = values[4:].max(dim=0).values
        keep = scores >= 0.25
        boxes = boxes[keep]
        if boxes.numel() == 0:
            return torch.zeros((0, 4))
        cx, cy, width_box, height_box = boxes.unbind(1)
        boxes_xyxy = torch.stack(
            (cx - width_box / 2, cy - height_box / 2,
             cx + width_box / 2, cy + height_box / 2), dim=1
        )
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / ratio
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / ratio
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, width)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, height)
        return nms_ars(boxes_xyxy, scores[keep])


def load_deeplab(
    project_dir: str | Path,
    weights: str | Path,
    num_classes: int,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load the existing DeepLab implementation from its project directory."""
    project = str(Path(project_dir).resolve())
    if project not in sys.path:
        sys.path.insert(0, project)
    from nets.deeplabv3_plus import DeepLab

    model = DeepLab(num_classes=num_classes, backbone="mobilenet", pretrained=False)
    checkpoint = torch.load(weights, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=False)
    return model

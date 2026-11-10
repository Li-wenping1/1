"""YOLOv12 backbone with a parallel 20x20 binary grid head."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor, nn


class MultiTaskYOLO(nn.Module):
    """Wrap an Ultralytics ``DetectionModel`` without modifying its source."""

    def __init__(
        self,
        model_cfg: str | Path,
        num_classes: int = 1,
        yolo_root: str | Path = "yolov12",
        grid_size: int = 20,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        root = str(Path(yolo_root).resolve())
        import sys

        if root not in sys.path:
            sys.path.insert(0, root)
        from ultralytics.nn.tasks import DetectionModel

        self.detector = DetectionModel(str(model_cfg), nc=num_classes, verbose=verbose)
        detect_head = self.detector.model[-1]
        channels = int(detect_head.cv2[0][0].conv.in_channels)
        self.grid_size = grid_size
        self.grid_head = nn.Sequential(
            nn.Conv2d(channels, max(32, channels // 4), 3, padding=1),
            nn.BatchNorm2d(max(32, channels // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(32, channels // 4), 1, 1),
        )

    def _forward_backbone(self, x: Tensor) -> tuple[Any, Tensor]:
        outputs = []
        for module in self.detector.model[:-1]:
            if module.f != -1:
                x = outputs[module.f] if isinstance(module.f, int) else [
                    x if index == -1 else outputs[index] for index in module.f
                ]
            x = module(x)
            outputs.append(x if module.i in self.detector.save else None)
        detect_indices = self.detector.model[-1].f
        detection_features = [outputs[index] for index in detect_indices]
        if any(feature is None for feature in detection_features):
            raise RuntimeError("YOLO Detect feature references were not saved")
        return detection_features, detection_features[0]

    def forward(self, x: Tensor) -> Dict[str, Any]:
        detection_features, grid_feature = self._forward_backbone(x)
        detection_output = self.detector.model[-1](list(detection_features))
        grid_logits = self.grid_head(grid_feature)
        grid_logits = nn.functional.interpolate(
            grid_logits, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False
        )
        return {"detection": detection_output, "grid_logits": grid_logits[:, 0]}

    def load_weights(self, weights: str | Path, map_location: str | torch.device = "cpu") -> None:
        checkpoint = torch.load(weights, map_location=map_location)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: {len(missing)} multitask parameters were not loaded")
        if unexpected:
            print(f"Warning: {len(unexpected)} checkpoint parameters were unused")

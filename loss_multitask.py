"""Joint grid and YOLO detection loss."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import Tensor, nn


class MultiTaskLoss(nn.Module):
    """Combine grid BCE/Focal BCE and an injected YOLO detection criterion.

    The Ultralytics criterion is version-specific, so it is supplied by the
    caller.  It receives the raw Detect output and the original batch dict.
    """

    def __init__(
        self,
        detection_loss: Optional[Callable[[object, Dict[str, object]], Tensor]] = None,
        alpha: float = 0.5,
        pos_weight: Optional[Tensor] = None,
        focal_gamma: float = 0.0,
        focal_alpha: Optional[float] = None,
        normalize_losses: bool = False,
        ema_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if focal_alpha is not None and not 0 < focal_alpha < 1:
            raise ValueError("focal_alpha must be in (0, 1)")
        if not 0 < ema_momentum < 1:
            raise ValueError("ema_momentum must be in (0, 1)")
        self.detection_loss = detection_loss
        self.alpha = alpha
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.normalize_losses = normalize_losses
        self.ema_momentum = ema_momentum
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor(1.0))
        self.register_buffer("grid_loss_ema", torch.tensor(1.0))
        self.register_buffer("box_loss_ema", torch.tensor(1.0))

    def forward(
        self,
        grid_logits: Tensor,
        grid_target: Tensor,
        detection_output: object = None,
        batch: Optional[Dict[str, object]] = None,
    ) -> Dict[str, Tensor]:
        grid_target = grid_target.to(device=grid_logits.device, dtype=grid_logits.dtype)
        bce = nn.functional.binary_cross_entropy_with_logits(
            grid_logits,
            grid_target,
            pos_weight=self.pos_weight.to(grid_logits.device),
            reduction="none",
        )
        if self.focal_gamma:
            probability = grid_logits.sigmoid()
            p_t = probability * grid_target + (1 - probability) * (1 - grid_target)
            focal_factor = (1 - p_t).pow(self.focal_gamma)
            if self.focal_alpha is not None:
                alpha_factor = self.focal_alpha * grid_target + (1 - self.focal_alpha) * (1 - grid_target)
                focal_factor = focal_factor * alpha_factor
            grid_loss = (focal_factor * bce).mean()
        else:
            grid_loss = bce.mean()
        if self.detection_loss is None:
            box_loss = grid_loss.new_zeros(())
        else:
            if batch is None:
                raise ValueError("batch is required when detection_loss is configured")
            box_loss = self.detection_loss(detection_output, batch)
            if box_loss.ndim != 0:
                box_loss = box_loss.mean()
        if self.training:
            with torch.no_grad():
                self.grid_loss_ema.lerp_(grid_loss.detach(), 1 - self.ema_momentum)
                self.box_loss_ema.lerp_(box_loss.detach(), 1 - self.ema_momentum)
        grid_component = grid_loss
        box_component = box_loss
        if self.normalize_losses:
            grid_component = grid_loss / self.grid_loss_ema.detach().clamp_min(1e-6)
            box_component = box_loss / self.box_loss_ema.detach().clamp_min(1e-6)
        total = self.alpha * grid_component + (1.0 - self.alpha) * box_component
        return {
            "loss": total,
            "grid_loss": grid_loss,
            "box_loss": box_loss,
            "grid_component": grid_component,
            "box_component": box_component,
        }

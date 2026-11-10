"""Dataset utilities for YOLO boxes and 20x20 binary disease grids."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def letterbox_image(
    image: Image.Image,
    image_size: int,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, int, int, Tuple[int, int]]:
    """Resize an image without changing aspect ratio and add centered padding."""
    width, height = image.size
    ratio = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), fill)
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, ratio, pad_x, pad_y, (resized_width, resized_height)


def _apply_hsv(image: Image.Image, hgain: float, sgain: float, vgain: float) -> Image.Image:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    hsv[..., 0] = (
        hsv[..., 0].astype(np.int16)
        + int(random.uniform(-hgain, hgain) * 255)
    ) % 256
    hsv[..., 1] = np.clip(
        hsv[..., 1].astype(np.float32)
        * random.uniform(max(0.0, 1.0 - sgain), 1.0 + sgain),
        0,
        255,
    ).astype(np.uint8)
    hsv[..., 2] = np.clip(
        hsv[..., 2].astype(np.float32)
        * random.uniform(max(0.0, 1.0 - vgain), 1.0 + vgain),
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _inverse_affine(
    gain: float, translate_x: float, translate_y: float, image_size: int
) -> Tuple[float, float, float, float, float, float]:
    center = image_size / 2.0
    return (
        1.0 / gain,
        0.0,
        center - (center + translate_x) / gain,
        0.0,
        1.0 / gain,
        center - (center + translate_y) / gain,
    )


def _transform_boxes(
    boxes: torch.Tensor,
    gain: float,
    translate_x: float,
    translate_y: float,
    image_size: int,
) -> torch.Tensor:
    if not boxes.numel():
        return boxes
    output = boxes.clone()
    center_x, center_y, width, height = (output[:, 1:] * image_size).unbind(1)
    left = center_x - width / 2
    top = center_y - height / 2
    right = center_x + width / 2
    bottom = center_y + height / 2
    corners_x = torch.stack((left, right, left, right), dim=1) * gain + (
        image_size / 2 * (1 - gain) + translate_x
    )
    corners_y = torch.stack((top, top, bottom, bottom), dim=1) * gain + (
        image_size / 2 * (1 - gain) + translate_y
    )
    new_left = corners_x.min(1).values.clamp(0, image_size)
    new_top = corners_y.min(1).values.clamp(0, image_size)
    new_right = corners_x.max(1).values.clamp(0, image_size)
    new_bottom = corners_y.max(1).values.clamp(0, image_size)
    new_width = new_right - new_left
    new_height = new_bottom - new_top
    keep = (new_width > 1.0) & (new_height > 1.0)
    output[:, 1] = ((new_left + new_right) / 2) / image_size
    output[:, 2] = ((new_top + new_bottom) / 2) / image_size
    output[:, 3] = new_width / image_size
    output[:, 4] = new_height / image_size
    return output[keep]


class MultiTaskDataset(Dataset):
    """Read images, YOLO labels, and one ``.npy`` grid label per image.

    ``image_dir`` and ``label_dir`` follow the standard YOLO layout.  Grid
    files are looked up by image stem in ``grid_dir`` and contain either
    ``(20, 20)`` values or an array whose first two dimensions are 20.
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        grid_dir: str | Path,
        image_size: int = 640,
        grid_size: int = 20,
        augment: bool = False,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        translate: float = 0.1,
        scale: float = 0.5,
        fliplr: float = 0.5,
        mosaic: float = 1.0,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.grid_dir = Path(grid_dir)
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.translate = translate
        self.scale = scale
        self.fliplr = fliplr
        self.mosaic = mosaic
        self.images = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in extensions
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _read_boxes(self, stem: str) -> torch.Tensor:
        path = self.label_dir / f"{stem}.txt"
        if not path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)
        rows: List[List[float]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
            values = [float(value) for value in fields]
            if not 0 <= values[0] == int(values[0]):
                raise ValueError(f"{path}:{line_number}: invalid class id")
            if any(value < 0 or value > 1 for value in values[1:]):
                raise ValueError(f"{path}:{line_number}: normalized values must be in [0, 1]")
            rows.append(values)
        return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)

    def _read_grid(self, stem: str) -> torch.Tensor:
        path = self.grid_dir / f"{stem}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing grid label: {path}")
        grid = np.asarray(np.load(path))
        if grid.shape != (self.grid_size, self.grid_size):
            raise ValueError(
                f"{path}: expected {(self.grid_size, self.grid_size)}, got {grid.shape}"
            )
        return torch.from_numpy((grid > 0).astype(np.float32))

    def _get_single(self, index: int, apply_augment: bool = True) -> Dict[str, object]:
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        original_size = (image.height, image.width)
        original_width, original_height = image.size
        image, ratio, pad_x, pad_y, resized_size = letterbox_image(image, self.image_size)
        boxes = self._read_boxes(path.stem)
        if boxes.numel():
            boxes = boxes.clone()
            boxes[:, 1] = (boxes[:, 1] * original_width * ratio + pad_x) / self.image_size
            boxes[:, 2] = (boxes[:, 2] * original_height * ratio + pad_y) / self.image_size
            boxes[:, 3] = boxes[:, 3] * original_width * ratio / self.image_size
            boxes[:, 4] = boxes[:, 4] * original_height * ratio / self.image_size
        grid = self._read_grid(path.stem)
        grid_width = max(1, round(resized_size[0] * self.grid_size / self.image_size))
        grid_height = max(1, round(resized_size[1] * self.grid_size / self.image_size))
        grid = torch.from_numpy(
            np.asarray(
                Image.fromarray((grid.numpy() * 255).astype(np.uint8)).resize(
                    (grid_width, grid_height), Image.Resampling.NEAREST
                )
            ).copy()
        ).float().div(255)
        letterbox_grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)
        grid_width, grid_height = grid.shape[1], grid.shape[0]
        grid_pad_x = round(pad_x * self.grid_size / self.image_size)
        grid_pad_y = round(pad_y * self.grid_size / self.image_size)
        end_x = min(self.grid_size, grid_pad_x + grid_width)
        end_y = min(self.grid_size, grid_pad_y + grid_height)
        letterbox_grid[grid_pad_y:end_y, grid_pad_x:end_x] = (
            grid[: end_y - grid_pad_y, : end_x - grid_pad_x] > 0
        ).float()
        if apply_augment and self.augment:
            image = _apply_hsv(image, self.hsv_h, self.hsv_s, self.hsv_v)
            gain = random.uniform(1.0 - self.scale, 1.0 + self.scale)
            translate_x = random.uniform(-self.translate, self.translate) * self.image_size
            translate_y = random.uniform(-self.translate, self.translate) * self.image_size
            image = image.transform(
                (self.image_size, self.image_size),
                Image.Transform.AFFINE,
                _inverse_affine(gain, translate_x, translate_y, self.image_size),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(114, 114, 114),
            )
            grid_image = Image.fromarray(
                (letterbox_grid.numpy() * 255).astype(np.uint8)
            )
            grid_image = grid_image.transform(
                (self.grid_size, self.grid_size),
                Image.Transform.AFFINE,
                _inverse_affine(
                    gain, translate_x * self.grid_size / self.image_size,
                    translate_y * self.grid_size / self.image_size, self.grid_size,
                ),
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )
            letterbox_grid = torch.from_numpy(
                np.asarray(grid_image).copy()
            ).float().div(255).round()
            boxes = _transform_boxes(
                boxes, gain, translate_x, translate_y, self.image_size
            )
            if random.random() < self.fliplr:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                letterbox_grid = torch.flip(letterbox_grid, dims=(1,))
                if boxes.numel():
                    boxes[:, 1] = 1.0 - boxes[:, 1]
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return {
            "img": tensor,
            "boxes": boxes,
            "grid": letterbox_grid,
            "path": str(path),
            "original_size": original_size,
            "letterbox": (ratio, pad_x, pad_y),
        }

    def _get_mosaic(self, index: int) -> Dict[str, object]:
        indices = [index] + random.choices(range(len(self.images)), k=3)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        mosaic_grid = torch.zeros(
            (self.grid_size, self.grid_size), dtype=torch.float32
        )
        mosaic_boxes: list[torch.Tensor] = []
        half_size = self.image_size // 2
        half_grid = self.grid_size // 2
        for tile_index, source_index in enumerate(indices):
            sample = self._get_single(source_index, apply_augment=False)
            tile = Image.fromarray(
                (sample["img"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            ).resize((half_size, half_size), Image.Resampling.BILINEAR)
            offset_x = (tile_index % 2) * half_size
            offset_y = (tile_index // 2) * half_size
            canvas.paste(tile, (offset_x, offset_y))

            grid = sample["grid"].numpy()
            grid_tile = Image.fromarray((grid * 255).astype(np.uint8)).resize(
                (half_grid, half_grid), Image.Resampling.NEAREST
            )
            grid_tensor = torch.from_numpy(np.asarray(grid_tile).copy()).float().div(255)
            grid_offset_x = (tile_index % 2) * half_grid
            grid_offset_y = (tile_index // 2) * half_grid
            mosaic_grid[
                grid_offset_y:grid_offset_y + half_grid,
                grid_offset_x:grid_offset_x + half_grid,
            ] = grid_tensor

            boxes = sample["boxes"].clone()
            if boxes.numel():
                boxes[:, 1] = (boxes[:, 1] * half_size + offset_x) / self.image_size
                boxes[:, 2] = (boxes[:, 2] * half_size + offset_y) / self.image_size
                boxes[:, 3] *= half_size / self.image_size
                boxes[:, 4] *= half_size / self.image_size
                mosaic_boxes.append(boxes)

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return {
            "img": tensor,
            "boxes": (
                torch.cat(mosaic_boxes, dim=0)
                if mosaic_boxes
                else torch.zeros((0, 5), dtype=torch.float32)
            ),
            "grid": mosaic_grid,
            "path": str(self.images[index]),
            "original_size": (self.image_size, self.image_size),
            "letterbox": (1.0, 0, 0),
        }

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.augment and random.random() < self.mosaic:
            return self._get_mosaic(index)
        return self._get_single(index)


def multitask_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate variable-length boxes while stacking image and grid tensors."""
    batch_indices = []
    classes = []
    bboxes = []
    for batch_index, item in enumerate(batch):
        boxes = item["boxes"]
        if boxes.numel():
            batch_indices.append(torch.full((boxes.shape[0],), batch_index, dtype=torch.long))
            classes.append(boxes[:, 0])
            bboxes.append(boxes[:, 1:])
    empty = torch.zeros((0,), dtype=torch.float32)
    return {
        "img": torch.stack([item["img"] for item in batch]),
        "grid": torch.stack([item["grid"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "batch_idx": torch.cat(batch_indices) if batch_indices else torch.zeros((0,), dtype=torch.long),
        "cls": torch.cat(classes) if classes else empty,
        "bboxes": torch.cat(bboxes) if bboxes else torch.zeros((0, 4), dtype=torch.float32),
        "path": [item["path"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
        "letterbox": [item["letterbox"] for item in batch],
    }

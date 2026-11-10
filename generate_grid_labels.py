"""Generate 20x20 binary grid labels from YOLO detection labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ===================== Edit these paths and settings =====================
LABELS_DIR = r"D:\LWP_data1\Object_Detection\lab_new"
GRIDS_DIR = r"D:\LWP_data1\Object_Detection\lab_new_grid"
GRID_SIZE = 20
MIN_IOA = 0.0
# ========================================================================


def box_intersection_over_cell(
    cell_x1: float,
    cell_y1: float,
    cell_x2: float,
    cell_y2: float,
    box_x1: float,
    box_y1: float,
    box_x2: float,
    box_y2: float,
) -> float:
    left = max(cell_x1, box_x1)
    top = max(cell_y1, box_y1)
    right = min(cell_x2, box_x2)
    bottom = min(cell_y2, box_y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    cell_area = (cell_x2 - cell_x1) * (cell_y2 - cell_y1)
    return intersection / cell_area if cell_area > 0 else 0.0


def generate_grid(
    label_path: Path,
    grid_size: int = 20,
    min_ioa: float = 0.0,
) -> np.ndarray:
    """Mark cells whose area overlap with any YOLO box reaches ``min_ioa``."""
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if not label_path.exists():
        return grid
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
        _, center_x, center_y, box_width, box_height = map(float, fields)
        if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, box_width, box_height)):
            raise ValueError(f"{label_path}:{line_number}: coordinates must be in [0, 1]")
        box_x1 = max(0.0, center_x - box_width / 2)
        box_y1 = max(0.0, center_y - box_height / 2)
        box_x2 = min(1.0, center_x + box_width / 2)
        box_y2 = min(1.0, center_y + box_height / 2)
        for row in range(grid_size):
            for col in range(grid_size):
                cell_x1, cell_x2 = col / grid_size, (col + 1) / grid_size
                cell_y1, cell_y2 = row / grid_size, (row + 1) / grid_size
                if box_intersection_over_cell(
                    cell_x1, cell_y1, cell_x2, cell_y2,
                    box_x1, box_y1, box_x2, box_y2,
                ) > min_ioa:
                    grid[row, col] = 1
    return grid


def generate_labels(labels_dir: Path, grids_dir: Path, grid_size: int, min_ioa: float) -> int:
    grids_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        np.save(grids_dir / f"{label_path.stem}.npy", generate_grid(label_path, grid_size, min_ioa))
    return len(label_files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=LABELS_DIR, help="YOLO .txt label directory")
    parser.add_argument("--grids", default=GRIDS_DIR, help="Output .npy directory")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument(
        "--min-ioa",
        type=float,
        default=MIN_IOA,
        help="Minimum intersection-over-cell area; 0 marks every touched cell",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    labels_dir = Path(args.labels).resolve()
    if not labels_dir.exists():
        raise FileNotFoundError(
            f"YOLO label directory does not exist: {labels_dir}. "
            "Edit LABELS_DIR at the top of generate_grid_labels.py."
        )
    if not 0.0 <= args.min_ioa <= 1.0:
        raise ValueError("--min-ioa must be in [0, 1]")
    count = generate_labels(
        labels_dir,
        Path(args.grids).resolve(),
        args.grid_size,
        args.min_ioa,
    )
    print(f"Generated {count} grid labels.")

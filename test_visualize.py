"""Run multitask inference on a test image directory and save visualizations.

Edit the absolute paths in the configuration section, then run this file with
the same Python environment used for training.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multitask.dataset_multitask import letterbox_image
from multitask.model_multitask import MultiTaskYOLO
from multitask.post_process import restore_grid_from_letterbox


# ========================= Edit these paths/settings =========================
TEST_IMAGES_DIR = r"D:\MobileNetV4\data_test2_yolo\images\val"
TEST_LABELS_DIR = r"D:\MobileNetV4\data_test2_yolo\labels\val"
TEST_GRIDS_DIR = r"D:\MobileNetV4\data_test2_yolo\grids\val"
WEIGHTS = r"D:\MobileNetV4\weight_multitask\multitask_best_map50.pt"
MODEL_CFG = r"D:\MobileNetV4\yolov12\ultralytics\cfg\models\12\yolo12.yaml"
YOLO_ROOT = r"D:\MobileNetV4\yolov12"
OUTPUT_DIR = r"D:\test_data\output\v5"

IMAGE_SIZE = 640
NUM_CLASSES = 2
GRID_SIZE = 20
GRID_THRESHOLD = 0.5
BOX_CONFIDENCE_THRESHOLD = 0.2
NMS_IOU_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =============================================================================


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    left = torch.maximum(box[0], boxes[:, 0])
    top = torch.maximum(box[1], boxes[:, 1])
    right = torch.minimum(box[2], boxes[:, 2])
    bottom = torch.minimum(box[3], boxes[:, 3])
    intersection = (right - left).clamp_min(0) * (bottom - top).clamp_min(0)
    box_area = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp_min(0)
    return intersection / (box_area + boxes_area - intersection).clamp_min(1e-9)


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Return kept indices after per-class greedy NMS."""
    kept: list[torch.Tensor] = []
    for class_id in classes.unique().tolist():
        class_indices = torch.where(classes == class_id)[0]
        order = class_indices[scores[class_indices].argsort(descending=True)]
        while order.numel():
            current = order[0]
            kept.append(current)
            if order.numel() == 1:
                break
            rest = order[1:]
            overlaps = box_iou_one_to_many(boxes[current], boxes[rest])
            order = rest[overlaps <= iou_threshold]
    if not kept:
        return torch.zeros((0,), dtype=torch.long)
    kept_tensor = torch.stack(kept)
    return kept_tensor[scores[kept_tensor].argsort(descending=True)]


def decode_detections(
    detection: object,
    image_width: int,
    image_height: int,
    ratio: float,
    pad_x: int,
    pad_y: int,
    num_classes: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one image's YOLO output and restore boxes to original pixels."""
    if isinstance(detection, tuple):
        detection = detection[0]
    if not isinstance(detection, torch.Tensor) or detection.ndim != 3:
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    values = detection[0].detach().cpu()
    if values.shape[0] < 4 + num_classes:
        raise ValueError(
            f"Detection output has {values.shape[0]} channels, expected at least "
            f"{4 + num_classes}"
        )
    boxes = values[:4].transpose(0, 1)
    class_scores, classes = values[4 : 4 + num_classes].max(dim=0)
    keep = class_scores >= confidence_threshold
    boxes = boxes[keep]
    scores = class_scores[keep]
    classes = classes[keep]
    if not boxes.numel():
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    center_x, center_y, width, height = boxes.unbind(1)
    boxes = torch.stack(
        (center_x - width / 2, center_y - height / 2,
         center_x + width / 2, center_y + height / 2),
        dim=1,
    )
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_height)
    kept = class_aware_nms(boxes, scores, classes, nms_iou_threshold)
    return boxes[kept], scores[kept], classes[kept]


def grid_color(score: float) -> tuple[int, int, int]:
    """Blue-to-red color for a Grid probability."""
    score = max(0.0, min(1.0, score))
    return (int(255 * score), int(80 * (1 - score)), int(255 * (1 - score)))


def draw_result(
    image: Image.Image,
    grid_prob: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    grid_threshold: float,
) -> Image.Image:
    """Draw Grid heatmap/cells and detection boxes on a copy of the image."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    grid_height, grid_width = grid_prob.shape

    for row in range(grid_height):
        for col in range(grid_width):
            score = float(grid_prob[row, col])
            x1 = round(col * width / grid_width)
            y1 = round(row * height / grid_height)
            x2 = round((col + 1) * width / grid_width)
            y2 = round((row + 1) * height / grid_height)
            if score >= grid_threshold:
                draw.rectangle(
                    (x1, y1, x2, y2),
                    fill=(*grid_color(score), 70),
                    outline=(*grid_color(score), 180),
                    width=1,
                )

    for box, score, class_id in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        x1, y1, x2, y2 = box
        color = (0, 255, 0)
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=3)
        label = f"class {class_id}: {score:.2f}"
        draw.rectangle((x1, max(0, y1 - 20), x1 + 110, y1), fill=(*color, 210))
        draw.text((x1 + 3, max(0, y1 - 18)), label, fill=(0, 0, 0, 255))

    positive = int((grid_prob >= grid_threshold).sum())
    draw.text(
        (10, 10),
        f"Grid positive: {positive}/{grid_height * grid_width} | "
        f"Boxes: {len(boxes)}",
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )
    return canvas


def prepare_input(image: Image.Image) -> tuple[torch.Tensor, float, int, int]:
    letterboxed, ratio, pad_x, pad_y, _ = letterbox_image(image, IMAGE_SIZE)
    array = np.asarray(letterboxed, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, ratio, pad_x, pad_y


def average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))
    for index in range(precision.numel() - 1, 0, -1):
        precision[index - 1] = torch.maximum(precision[index - 1], precision[index])
    changes = torch.where(recall[1:] != recall[:-1])[0]
    return float(((recall[changes + 1] - recall[changes]) * precision[changes + 1]).sum())


def grid_average_precision(
    scores: list[torch.Tensor], targets: list[torch.Tensor]
) -> float:
    if not scores:
        return 0.0
    all_scores = torch.cat(scores).float()
    all_targets = torch.cat(targets).bool()
    positive_count = int(all_targets.sum())
    if positive_count == 0:
        return 0.0
    order = torch.argsort(all_scores, descending=True, stable=True)
    sorted_targets = all_targets[order]
    true_positive = sorted_targets.float().cumsum(0)
    false_positive = (~sorted_targets).float().cumsum(0)
    recall = true_positive / positive_count
    precision = true_positive / (true_positive + false_positive).clamp_min(1e-9)
    return average_precision(recall, precision)


def read_grid_target(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Grid label file does not exist: {path}")
    target = torch.from_numpy(np.load(path)).float()
    if target.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f"{path}: expected {(GRID_SIZE, GRID_SIZE)}, got {tuple(target.shape)}"
        )
    return target


def read_boxes_target(
    path: Path, image_width: int, image_height: int
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not path.exists():
        raise FileNotFoundError(f"Detection label file does not exist: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id, center_x, center_y, box_width, box_height = fields
        cx = float(center_x) * image_width
        cy = float(center_y) * image_height
        width = float(box_width) * image_width
        height = float(box_height) * image_height
        boxes.append([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
        classes.append(int(class_id))
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(classes, dtype=torch.long),
    )


def detection_map50(
    predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    targets: list[tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
) -> float:
    class_detections: dict[int, list[tuple[float, int, torch.Tensor]]] = {
        class_id: [] for class_id in range(num_classes)
    }
    class_ground_truths: dict[int, dict[int, torch.Tensor]] = {
        class_id: {} for class_id in range(num_classes)
    }
    for image_index, ((boxes, scores, classes), (target_boxes, target_classes)) in enumerate(
        zip(predictions, targets)
    ):
        for class_id in range(num_classes):
            class_ground_truths[class_id][image_index] = target_boxes[
                target_classes == class_id
            ]
            for box, score in zip(boxes[classes == class_id], scores[classes == class_id]):
                class_detections[class_id].append(
                    (float(score), image_index, box)
                )

    class_aps: list[float] = []
    for class_id in range(num_classes):
        detections = sorted(class_detections[class_id], key=lambda item: item[0], reverse=True)
        ground_truths = class_ground_truths[class_id]
        total_ground_truth = sum(len(value) for value in ground_truths.values())
        if total_ground_truth == 0:
            continue
        true_positive = torch.zeros(len(detections))
        false_positive = torch.zeros(len(detections))
        matched: dict[int, set[int]] = {}
        for index, (_, image_index, box) in enumerate(detections):
            target = ground_truths[image_index]
            if not len(target):
                false_positive[index] = 1
                continue
            ious = box_iou_one_to_many(box, target)
            best_iou, best_index = ious.max(0)
            used = matched.setdefault(image_index, set())
            if best_iou >= 0.5 and int(best_index) not in used:
                true_positive[index] = 1
                used.add(int(best_index))
            else:
                false_positive[index] = 1
        if detections:
            recall = true_positive.cumsum(0) / total_ground_truth
            precision = true_positive.cumsum(0) / (
                true_positive + false_positive
            ).cumsum(0).clamp_min(1e-9)
            class_aps.append(average_precision(recall, precision))
        else:
            class_aps.append(0.0)
    return sum(class_aps) / len(class_aps) if class_aps else 0.0


@torch.inference_mode()
def run_test() -> None:
    image_dir = Path(TEST_IMAGES_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"Test image directory does not exist: {image_dir}")
    labels_dir = Path(TEST_LABELS_DIR)
    grids_dir = Path(TEST_GRIDS_DIR)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Test label directory does not exist: {labels_dir}")
    if not grids_dir.exists():
        raise FileNotFoundError(f"Test grid directory does not exist: {grids_dir}")
    weights_path = Path(WEIGHTS)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file does not exist: {weights_path}")

    device = torch.device(DEVICE)
    model = MultiTaskYOLO(
        MODEL_CFG,
        NUM_CLASSES,
        YOLO_ROOT,
        grid_size=GRID_SIZE,
        verbose=False,
    ).to(device)
    model.load_weights(WEIGHTS, device)
    model.eval()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "test_metrics.csv"
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {image_dir}")

    grid_scores: list[torch.Tensor] = []
    grid_targets: list[torch.Tensor] = []
    detection_predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    detection_targets: list[tuple[torch.Tensor, torch.Tensor]] = []
    grid_tp = grid_fp = grid_fn = grid_correct = grid_total = 0.0
    fields = [
        "image", "grid_positive", "grid_mean", "grid_max",
        "box_count", "boxes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, image_path in enumerate(image_paths, 1):
            image = Image.open(image_path).convert("RGB")
            original_width, original_height = image.size
            input_tensor, ratio, pad_x, pad_y = prepare_input(image)
            output = model(input_tensor.to(device))
            grid_prob = restore_grid_from_letterbox(
                output["grid_logits"].sigmoid()[0].cpu(),
                (original_height, original_width),
                IMAGE_SIZE,
                ratio,
                pad_x,
                pad_y,
            )
            boxes, scores, classes = decode_detections(
                output["detection"],
                original_width,
                original_height,
                ratio,
                pad_x,
                pad_y,
                NUM_CLASSES,
                BOX_CONFIDENCE_THRESHOLD,
                NMS_IOU_THRESHOLD,
            )
            target_grid = read_grid_target(grids_dir / f"{image_path.stem}.npy") >= 0.5
            grid_predictions = grid_prob >= GRID_THRESHOLD
            grid_tp += float((grid_predictions & target_grid).sum())
            grid_fp += float((grid_predictions & ~target_grid).sum())
            grid_fn += float((~grid_predictions & target_grid).sum())
            grid_correct += float((grid_predictions == target_grid).sum())
            grid_total += float(target_grid.numel())
            grid_scores.append(grid_prob.reshape(-1).cpu())
            grid_targets.append(target_grid.reshape(-1).cpu())
            detection_predictions.append((boxes, scores, classes))
            detection_targets.append(
                read_boxes_target(
                    labels_dir / f"{image_path.stem}.txt",
                    original_width,
                    original_height,
                )
            )
            result = draw_result(
                image, grid_prob, boxes, scores, classes, GRID_THRESHOLD
            )
            result.save(output_dir / f"{image_path.stem}_result.jpg", quality=95)
            box_text = ";".join(
                f"{int(class_id)}:{float(score):.4f}:"
                f"{','.join(str(round(value, 1)) for value in box)}"
                for box, score, class_id in zip(
                    boxes.tolist(), scores.tolist(), classes.tolist()
                )
            )
            writer.writerow({
                "image": str(image_path),
                "grid_positive": int((grid_prob >= GRID_THRESHOLD).sum()),
                "grid_mean": float(grid_prob.mean()),
                "grid_max": float(grid_prob.max()),
                "box_count": len(boxes),
                "boxes": box_text,
            })
            print(f"[{index}/{len(image_paths)}] {image_path.name}")

    grid_precision = grid_tp / max(grid_tp + grid_fp, 1.0)
    grid_recall = grid_tp / max(grid_tp + grid_fn, 1.0)
    grid_f1 = 2 * grid_precision * grid_recall / max(
        grid_precision + grid_recall, 1e-8
    )
    metrics = {
        "IMAGE_SIZE": IMAGE_SIZE,
        "NUM_CLASSES": NUM_CLASSES,
        "GRID_SIZE": GRID_SIZE,
        "GRID_THRESHOLD": GRID_THRESHOLD,
        "BOX_CONFIDENCE_THRESHOLD": BOX_CONFIDENCE_THRESHOLD,
        "NMS_IOU_THRESHOLD": NMS_IOU_THRESHOLD,
        "grid_acc": grid_correct / max(grid_total, 1.0),
        "grid_precision": grid_precision,
        "grid_recall": grid_recall,
        "grid_f1": grid_f1,
        "grid_ap": grid_average_precision(grid_scores, grid_targets),
        "map50": detection_map50(
            detection_predictions, detection_targets, NUM_CLASSES
        ),
    }
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(
            {"metric": name, "value": value} for name, value in metrics.items()
        )
    print(
        "test metrics: "
        + " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    )
    print(f"visualizations: {output_dir}")
    print(f"predictions csv: {csv_path}")
    print(f"metrics csv: {metrics_path}")


if __name__ == "__main__":
    run_test()

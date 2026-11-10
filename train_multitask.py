"""Minimal configurable trainer for the multitask model."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow both `python -m multitask.train_multitask` and direct execution of
# this file by adding the project root to Python's import search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multitask.dataset_multitask import MultiTaskDataset, multitask_collate
from multitask.loss_multitask import MultiTaskLoss
from multitask.model_multitask import MultiTaskYOLO


# ========================= Edit these paths/settings =========================
IMAGES_DIR = r"D:\MobileNetV4\data_test2_yolo\images\train"
LABELS_DIR = r"D:\MobileNetV4\data_test2_yolo\labels\train"
GRIDS_DIR = r"D:\MobileNetV4\data_test2_yolo\grids\train"
VAL_IMAGES_DIR = r"D:\MobileNetV4\data_test2_yolo\images\val"
VAL_LABELS_DIR = r"D:\MobileNetV4\data_test2_yolo\labels\val"
VAL_GRIDS_DIR = r"D:\MobileNetV4\data_test2_yolo\grids\val"
MODEL_CFG = r"D:\MobileNetV4\yolov12\ultralytics\cfg\models\12\yolo12.yaml"
YOLO_ROOT = r"D:\MobileNetV4\yolov12"
INITIAL_WEIGHTS = ""
OUTPUT_WEIGHTS = r"D:\MobileNetV4\weight_multitask\multitask.pt"
METRICS_CSV = r"D:\MobileNetV4\multitask\output\metrics.csv"
MAP_PLOT = r"D:\MobileNetV4\multitask\output\map50_curve.png"

IMAGE_SIZE = 640
NUM_CLASSES = 2
GRID_SIZE = 20
BATCH_SIZE = 4
EPOCHS = 300
WORKERS = 0
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.001
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
ALPHA = 1.0
GRID_LOSS = "bce"  # "bce" or "focal"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None
NORMALIZE_LOSSES = False
LOSS_EMA_MOMENTUM = 0.99
AUTO_POS_WEIGHT = True
POS_WEIGHT = None
GRAD_CLIP_NORM = 10.0
USE_COSINE_SCHEDULER = True
SCHEDULER_MIN_LR = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 300
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
TRANSLATE = 0.1
SCALE = 0.5
FLIPLR = 0.5
MOSAIC = 0
CLOSE_MOSAIC = 0
USE_AUGMENTATION = True
WARMUP_EPOCHS = 0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1
TASK_MODE = "grid"  #grid   #detection  #multitask
# =============================================================================


def move_detection_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, list):
            out[key] = [
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            out[key] = value
    return out


def estimate_grid_pos_weight(dataset: MultiTaskDataset) -> torch.Tensor:
    """Estimate negative/positive cell ratio from the complete training set."""
    positive = 0.0
    total = 0
    for index in range(len(dataset)):
        target = dataset[index]["grid"]
        positive += float(target.sum())
        total += target.numel()
    negative = total - positive
    if positive <= 0:
        raise ValueError("Training Grid labels contain no positive cells")
    return torch.tensor(max(1.0, negative / positive), dtype=torch.float32)


def grid_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    predictions = logits.sigmoid() >= threshold
    targets = targets >= 0.5
    true_positive = (predictions & targets).sum().float()
    false_positive = (predictions & ~targets).sum().float()
    false_negative = (~predictions & targets).sum().float()
    correct = (predictions == targets).sum().float()
    total = targets.numel()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "grid_acc": float(correct / max(1, total)),
        "grid_precision": float(precision),
        "grid_recall": float(recall),
        "grid_f1": float(f1),
    }


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(2)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(1)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp_min(1e-9)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Return indices kept by greedy non-maximum suppression."""
    order = scores.argsort(descending=True)
    kept: list[torch.Tensor] = []
    while order.numel():
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        overlaps = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][overlaps <= iou_threshold]
    return torch.stack(kept) if kept else torch.zeros(0, dtype=torch.long)


def average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))
    for index in range(precision.numel() - 1, 0, -1):
        precision[index - 1] = torch.maximum(precision[index - 1], precision[index])
    sample_points = torch.linspace(0.0, 1.0, 101)
    right_indices = torch.searchsorted(recall, sample_points, right=True).clamp(
        1, recall.numel() - 1
    )
    left_indices = right_indices - 1
    recall_left = recall[left_indices]
    recall_right = recall[right_indices]
    precision_left = precision[left_indices]
    precision_right = precision[right_indices]
    interpolation = precision_left + (
        (sample_points - recall_left)
        / (recall_right - recall_left).clamp_min(1e-12)
        * (precision_right - precision_left)
    )
    interpolation[sample_points >= recall[-1]] = precision[-1]
    return float(interpolation.mean())


def grid_average_precision(
    scores: list[torch.Tensor], targets: list[torch.Tensor]
) -> float:
    """Calculate binary AP after aggregating every validation Grid cell."""
    if not scores:
        return 0.0
    all_scores = torch.cat(scores).float().cpu()
    all_targets = torch.cat(targets).bool().cpu()
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


def detection_map50(
    records: list[tuple[object, dict[str, object], int]],
    num_classes: int,
    image_size: int,
    confidence_threshold: float = 0.001,
    nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    max_det: int = 300,
) -> float:
    """Calculate dataset-level mAP50 from all validation predictions.

    Predictions from every validation batch are accumulated before sorting,
    matching, and AP calculation. This avoids averaging per-batch mAP values,
    which gives batches with few objects the same weight as larger batches.
    """
    class_detections: dict[int, list[tuple[float, int, torch.Tensor]]] = {
        class_id: [] for class_id in range(num_classes)
    }
    class_ground_truths: dict[int, dict[int, torch.Tensor]] = {
        class_id: {} for class_id in range(num_classes)
    }
    for detection, batch, image_offset in records:
        if isinstance(detection, tuple):
            detection = detection[0]
        if not isinstance(detection, torch.Tensor) or detection.ndim != 3:
            continue
        predictions = detection.detach().cpu()
        batch_indices = batch["batch_idx"].detach().cpu()
        labels = batch["cls"].detach().cpu()
        targets = batch["bboxes"].detach().cpu()
        for image_index in range(predictions.shape[0]):
            global_image_index = image_offset + image_index
            image_target_mask = batch_indices == image_index
            image_targets = targets[image_target_mask]
            image_labels = labels[image_target_mask]
            for class_id in range(num_classes):
                target = image_targets[image_labels == class_id]
                if target.numel():
                    center_x, center_y, width, height = (target * image_size).unbind(1)
                    class_ground_truths[class_id][global_image_index] = torch.stack(
                        (center_x - width / 2, center_y - height / 2,
                         center_x + width / 2, center_y + height / 2), dim=1
                    )

                scores = predictions[image_index, 4 + class_id]
                keep = scores >= confidence_threshold
                boxes = predictions[image_index, :4, keep].T
                if boxes.numel():
                    center_x, center_y, width, height = boxes.unbind(1)
                    boxes = torch.stack(
                        (center_x - width / 2, center_y - height / 2,
                         center_x + width / 2, center_y + height / 2), dim=1
                    )
                    keep_indices = nms(boxes, scores[keep], nms_iou_threshold)
                    class_detections[class_id].extend(
                        (float(scores[keep][index]), global_image_index, boxes[index])
                        for index in keep_indices
                    )

    if max_det < 1:
        raise ValueError("max_det must be at least 1")
    image_detections: dict[int, list[tuple[float, int, torch.Tensor]]] = {}
    for class_id, detections in class_detections.items():
        for score, image_index, box in detections:
            image_detections.setdefault(image_index, []).append(
                (score, class_id, box)
            )
    limited_detections = {class_id: [] for class_id in range(num_classes)}
    for image_index, detections in image_detections.items():
        detections.sort(key=lambda item: item[0], reverse=True)
        for score, class_id, box in detections[:max_det]:
            limited_detections[class_id].append((score, image_index, box))

    class_aps = []
    for class_id in range(num_classes):
        detections = limited_detections[class_id]
        ground_truths = class_ground_truths[class_id]
        total_ground_truth = sum(len(value) for value in ground_truths.values())
        if total_ground_truth == 0:
            continue
        detections.sort(key=lambda item: item[0], reverse=True)
        true_positive = torch.zeros(len(detections))
        false_positive = torch.zeros(len(detections))
        matched: dict[int, set[int]] = {}
        for index, (_, image_index, box) in enumerate(detections):
            target = ground_truths.get(image_index, torch.zeros((0, 4)))
            if not len(target):
                false_positive[index] = 1
                continue
            ious = box_iou(box.unsqueeze(0), target)[0]
            best_iou, best_index = ious.max(0)
            used = matched.setdefault(image_index, set())
            if best_iou >= 0.5 and int(best_index) not in used:
                true_positive[index] = 1
                used.add(int(best_index))
            else:
                false_positive[index] = 1
        if detections:
            recall = true_positive.cumsum(0) / total_ground_truth
            precision = true_positive.cumsum(0) / (true_positive + false_positive).cumsum(0).clamp_min(1e-9)
            class_aps.append(average_precision(recall, precision))
        else:
            class_aps.append(0.0)
    return sum(class_aps) / len(class_aps) if class_aps else 0.0


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    task_mode: str,
    threshold: float,
    num_classes: int,
    image_size: int,
    nms_iou_threshold: float,
) -> dict[str, float]:
    if task_mode not in {"multitask", "grid", "detection"}:
        raise ValueError(f"Unsupported task mode: {task_mode}")
    evaluate_grid = task_mode in {"multitask", "grid"}
    evaluate_detection = task_mode in {"multitask", "detection"}
    model.eval()
    criterion.eval()
    totals = {"loss": 0.0, "grid_loss": 0.0, "box_loss": 0.0}
    grid_tp = 0.0
    grid_fp = 0.0
    grid_fn = 0.0
    grid_correct = 0.0
    grid_total = 0.0
    grid_scores: list[torch.Tensor] = []
    grid_targets: list[torch.Tensor] = []
    detection_records: list[tuple[object, dict[str, object], int]] = []
    image_offset = 0
    batches = 0
    progress = tqdm(
        loader,
        desc="Validating",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        images = batch["img"].to(device)
        targets = batch["grid"].to(device)
        detection_batch = move_detection_batch(batch, device)
        output = model(images)
        losses = criterion(output["grid_logits"], targets, output["detection"], detection_batch)
        for name in totals:
            totals[name] += float(losses[name])
        logits = output["grid_logits"]
        predictions = logits.sigmoid() >= threshold
        targets_bin = targets >= 0.5
        grid_tp += float((predictions & targets_bin).sum())
        grid_fp += float((predictions & ~targets_bin).sum())
        grid_fn += float((~predictions & targets_bin).sum())
        grid_correct += float((predictions == targets_bin).sum())
        grid_total += float(targets_bin.numel())
        grid_scores.append(logits.sigmoid().detach().reshape(-1).cpu())
        grid_targets.append(targets_bin.detach().reshape(-1).cpu())
        if evaluate_detection:
            detection_records.append((output["detection"], detection_batch, image_offset))
        image_offset += images.shape[0]
        batches += 1
        progress.set_postfix(
            loss=f"{float(losses['loss']):.4f}",
            grid=f"{float(losses['grid_loss']):.4f}",
            box=f"{float(losses['box_loss']):.4f}",
            map50="pending",
        )
    if not batches:
        raise RuntimeError("Validation dataset is empty")
    dataset_map50 = (
        detection_map50(
            detection_records, num_classes, image_size,
            nms_iou_threshold=nms_iou_threshold,
        )
        if evaluate_detection
        else 0.0
    )
    precision = grid_tp / max(grid_tp + grid_fp, 1.0)
    recall = grid_tp / max(grid_tp + grid_fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    grid_ap = grid_average_precision(grid_scores, grid_targets)
    validation_metrics = {
        **{name: value / batches for name, value in totals.items()},
        "grid_acc": grid_correct / max(grid_total, 1.0) if evaluate_grid else 0.0,
        "grid_precision": precision if evaluate_grid else 0.0,
        "grid_recall": recall if evaluate_grid else 0.0,
        "grid_f1": f1 if evaluate_grid else 0.0,
        "grid_ap": grid_ap if evaluate_grid else 0.0,
        "map50": dataset_map50,
    }
    model.train()
    criterion.train()
    return validation_metrics


def plot_map_curve(history: list[tuple[int, float]], output_path: str) -> None:
    if not history:
        raise ValueError("Cannot plot mAP curve without validation history")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to draw the mAP curve") from error
    epochs, maps = zip(*history)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, maps, marker="o", label="mAP50")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("mAP50")
    axis.set_title("Validation mAP50 by Epoch")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    if args.task_mode not in {"multitask", "grid", "detection"}:
        raise ValueError(f"Unsupported task mode: {args.task_mode}")
    if args.patience < 1:
        raise ValueError("patience must be at least 1")
    if not 0 <= args.nms_iou_threshold <= 1:
        raise ValueError("nms_iou_threshold must be between 0 and 1")
    for name in (
        "images", "labels", "grids", "val_images", "val_labels", "val_grids",
        "model_cfg", "yolo_root",
    ):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(
                f"{name} path does not exist: {path}. "
                "Edit the absolute paths at the top of train_multitask.py."
            )
    device = torch.device(args.device)
    dataset = MultiTaskDataset(
        args.images, args.labels, args.grids, args.image_size, args.grid_size,
        augment=args.use_augmentation,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=multitask_collate, pin_memory=device.type == "cuda",
    )
    validation_dataset = MultiTaskDataset(
        args.val_images, args.val_labels, args.val_grids, args.image_size,
        args.grid_size, augment=False,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=multitask_collate,
        pin_memory=device.type == "cuda",
    )
    model = MultiTaskYOLO(
        args.model_cfg,
        args.num_classes,
        args.yolo_root,
        grid_size=args.grid_size,
        verbose=False,
    ).to(device)
    if args.weights:
        model.load_weights(args.weights, device)
    yolo_criterion = None
    if args.task_mode in {"multitask", "detection"}:
        try:
            from ultralytics.cfg import get_cfg
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics and its dependencies are required for detection loss"
            ) from error
        model.detector.args = get_cfg(overrides={"imgsz": args.image_size})
        yolo_criterion = model.detector.init_criterion()
    if args.task_mode in {"multitask", "grid"} and args.auto_pos_weight:
        pos_weight = estimate_grid_pos_weight(dataset)
    elif args.task_mode in {"multitask", "grid"} and args.pos_weight is not None:
        pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32)
    else:
        pos_weight = torch.tensor(1.0, dtype=torch.float32)
    print(f"grid pos_weight={float(pos_weight):.4f}")

    def detection_loss(prediction: object, detection_batch: dict[str, object]) -> torch.Tensor:
        if yolo_criterion is None:
            raise RuntimeError("Detection loss is disabled for this task mode")
        result = yolo_criterion(prediction, detection_batch)
        loss = result[0] if isinstance(result, tuple) else result
        if loss.ndim != 1:
            raise RuntimeError(
                f"Expected Ultralytics detection loss vector, got shape {tuple(loss.shape)}"
            )
        images = detection_batch.get("img")
        if not isinstance(images, torch.Tensor) or images.ndim == 0:
            raise RuntimeError("Detection batch must contain a batched 'img' tensor")
        return loss.sum() / images.shape[0]

    criterion = MultiTaskLoss(
        detection_loss=(
            detection_loss if args.task_mode in {"multitask", "detection"} else None
        ),
        alpha=(
            args.alpha if args.task_mode == "multitask"
            else 1.0 if args.task_mode == "grid"
            else 0.0
        ),
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma if args.grid_loss == "focal" else 0.0,
        focal_alpha=args.focal_alpha,
        normalize_losses=args.normalize_losses,
        ema_momentum=args.loss_ema_momentum,
    ).to(device)
    initial_learning_rate = (
        args.learning_rate * args.warmup_bias_lr
        if args.warmup_epochs > 0
        else args.learning_rate
    )
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=initial_learning_rate,
            momentum=args.warmup_momentum if args.warmup_epochs > 0 else args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=initial_learning_rate,
            weight_decay=args.weight_decay,
        )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.scheduler_min_lr
        )
        if args.use_cosine_scheduler
        else None
    )
    if scheduler is not None and args.warmup_epochs > 0:
        scheduler.base_lrs = [args.learning_rate] * len(optimizer.param_groups)
    best_val_loss = float("inf")
    best_grid_f1 = float("-inf")
    best_map50 = float("-inf")
    epochs_without_improvement = 0
    monitor_name = "grid_f1" if args.task_mode == "grid" else "map50"            ######### #早停
    metrics_path = Path(args.metrics_csv)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = metrics_path.with_name(
        f"{metrics_path.stem}_{run_timestamp}{metrics_path.suffix}"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "phase", "epoch", "batch", "loss", "grid_loss", "box_loss",
        "grid_acc", "grid_precision", "grid_recall", "grid_f1", "map50",
        "grid_ap", "learning_rate",
    ]
    map_history: list[tuple[int, float]] = []
    warmup_steps = max(1, round(args.warmup_epochs * len(loader)))
    global_step = 0
    with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
        metrics_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        metrics_writer.writeheader()
        for epoch in range(args.epochs):
            dataset.mosaic = (
                0.0 if args.close_mosaic and epoch >= args.epochs - args.close_mosaic
                else args.mosaic
            )
            model.train()
            total = 0.0
            train_batches = 0
            progress = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch in progress:
                if global_step < warmup_steps:
                    warmup_progress = (global_step + 1) / warmup_steps
                    current_learning_rate = initial_learning_rate + (
                        args.learning_rate - initial_learning_rate
                    ) * warmup_progress
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = current_learning_rate
                        if args.optimizer == "SGD":
                            parameter_group["momentum"] = (
                                args.warmup_momentum
                                + (args.momentum - args.warmup_momentum)
                                * warmup_progress
                            )
                global_step += 1
                images = batch["img"].to(device)
                targets = batch["grid"].to(device)
                detection_batch = move_detection_batch(batch, device)
                output = model(images)
                losses = criterion(
                    output["grid_logits"], targets, output["detection"], detection_batch
                )
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                total += float(losses["loss"].detach())
                train_batches += 1
                memory = (
                    f"{torch.cuda.memory_reserved(device) / 1e9:.2f}G"
                    if device.type == "cuda" and torch.cuda.is_available()
                    else "CPU"
                )
                progress.set_postfix(
                    mem=memory,
                    loss=f"{float(losses['loss'].detach()):.4f}",
                    grid=f"{float(losses['grid_loss'].detach()):.4f}",
                    box=f"{float(losses['box_loss'].detach()):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
            if not train_batches:
                raise RuntimeError("Training dataset is empty")
            validation = validate(
                model, validation_loader, criterion, device, args.task_mode,
                args.grid_threshold, args.num_classes, args.image_size,
                args.nms_iou_threshold,
            )
            metrics_writer.writerow({
                "phase": "epoch",
                "epoch": epoch + 1,
                "batch": "",
                "loss": validation["loss"],
                "grid_loss": validation["grid_loss"],
                "box_loss": validation["box_loss"],
                "grid_acc": validation["grid_acc"],
                "grid_precision": validation["grid_precision"],
                "grid_recall": validation["grid_recall"],
                "grid_f1": validation["grid_f1"],
                "grid_ap": validation["grid_ap"],
                "map50": validation["map50"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
            csv_file.flush()
            train_loss = total / train_batches
            map_history.append((epoch + 1, validation["map50"]))
            print(
                f"epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.6f} val_loss={validation['loss']:.6f} "
                f"map50={validation['map50']:.4f} "
                f"grid_acc={validation['grid_acc']:.4f} "
                f"grid_precision={validation['grid_precision']:.4f} "
                f"grid_recall={validation['grid_recall']:.4f} "
                f"grid_f1={validation['grid_f1']:.4f} "
                f"grid_ap={validation['grid_ap']:.4f}"
            )
            if scheduler is not None:
                scheduler.step()
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch + 1,
                "metrics": validation,
            }
            last_path = Path(args.output).with_name(
                f"{Path(args.output).stem}_last.pt"
            )
            torch.save(checkpoint, last_path)
            previous_best_monitor = (
                best_grid_f1 if monitor_name == "grid_f1" else best_map50
            )
            if validation["loss"] < best_val_loss:
                best_val_loss = validation["loss"]
                best_path = Path(args.output).with_name(
                    f"{Path(args.output).stem}_best_loss.pt"
                )
                torch.save(checkpoint, best_path)
            if validation["grid_f1"] > best_grid_f1:
                best_grid_f1 = validation["grid_f1"]
                best_path = Path(args.output).with_name(
                    f"{Path(args.output).stem}_best_f1.pt"
                )
                torch.save(checkpoint, best_path)
            if validation["map50"] > best_map50:
                best_map50 = validation["map50"]
                best_path = Path(args.output).with_name(
                    f"{Path(args.output).stem}_best_map50.pt"
                )
                torch.save(checkpoint, best_path)
            monitor_value = validation[monitor_name]
            if monitor_value > previous_best_monitor:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"early stopping at epoch {epoch + 1}: "
                    f"{monitor_name} did not improve for {args.patience} epochs"
                )
                break
    plot_path = Path(args.map_plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_map_curve(map_history, str(plot_path))
    print(f"metrics csv: {csv_path}")
    print(f"mAP50 curve: {plot_path}")
    print(
        "weights: "
        f"{Path(args.output).with_name(f'{Path(args.output).stem}_last.pt')}, "
        f"{Path(args.output).with_name(f'{Path(args.output).stem}_best_loss.pt')}, "
        f"{Path(args.output).with_name(f'{Path(args.output).stem}_best_f1.pt')}, "
        f"{Path(args.output).with_name(f'{Path(args.output).stem}_best_map50.pt')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-mode",
        choices=("multitask", "grid", "detection"),
        default=TASK_MODE,
        help="Train both heads, only the Grid head, or only the detection head.",
    )
    parser.add_argument("--images", default=IMAGES_DIR)
    parser.add_argument("--labels", default=LABELS_DIR)
    parser.add_argument("--grids", default=GRIDS_DIR)
    parser.add_argument("--val-images", default=VAL_IMAGES_DIR)
    parser.add_argument("--val-labels", default=VAL_LABELS_DIR)
    parser.add_argument("--val-grids", default=VAL_GRIDS_DIR)
    parser.add_argument("--model-cfg", default=MODEL_CFG)
    parser.add_argument("--yolo-root", default=YOLO_ROOT)
    parser.add_argument("--weights", default=INITIAL_WEIGHTS or None)
    parser.add_argument("--output", default=OUTPUT_WEIGHTS)
    parser.add_argument("--metrics-csv", default=METRICS_CSV)
    parser.add_argument("--map-plot", default=MAP_PLOT)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOP_PATIENCE,
        help="Stop when validation grid_f1 does not improve for this many epochs.",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--optimizer", choices=("SGD", "AdamW"), default=OPTIMIZER)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=float, default=WARMUP_EPOCHS)
    parser.add_argument("--warmup-momentum", type=float, default=WARMUP_MOMENTUM)
    parser.add_argument("--warmup-bias-lr", type=float, default=WARMUP_BIAS_LR)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--grid-loss", choices=("bce", "focal"), default=GRID_LOSS)
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--focal-alpha", type=float, default=FOCAL_ALPHA)
    parser.add_argument("--normalize-losses", action="store_true", default=NORMALIZE_LOSSES)
    parser.add_argument("--loss-ema-momentum", type=float, default=LOSS_EMA_MOMENTUM)
    parser.add_argument("--auto-pos-weight", action="store_true", default=AUTO_POS_WEIGHT)
    parser.add_argument("--no-auto-pos-weight", action="store_false", dest="auto_pos_weight")
    parser.add_argument("--pos-weight", type=float, default=POS_WEIGHT)
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument(
        "--use-cosine-scheduler",
        action="store_true",
        default=USE_COSINE_SCHEDULER,
    )
    parser.add_argument(
        "--no-cosine-scheduler",
        action="store_false",
        dest="use_cosine_scheduler",
    )
    parser.add_argument("--scheduler-min-lr", type=float, default=SCHEDULER_MIN_LR)
    parser.add_argument("--grid-threshold", type=float, default=GRID_THRESHOLD)
    parser.add_argument("--hsv-h", type=float, default=HSV_H)
    parser.add_argument("--hsv-s", type=float, default=HSV_S)
    parser.add_argument("--hsv-v", type=float, default=HSV_V)
    parser.add_argument("--translate", type=float, default=TRANSLATE)
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument("--fliplr", type=float, default=FLIPLR)
    parser.add_argument("--mosaic", type=float, default=MOSAIC)
    parser.add_argument("--close-mosaic", type=int, default=CLOSE_MOSAIC)
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        default=USE_AUGMENTATION,
        help="Enable training data augmentation.",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_false",
        dest="use_augmentation",
        help="Disable HSV, affine, flip, and Mosaic augmentation.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=NMS_IOU_THRESHOLD,
        help="IoU threshold used by validation NMS before mAP50 calculation.",
    )
    parser.add_argument("--device", default=DEVICE)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())

"""
Faster R-CNN baseline for Global Wheat Detection.

Usage example:
python faster_rcnn_wheat.py ^
  --data-dir "C:\\path\\to\\global-wheat-detection" ^
  --epochs 10 ^
  --batch-size 2
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torchvision.models import ResNet50_Weights, ResNet101_Weights, resnet101
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes


def make_tqdm(*args, **kwargs):
    # Colab `!python` runs in a non-interactive stream where tqdm redraw control
    # characters are printed as many lines. Disable bars there for clean logs.
    kwargs.setdefault("disable", not sys.stdout.isatty())
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


def safe_tqdm_write(msg: str) -> None:
    # In some notebook/runpy contexts, tqdm.write can crash due to stale/disposed notebook bars.
    # Fallback to plain print keeps training running and still shows logs.
    try:
        tqdm.write(msg)
    except Exception:
        print(msg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_bbox_list(bbox_series: pd.Series) -> List[List[float]]:
    boxes: List[List[float]] = []
    for raw in bbox_series:
        # Kaggle format example: "[834.0, 222.0, 56.0, 36.0]"
        x, y, w, h = ast.literal_eval(raw)
        x2, y2 = x + w, y + h
        boxes.append([x, y, x2, y2])
    return boxes


@dataclass
class SplitData:
    train_ids: List[str]
    val_ids: List[str]


def build_split(image_ids: List[str], val_ratio: float, seed: int) -> SplitData:
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = max(1, int(len(ids) * val_ratio))
    return SplitData(train_ids=ids[val_count:], val_ids=ids[:val_count])


class WheatDataset(Dataset):
    def __init__(
        self,
        annotations_df: pd.DataFrame,
        image_dir: Path,
        image_ids: List[str],
        augment: bool = False,
        scale_jitter: bool = False,
        random_crop: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.augment = augment
        self.scale_jitter = scale_jitter
        self.random_crop = random_crop
        self.grouped = annotations_df.groupby("image_id")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _hflip_boxes(self, boxes: torch.Tensor, width: int) -> torch.Tensor:
        flipped = boxes.clone()
        flipped[:, [0, 2]] = width - boxes[:, [2, 0]]
        return flipped

    def _vflip_boxes(self, boxes: torch.Tensor, height: int) -> torch.Tensor:
        flipped = boxes.clone()
        flipped[:, [1, 3]] = height - boxes[:, [3, 1]]
        return flipped

    @staticmethod
    def _clip_and_filter_boxes(boxes: torch.Tensor, width: int, height: int, min_size: float = 2.0) -> torch.Tensor:
        boxes = boxes.clone()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
        keep = (boxes[:, 2] - boxes[:, 0] >= min_size) & (boxes[:, 3] - boxes[:, 1] >= min_size)
        return boxes[keep]

    def _apply_scale_jitter(self, image_tensor: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.scale_jitter:
            return image_tensor, boxes
        scale = random.uniform(0.8, 1.2)
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        new_h = max(64, int(round(h * scale)))
        new_w = max(64, int(round(w * scale)))
        image_tensor = F.resize(image_tensor, [new_h, new_w], antialias=True)
        sx = new_w / w
        sy = new_h / h
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        return image_tensor, boxes

    def _apply_random_crop(self, image_tensor: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.random_crop:
            return image_tensor, boxes
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        crop_ratio = random.uniform(0.8, 1.0)
        crop_h = int(round(h * crop_ratio))
        crop_w = int(round(w * crop_ratio))
        if crop_h >= h or crop_w >= w:
            return image_tensor, boxes
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        image_tensor = F.crop(image_tensor, top=top, left=left, height=crop_h, width=crop_w)
        boxes = boxes.clone()
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes = self._clip_and_filter_boxes(boxes, width=crop_w, height=crop_h, min_size=2.0)
        return image_tensor, boxes

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.pil_to_tensor(image).float() / 255.0

        rows = self.grouped.get_group(image_id)
        boxes = torch.tensor(parse_bbox_list(rows["bbox"]), dtype=torch.float32)
        if self.augment and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            boxes = self._hflip_boxes(boxes, width=image_tensor.shape[2])
        if self.augment and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[1])
            boxes = self._vflip_boxes(boxes, height=image_tensor.shape[1])
        if self.augment:
            image_tensor, boxes = self._apply_scale_jitter(image_tensor, boxes)
            image_tensor, boxes = self._apply_random_crop(image_tensor, boxes)

        boxes = self._clip_and_filter_boxes(
            boxes, width=image_tensor.shape[2], height=image_tensor.shape[1], min_size=2.0
        )
        labels = torch.ones((boxes.size(0),), dtype=torch.int64)  # wheat class id = 1
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.size(0),), dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }
        return image_tensor, target


def _parse_anchor_sizes(anchor_sizes: str) -> Tuple[Tuple[int], ...]:
    vals = [int(v.strip()) for v in anchor_sizes.split(",") if v.strip()]
    if not vals:
        raise ValueError("anchor_sizes must contain at least one integer value.")
    return tuple((v,) for v in vals)


def _parse_anchor_aspects(anchor_aspects: str, num_levels: int) -> Tuple[Tuple[float, ...], ...]:
    vals = [float(v.strip()) for v in anchor_aspects.split(",") if v.strip()]
    if not vals:
        raise ValueError("anchor_aspects must contain at least one float value.")
    return tuple((tuple(vals) for _ in range(num_levels)))


def get_model(
    num_classes: int,
    backbone_name: str = "resnet50",
    anchor_sizes: Optional[str] = None,
    anchor_aspects: str = "0.5,1.0,2.0",
) -> nn.Module:
    if backbone_name == "resnet50":
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    elif backbone_name == "resnet101":
        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, progress=True)
        backbone = _resnet_fpn_extractor(backbone, 3, norm_layer=nn.BatchNorm2d)
        model = FasterRCNN(backbone, num_classes=num_classes + 1)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if anchor_sizes:
        sizes = _parse_anchor_sizes(anchor_sizes)
        aspects = _parse_anchor_aspects(anchor_aspects, num_levels=len(sizes))
        model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspects)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_sums: DefaultDict[str, float] = defaultdict(float)
    step_count = 0

    train_bar = make_tqdm(loader, desc="Train", leave=False)
    for images, targets in train_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_sums["loss_total"] += losses.item()
        for key, value in loss_dict.items():
            loss_sums[key] += float(value.item())
        step_count += 1

        running_loss = loss_sums["loss_total"] / step_count
        train_bar.set_postfix({"loss": f"{running_loss:.4f}"})

    step_count = max(1, step_count)
    return {k: v / step_count for k, v in loss_sums.items()}


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    # Detection models in torchvision return losses only in train mode with targets.
    model.train()
    loss_sums: DefaultDict[str, float] = defaultdict(float)
    step_count = 0

    for images, targets in make_tqdm(loader, desc="Val Loss", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_sums["loss_total"] += float(losses.item())
        for key, value in loss_dict.items():
            loss_sums[key] += float(value.item())
        step_count += 1

    step_count = max(1, step_count)
    return {k: v / step_count for k, v in loss_sums.items()}


@torch.no_grad()
def evaluate_inference_count(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
) -> Tuple[float, float]:
    model.eval()
    pred_counts: List[float] = []
    gt_counts: List[float] = []

    for images, targets in make_tqdm(loader, desc="Val", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            keep = out["scores"].detach().cpu() >= score_threshold
            pred_counts.append(float(keep.sum().item()))
            gt_counts.append(float(tgt["boxes"].shape[0]))

    return float(np.mean(pred_counts)), float(np.mean(gt_counts))


def _count_iou_matches(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_threshold: float) -> int:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0

    ious = box_iou(pred_boxes, gt_boxes)
    matches = 0
    while True:
        max_iou = ious.max()
        if float(max_iou.item()) < iou_threshold:
            break
        flat_idx = int(ious.argmax().item())
        pred_idx = flat_idx // ious.shape[1]
        gt_idx = flat_idx % ious.shape[1]
        matches += 1
        ious[pred_idx, :] = -1.0
        ious[:, gt_idx] = -1.0
    return matches


@torch.no_grad()
def evaluate_detection_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float = 0.5,
    desc: str = "Acc",
) -> float:
    model.eval()
    matched_total = 0
    gt_total = 0

    for images, targets in make_tqdm(loader, desc=desc, leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            pred_boxes = out["boxes"].detach().cpu()
            pred_scores = out["scores"].detach().cpu()
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            gt_boxes = tgt["boxes"].detach().cpu()

            matched_total += _count_iou_matches(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
            gt_total += int(gt_boxes.shape[0])

    if gt_total == 0:
        return float("nan")
    return float(matched_total / gt_total)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate_map50(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
    desc: str = "mAP50",
) -> Dict[str, float]:
    model.eval()
    all_scores: List[float] = []
    all_tp: List[int] = []
    all_fp: List[int] = []
    total_gt = 0

    for images, targets in make_tqdm(loader, desc=desc, leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            pred_boxes = out["boxes"].detach().cpu()
            pred_scores = out["scores"].detach().cpu()
            gt_boxes = tgt["boxes"].detach().cpu()
            total_gt += int(gt_boxes.shape[0])

            if pred_boxes.numel() == 0:
                continue

            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]

            if gt_boxes.numel() == 0:
                for score in pred_scores.tolist():
                    all_scores.append(float(score))
                    all_tp.append(0)
                    all_fp.append(1)
                continue

            matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
            ious = box_iou(pred_boxes, gt_boxes)
            for pred_idx in range(pred_boxes.shape[0]):
                score = float(pred_scores[pred_idx].item())
                best_iou, best_gt = torch.max(ious[pred_idx], dim=0)
                if float(best_iou.item()) >= iou_threshold and not matched_gt[best_gt]:
                    matched_gt[best_gt] = True
                    all_scores.append(score)
                    all_tp.append(1)
                    all_fp.append(0)
                else:
                    all_scores.append(score)
                    all_tp.append(0)
                    all_fp.append(1)

    if total_gt == 0:
        return {"ap50": float("nan"), "precision": float("nan"), "recall": float("nan")}
    if len(all_scores) == 0:
        return {"ap50": 0.0, "precision": 0.0, "recall": 0.0}

    scores_np = np.array(all_scores)
    tp_np = np.array(all_tp)
    fp_np = np.array(all_fp)
    order = np.argsort(-scores_np)
    tp_np = tp_np[order]
    fp_np = fp_np[order]

    cum_tp = np.cumsum(tp_np)
    cum_fp = np.cumsum(fp_np)
    recalls = cum_tp / max(1, total_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    ap50 = _compute_ap(recalls, precisions)

    return {
        "ap50": float(ap50),
        "precision": float(precisions[-1]) if precisions.size else 0.0,
        "recall": float(recalls[-1]) if recalls.size else 0.0,
    }


@torch.no_grad()
def save_prediction_preview(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    output_path: Path,
    score_threshold: float = 0.4,
    sample_index: int = 0,
) -> None:
    model.eval()
    image, _ = dataset[sample_index]
    pred = model([image.to(device)])[0]
    keep = pred["scores"].detach().cpu() >= score_threshold

    img_uint8 = (image * 255).to(torch.uint8)
    boxes = pred["boxes"].detach().cpu()[keep]
    scores = pred["scores"].detach().cpu()[keep]
    labels = [f"wheat:{s:.2f}" for s in scores.tolist()]

    rendered = draw_bounding_boxes(img_uint8, boxes=boxes, labels=labels, colors="red", width=2)
    rendered_np = rendered.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(rendered_np)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_training_plots(history_df: pd.DataFrame, output_dir: Path) -> None:
    epochs = history_df["epoch"].tolist()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history_df["loss_total"], marker="o", label="train_total")
    if "val_loss_total" in history_df.columns:
        plt.plot(epochs, history_df["val_loss_total"], marker="o", label="val_total")
    for key in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        if key in history_df.columns:
            plt.plot(epochs, history_df[key], marker="o", label=key.replace("loss_", ""))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png")
    plt.close()

    if (
        "train_map50" in history_df.columns
        and "val_map50" in history_df.columns
        and (history_df["train_map50"].notna().any() or history_df["val_map50"].notna().any())
    ):
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, history_df["train_map50"], marker="o", label="train_mAP50")
        plt.plot(epochs, history_df["val_map50"], marker="o", label="val_mAP50")
        plt.xlabel("Epoch")
        plt.ylabel("mAP@0.5")
        plt.title("Detection mAP50 Curves")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "map50_curves.png")
        plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history_df["epoch_time_sec"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title("Epoch Runtime")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "epoch_runtime.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on Global Wheat Detection")
    parser.add_argument("--data-dir", type=str, required=True, help="Path containing train.csv and train/")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument(
        "--anchor-sizes",
        type=str,
        default="",
        help="Comma-separated RPN anchor sizes, e.g. '16,32,64,128,256'. Empty uses torchvision default.",
    )
    parser.add_argument(
        "--anchor-aspects",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated aspect ratios used for all feature levels when custom anchors are enabled.",
    )
    parser.add_argument("--use-scale-jitter", action="store_true", help="Enable random scaling augmentation.")
    parser.add_argument("--use-random-crop", action="store_true", help="Enable random crop augmentation.")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate val mAP50 every N epochs during training. 0 disables periodic eval (final eval only).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early stop patience based on periodic val mAP50 checks. 0 disables early stopping.",
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default="baseline",
        help="Tag saved into training history for experiment comparison.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "train.csv"
    image_dir = data_dir / "train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert csv_path.exists(), f"Missing file: {csv_path}"
    assert image_dir.exists(), f"Missing directory: {image_dir}"

    df = pd.read_csv(csv_path)
    image_ids = df["image_id"].unique().tolist()
    split = build_split(image_ids, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = WheatDataset(
        df,
        image_dir=image_dir,
        image_ids=split.train_ids,
        augment=True,
        scale_jitter=args.use_scale_jitter,
        random_crop=args.use_random_crop,
    )
    val_ds = WheatDataset(df, image_dir=image_dir, image_ids=split.val_ids, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        num_classes=1,
        backbone_name=args.backbone,
        anchor_sizes=args.anchor_sizes if args.anchor_sizes else None,
        anchor_aspects=args.anchor_aspects,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

    best_train_loss = float("inf")
    best_val_map50 = -1.0
    no_improve_checks = 0
    history: List[Dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")
    print(f"Backbone: {args.backbone}")
    print(f"Custom anchors: {args.anchor_sizes if args.anchor_sizes else 'default'}")
    print(f"Augmentations: hflip+vflip, scale_jitter={args.use_scale_jitter}, random_crop={args.use_random_crop}")
    if args.eval_every > 0:
        print(f"Periodic eval: every {args.eval_every} epoch(s)")
        print(f"Early-stop patience (checks): {args.early_stop_patience}")
    else:
        print("Per-epoch mode: train only (evaluation runs once after training).")

    epoch_progress = make_tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_progress:
        epoch_start = time.time()
        train_losses = train_one_epoch(model, train_loader, optimizer, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "loss_total": float(train_losses.get("loss_total", np.nan)),
            "loss_classifier": float(train_losses.get("loss_classifier", np.nan)),
            "loss_box_reg": float(train_losses.get("loss_box_reg", np.nan)),
            "loss_objectness": float(train_losses.get("loss_objectness", np.nan)),
            "loss_rpn_box_reg": float(train_losses.get("loss_rpn_box_reg", np.nan)),
            "val_loss_total": np.nan,
            "val_loss_classifier": np.nan,
            "val_loss_box_reg": np.nan,
            "val_loss_objectness": np.nan,
            "val_loss_rpn_box_reg": np.nan,
            "val_avg_pred_boxes": np.nan,
            "val_avg_gt_boxes": np.nan,
            "train_map50": np.nan,
            "val_map50": np.nan,
            "lr": float(current_lr),
            "epoch_time_sec": float(epoch_time),
            "backbone": args.backbone,
            "anchor_sizes": args.anchor_sizes if args.anchor_sizes else "default",
            "anchor_aspects": args.anchor_aspects,
            "use_scale_jitter": bool(args.use_scale_jitter),
            "use_random_crop": bool(args.use_random_crop),
            "experiment_tag": args.experiment_tag,
        }
        # Optional periodic validation during training.
        periodic_eval = args.eval_every > 0 and (epoch % args.eval_every == 0)
        if periodic_eval:
            val_losses = evaluate_loss(model, val_loader, device=device)
            val_map = evaluate_map50(
                model, val_loader, device=device, iou_threshold=args.iou_threshold, desc=f"Val mAP50@Ep{epoch:02d}"
            )
            row["val_loss_total"] = float(val_losses.get("loss_total", np.nan))
            row["val_loss_classifier"] = float(val_losses.get("loss_classifier", np.nan))
            row["val_loss_box_reg"] = float(val_losses.get("loss_box_reg", np.nan))
            row["val_loss_objectness"] = float(val_losses.get("loss_objectness", np.nan))
            row["val_loss_rpn_box_reg"] = float(val_losses.get("loss_rpn_box_reg", np.nan))
            row["val_map50"] = float(val_map.get("ap50", np.nan))

            current_val_map50 = float(val_map.get("ap50", np.nan))
            if np.isfinite(current_val_map50) and current_val_map50 > best_val_map50:
                best_val_map50 = current_val_map50
                no_improve_checks = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                safe_tqdm_write(
                    f"  -> New best val_mAP50={best_val_map50:.4f} at epoch {epoch:02d}. Saved best_model.pt"
                )
            else:
                no_improve_checks += 1
                safe_tqdm_write(
                    f"  -> No val_mAP50 improvement (best={best_val_map50:.4f}, checks_without_improve={no_improve_checks})"
                )

        history.append(row)

        postfix = {
            "train_loss": f"{row['loss_total']:.4f}",
            "lr": f"{row['lr']:.2e}",
        }
        if np.isfinite(row.get("val_map50", np.nan)):
            postfix["val_mAP50"] = f"{row['val_map50']:.4f}"
        epoch_progress.set_postfix(postfix)

        msg = (
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={row['loss_total']:.4f} | "
            f"loss_cls={row['loss_classifier']:.4f} | "
            f"loss_box={row['loss_box_reg']:.4f} | "
            f"loss_obj={row['loss_objectness']:.4f} | "
            f"loss_rpn={row['loss_rpn_box_reg']:.4f} | "
        )
        if np.isfinite(row.get("val_map50", np.nan)):
            msg += f"val_mAP50={row['val_map50']:.4f} | "
        msg += f"lr={current_lr:.2e} | epoch_time={epoch_time:.1f}s"
        safe_tqdm_write(msg)

        # Fallback best-model criterion when periodic val mAP50 is not enabled.
        if args.eval_every <= 0 and row["loss_total"] < best_train_loss:
            best_train_loss = row["loss_total"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        # Early stopping checks are based on periodic validation only.
        if args.eval_every > 0 and args.early_stop_patience > 0 and no_improve_checks >= args.early_stop_patience:
            safe_tqdm_write(
                f"Early stopping triggered at epoch {epoch:02d} (no val_mAP50 improvement in {no_improve_checks} checks)."
            )
            break

    torch.save(model.state_dict(), output_dir / "last_model.pt")

    # Run evaluation once after training to avoid extra per-epoch computation.
    final_val_losses = evaluate_loss(model, val_loader, device=device)
    final_train_map = evaluate_map50(
        model,
        train_loader,
        device=device,
        iou_threshold=args.iou_threshold,
        desc="Final Train mAP50",
    )
    final_val_map = evaluate_map50(
        model,
        val_loader,
        device=device,
        iou_threshold=args.iou_threshold,
        desc="Final Val mAP50",
    )
    final_avg_pred, final_avg_gt = evaluate_inference_count(
        model,
        val_loader,
        device=device,
        score_threshold=args.score_threshold,
    )
    if history:
        history[-1].update(
            {
                "val_loss_total": float(final_val_losses.get("loss_total", np.nan)),
                "val_loss_classifier": float(final_val_losses.get("loss_classifier", np.nan)),
                "val_loss_box_reg": float(final_val_losses.get("loss_box_reg", np.nan)),
                "val_loss_objectness": float(final_val_losses.get("loss_objectness", np.nan)),
                "val_loss_rpn_box_reg": float(final_val_losses.get("loss_rpn_box_reg", np.nan)),
                "val_avg_pred_boxes": float(final_avg_pred),
                "val_avg_gt_boxes": float(final_avg_gt),
                "train_map50": float(final_train_map.get("ap50", np.nan)),
                "val_map50": float(final_val_map.get("ap50", np.nan)),
            }
        )
    print(
        f"Final Evaluation | val_loss={final_val_losses.get('loss_total', np.nan):.4f} | "
        f"train_mAP50={final_train_map.get('ap50', np.nan):.4f} | "
        f"val_mAP50={final_val_map.get('ap50', np.nan):.4f} | "
        f"val_pred={final_avg_pred:.2f} | val_gt={final_avg_gt:.2f}"
    )

    preview_path = output_dir / "val_prediction_preview.png"
    save_prediction_preview(
        model,
        dataset=val_ds,
        device=device,
        output_path=preview_path,
        score_threshold=args.score_threshold,
        sample_index=0,
    )
    print(f"Saved preview image: {preview_path}")

    history_df = pd.DataFrame(history)
    history_df["epoch"] = history_df["epoch"].astype(int)
    history_path = output_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    save_training_plots(history_df, output_dir)

    print(f"Saved training history: {history_path}")
    print(f"Saved loss curves: {output_dir / 'loss_curves.png'}")
    if (
        "train_map50" in history_df.columns
        and "val_map50" in history_df.columns
        and (history_df["train_map50"].notna().any() or history_df["val_map50"].notna().any())
    ):
        print(f"Saved mAP50 curves: {output_dir / 'map50_curves.png'}")
    print(f"Saved runtime trend: {output_dir / 'epoch_runtime.png'}")
    print(f"Saved best model: {output_dir / 'best_model.pt'}")
    print(f"Saved last model: {output_dir / 'last_model.pt'}")


if __name__ == "__main__":
    # CUDA fragmentation may happen on long runs; this setting can improve stability.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    main()

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
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

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
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes


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
    ) -> None:
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.augment = augment
        self.grouped = annotations_df.groupby("image_id")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _hflip_boxes(self, boxes: torch.Tensor, width: int) -> torch.Tensor:
        flipped = boxes.clone()
        flipped[:, [0, 2]] = width - boxes[:, [2, 0]]
        return flipped

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.pil_to_tensor(image).float() / 255.0

        rows = self.grouped.get_group(image_id)
        boxes = torch.tensor(parse_bbox_list(rows["bbox"]), dtype=torch.float32)
        labels = torch.ones((boxes.size(0),), dtype=torch.int64)  # wheat class id = 1

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.size(0),), dtype=torch.int64)

        if self.augment and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            boxes = self._hflip_boxes(boxes, width=image_tensor.shape[2])

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }
        return image_tensor, target


def get_model(num_classes: int) -> nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
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

    for images, targets in tqdm(loader, desc="Train", leave=False):
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

    for images, targets in tqdm(loader, desc="Val", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            keep = out["scores"].detach().cpu() >= score_threshold
            pred_counts.append(float(keep.sum().item()))
            gt_counts.append(float(tgt["boxes"].shape[0]))

    return float(np.mean(pred_counts)), float(np.mean(gt_counts))


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
    plt.plot(epochs, history_df["loss_total"], marker="o", label="total")
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

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history_df["val_avg_pred_boxes"], marker="o", label="avg predicted boxes")
    plt.plot(epochs, history_df["val_avg_gt_boxes"], marker="o", label="avg GT boxes")
    plt.xlabel("Epoch")
    plt.ylabel("Boxes per Image")
    plt.title("Validation Box Count Trend")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "val_box_count_trend.png")
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

    train_ds = WheatDataset(df, image_dir=image_dir, image_ids=split.train_ids, augment=True)
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
    model = get_model(num_classes=1).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

    best_loss = float("inf")
    history: List[Dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    epoch_progress = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_progress:
        epoch_start = time.time()
        train_losses = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        avg_pred, avg_gt = evaluate_inference_count(
            model,
            val_loader,
            device=device,
            score_threshold=args.score_threshold,
        )
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "loss_total": float(train_losses.get("loss_total", np.nan)),
            "loss_classifier": float(train_losses.get("loss_classifier", np.nan)),
            "loss_box_reg": float(train_losses.get("loss_box_reg", np.nan)),
            "loss_objectness": float(train_losses.get("loss_objectness", np.nan)),
            "loss_rpn_box_reg": float(train_losses.get("loss_rpn_box_reg", np.nan)),
            "val_avg_pred_boxes": avg_pred,
            "val_avg_gt_boxes": avg_gt,
            "lr": float(current_lr),
            "epoch_time_sec": float(epoch_time),
        }
        history.append(row)

        epoch_progress.set_postfix(
            {
                "loss": f"{row['loss_total']:.4f}",
                "pred/gt": f"{row['val_avg_pred_boxes']:.2f}/{row['val_avg_gt_boxes']:.2f}",
                "lr": f"{row['lr']:.2e}",
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss_total={row['loss_total']:.4f} | "
            f"loss_cls={row['loss_classifier']:.4f} | "
            f"loss_box={row['loss_box_reg']:.4f} | "
            f"loss_obj={row['loss_objectness']:.4f} | "
            f"loss_rpn={row['loss_rpn_box_reg']:.4f} | "
            f"val_pred={avg_pred:.2f} | val_gt={avg_gt:.2f} | "
            f"lr={current_lr:.2e} | epoch_time={epoch_time:.1f}s"
        )

        if row["loss_total"] < best_loss:
            best_loss = row["loss_total"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "last_model.pt")

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
    print(f"Saved validation trend: {output_dir / 'val_box_count_trend.png'}")
    print(f"Saved runtime trend: {output_dir / 'epoch_runtime.png'}")
    print(f"Saved best model: {output_dir / 'best_model.pt'}")
    print(f"Saved last model: {output_dir / 'last_model.pt'}")


if __name__ == "__main__":
    # CUDA fragmentation may happen on long runs; this setting can improve stability.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    main()

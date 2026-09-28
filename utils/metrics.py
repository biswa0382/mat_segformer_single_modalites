"""Streaming semantic-segmentation metrics."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction = prediction.detach().reshape(-1).to("cpu", dtype=torch.int64)
        target = target.detach().reshape(-1).to("cpu", dtype=torch.int64)
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        encoded = target[valid] * self.num_classes + prediction[valid]
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self, include_background: bool = False) -> dict[str, Any]:
        matrix = self.confusion.to(torch.float64)
        intersection = matrix.diag()
        ground_truth = matrix.sum(dim=1)
        prediction = matrix.sum(dim=0)
        union = ground_truth + prediction - intersection
        dice_denominator = ground_truth + prediction
        iou = torch.where(union > 0, intersection / union, torch.nan)
        dice = torch.where(
            dice_denominator > 0,
            2.0 * intersection / dice_denominator,
            torch.nan,
        )
        first_class = 0 if include_background else 1
        selected_iou = iou[first_class:]
        selected_dice = dice[first_class:]
        valid_iou = selected_iou[~selected_iou.isnan()]
        valid_dice = selected_dice[~selected_dice.isnan()]
        total = matrix.sum()
        return {
            "mIoU": valid_iou.mean().item() if valid_iou.numel() else float("nan"),
            "Dice": valid_dice.mean().item() if valid_dice.numel() else float("nan"),
            "pixel_accuracy": (intersection.sum() / total).item() if total > 0 else float("nan"),
            "class_iou": [None if value.isnan() else value.item() for value in iou],
            "class_dice": [None if value.isnan() else value.item() for value in dice],
        }


def named_class_iou(
    class_names: Sequence[str], class_iou: Sequence[float | None]
) -> dict[str, float | None]:
    if len(class_names) != len(class_iou):
        raise ValueError("Class names and IoU values have different lengths")
    return dict(zip(class_names, class_iou))


def format_class_iou(
    class_names: Sequence[str], class_iou: Sequence[float | None]
) -> str:
    rows = ["Class-wise IoU:"]
    width = max(len(name) for name in class_names)
    for class_id, (name, value) in enumerate(zip(class_names, class_iou)):
        score = "N/A" if value is None else f"{value:.4f}"
        rows.append(f"  [{class_id:02d}] {name:<{width}}  {score}")
    return "\n".join(rows)


__all__ = ["SegmentationMetrics", "format_class_iou", "named_class_iou"]

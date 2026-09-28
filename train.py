"""Train SegFormer on a selected aligned material-segmentation modality."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .datasets import ClassInfo, build_dataloaders
from .datasets.modalities import MODALITIES, MODALITY_CHOICES, canonical_modality, normalization_stats
from .model import MIT_CONFIGS, SegFormer, build_segformer
from .utils import SegmentationMetrics, format_class_iou, named_class_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a standalone MiT + SegFormer head on material masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--data-root", required=True)
    paths.add_argument("--split-file", required=True)
    paths.add_argument("--info-file", required=True)
    paths.add_argument("--output-dir", required=True)
    paths.add_argument("--missing-mask-report", required=True)
    paths.add_argument("--pretrained", default="", help="MiT ImageNet checkpoint")
    paths.add_argument("--resume", default="", help="Full training checkpoint")

    model = parser.add_argument_group("model")
    model.add_argument("--variant", choices=tuple(MIT_CONFIGS), default="b0")
    model.add_argument("--decoder-dim", type=int, default=0, help="0 uses variant default")
    model.add_argument("--drop-rate", type=float, default=0.0)
    model.add_argument("--attn-drop-rate", type=float, default=0.0)
    model.add_argument("--drop-path-rate", type=float, default=0.1)
    model.add_argument("--head-dropout", type=float, default=0.1)

    data = parser.add_argument_group("data and augmentation")
    data.add_argument("--modality", choices=MODALITY_CHOICES, default="rgb")
    data.add_argument("--image-height", type=int, default=512)
    data.add_argument("--image-width", type=int, default=640)
    data.add_argument("--mask-dirname", default="grayscale")
    data.add_argument(
        "--skip-missing-masks", action=argparse.BooleanOptionalAction, default=True
    )
    data.add_argument("--batch-size", type=int, default=4)
    data.add_argument("--num-workers", type=int, default=4)
    data.add_argument("--mean", type=float, nargs="+", default=None, help="Per-channel means; default depends on modality")
    data.add_argument("--std", type=float, nargs="+", default=None, help="Per-channel standard deviations; default depends on modality")
    data.add_argument("--hflip-prob", type=float, default=0.5)
    data.add_argument("--vflip-prob", type=float, default=0.0)
    data.add_argument("--brightness", type=float, default=0.2)
    data.add_argument("--contrast", type=float, default=0.2)
    data.add_argument("--saturation", type=float, default=0.2)
    data.add_argument("--hue", type=float, default=0.05)
    data.add_argument("--ignore-index", type=int, default=255)
    data.add_argument("--strict-labels", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )

    training = parser.add_argument_group("optimization")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="cuda", help="cuda, cuda:N, cpu, or auto")
    training.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--accumulation-steps", type=int, default=1)
    training.add_argument("--backbone-lr", type=float, default=6e-5)
    training.add_argument("--head-lr", type=float, default=6e-4)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--beta1", type=float, default=0.9)
    training.add_argument("--beta2", type=float, default=0.999)
    training.add_argument("--label-smoothing", type=float, default=0.0)
    training.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    training.add_argument("--scheduler", choices=("poly", "cosine", "constant"), default="poly")
    training.add_argument("--warmup-epochs", type=float, default=2.0)
    training.add_argument("--warmup-start-ratio", type=float, default=0.01)
    training.add_argument("--min-lr-ratio", type=float, default=0.01)
    training.add_argument("--poly-power", type=float, default=1.0)

    reporting = parser.add_argument_group("reporting and checkpoints")
    reporting.add_argument("--include-background", action=argparse.BooleanOptionalAction, default=False)
    reporting.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "accumulation_steps": args.accumulation_steps,
        "log_interval": args.log_interval,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    for name in ("hflip_prob", "vflip_prob"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in ("drop_rate", "attn_drop_rate", "drop_path_rate", "head_dropout"):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if not 0.0 <= args.warmup_start_ratio <= 1.0:
        raise ValueError("--warmup-start-ratio must be in [0, 1]")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_scheduler(
    optimizer: Optimizer,
    schedule: str,
    total_steps: int,
    warmup_steps: int,
    warmup_start_ratio: float,
    min_lr_ratio: float,
    poly_power: float,
) -> LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            progress = step / max(1, warmup_steps)
            return warmup_start_ratio + (1.0 - warmup_start_ratio) * progress
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if schedule == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            decay = (1.0 - progress) ** poly_power
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return LambdaLR(optimizer, multiplier)


def make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(
    model: SegFormer,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    include_background: bool,
    amp: bool,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: Any | None = None,
    accumulation_steps: int = 1,
    grad_clip: float = 0.0,
    log_interval: int = 10,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(num_classes, ignore_index)
    running_loss = 0.0
    seen_samples = 0
    total_batches = len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss / group_size).backward()
                should_step = batch_index % accumulation_steps == 0 or batch_index == total_batches
                if should_step:
                    if grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        batch_size = images.shape[0]
        running_loss += loss.detach().item() * batch_size
        seen_samples += batch_size
        metrics.update(logits.argmax(dim=1), targets)
        if training and (batch_index % log_interval == 0 or batch_index == total_batches):
            learning_rates = ", ".join(f"{group['lr']:.3e}" for group in optimizer.param_groups)
            print(
                f"    batch {batch_index:03d}/{total_batches:03d} "
                f"loss={running_loss / seen_samples:.4f} lr=[{learning_rates}]",
                flush=True,
            )

    result = metrics.compute(include_background)
    result["loss"] = running_loss / max(1, seen_samples)
    return result


def torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def checkpoint_state(
    epoch: int,
    model: SegFormer,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    args: argparse.Namespace,
    train_result: Mapping[str, Any],
    val_result: Mapping[str, Any],
    best_miou: float,
    class_info: ClassInfo,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
        "train": dict(train_result),
        "validation": dict(val_result),
        "best_mIoU": best_miou,
        "class_names": list(class_info.names),
        "class_colors": [list(color) for color in class_info.colors],
    }


def load_resume_checkpoint(
    path: str | Path,
    model: SegFormer,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    updates_per_epoch: int,
    expected_modality: str | None = None,
) -> tuple[int, float]:
    checkpoint = torch_load(path)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError(f"{path} is not a full training checkpoint")
    if expected_modality is not None:
        saved_modality = canonical_modality(checkpoint.get("args", {}).get("modality", "rgb"))
        if saved_modality != expected_modality:
            raise ValueError(f"Cannot resume {saved_modality} training as {expected_modality}")
    model.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint and checkpoint["scaler"]:
        scaler.load_state_dict(checkpoint["scaler"])
    completed_epoch = int(checkpoint.get("epoch", 0))
    expected_step = completed_epoch * updates_per_epoch
    loaded_step = int(scheduler.last_epoch)
    if loaded_step != expected_step:
        scheduler.last_epoch = expected_step
        scheduler._step_count = expected_step + 1
        adjusted_lrs = [
            base_lr * lr_lambda(expected_step)
            for base_lr, lr_lambda in zip(scheduler.base_lrs, scheduler.lr_lambdas)
        ]
        for parameter_group, learning_rate in zip(optimizer.param_groups, adjusted_lrs):
            parameter_group["lr"] = learning_rate
        scheduler._last_lr = adjusted_lrs
        print(
            f"Realigned scheduler from update {loaded_step} to {expected_step} "
            f"because the usable dataset size changed."
        )
    validation = checkpoint.get("validation", {})
    previous_score = checkpoint.get(
        "best_mIoU", validation.get("mIoU", checkpoint.get("val_mIoU", -math.inf))
    )
    return completed_epoch + 1, float(previous_score)


def main() -> None:
    args = parse_args()
    args.modality = canonical_modality(args.modality)
    args.mean, args.std = normalization_stats(args.modality, args.mean, args.std)
    args.in_channels = MODALITIES[args.modality].channels
    validate_args(args)
    device = resolve_device(args.device)
    seed_everything(args.seed, args.deterministic)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_info = build_dataloaders(
        data_root=args.data_root,
        split_file=args.split_file,
        info_file=args.info_file,
        mask_dirname=args.mask_dirname,
        missing_mask_report=args.missing_mask_report,
        skip_missing_masks=args.skip_missing_masks,
        height=args.image_height,
        width=args.image_width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        mean=args.mean,
        std=args.std,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
        ignore_index=args.ignore_index,
        strict_labels=args.strict_labels,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        modality=args.modality,
    )
    model = build_segformer(
        variant=args.variant,
        num_classes=class_info.num_classes,
        in_channels=args.in_channels,
        decoder_dim=args.decoder_dim or None,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        head_dropout=args.head_dropout,
    )


    if args.pretrained and not args.resume:
        report = model.load_pretrained_backbone(args.pretrained)
        print(
            f"Loaded {report['loaded']} MiT tensors from {args.pretrained}; "
            f"missing={len(report['missing'])}, skipped={len(report['skipped'])}, "
            f"adapted={report['adapted']}"
        )
    model.to(device)

    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    scheduler = make_scheduler(
        optimizer,
        args.scheduler,
        args.epochs * updates_per_epoch,
        round(args.warmup_epochs * updates_per_epoch),
        args.warmup_start_ratio,
        args.min_lr_ratio,
        args.poly_power,
    )
    amp = args.amp and device.type == "cuda"
    scaler = make_grad_scaler(amp)
    criterion = nn.CrossEntropyLoss(
        ignore_index=args.ignore_index,
        label_smoothing=args.label_smoothing,
    )

    start_epoch = 1
    best_miou = -math.inf
    if args.resume:
        start_epoch, best_miou = load_resume_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, updates_per_epoch,
            expected_modality=args.modality,
        )
        print(f"Resumed {args.resume} at epoch {start_epoch} (best mIoU={best_miou:.4f})")
    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume checkpoint starts at epoch {start_epoch}, beyond --epochs {args.epochs}"
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    skipped_train = len(train_loader.dataset.missing_samples)
    skipped_val = len(val_loader.dataset.missing_samples)
    if skipped_train or skipped_val:
        print(
            f"Skipping incomplete samples: train={skipped_train}, val={skipped_val}. "
            f"Details: {args.missing_mask_report}",
            flush=True,
        )
    print(
        f"modality={args.modality} channels={args.in_channels} Device={device} variant=MiT-{args.variant.upper()} classes={class_info.num_classes} "
        f"parameters={parameter_count:,} train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"AMP={amp}",
        flush=True,
    )
    history: list[dict[str, Any]] = []
    history_path = output_dir / "history.json"
    if args.resume and history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.monotonic()
        print(f"\nEpoch {epoch:03d}/{args.epochs:03d}", flush=True)
        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            class_info.num_classes,
            args.ignore_index,
            args.include_background,
            amp,
            optimizer,
            scheduler,
            scaler,
            args.accumulation_steps,
            args.grad_clip,
            args.log_interval,
        )
        val_result = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            class_info.num_classes,
            args.ignore_index,
            args.include_background,
            amp,
        )
        elapsed = time.monotonic() - started
        print(f"Modality {args.modality}")
        print(
            f"  train: loss={train_result['loss']:.4f} "
            f"mIoU={train_result['mIoU']:.4f} Dice={train_result['Dice']:.4f}"
        )
        print(
            f"  val:   loss={val_result['loss']:.4f} "
            f"mIoU={val_result['mIoU']:.4f} Dice={val_result['Dice']:.4f} "
            f"pixel_acc={val_result['pixel_accuracy']:.4f} time={elapsed:.1f}s"
        )
        print(format_class_iou(class_info.names, val_result["class_iou"]), flush=True)

        is_best = val_result["mIoU"] > best_miou
        if is_best:
            best_miou = val_result["mIoU"]
        state = checkpoint_state(
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            train_result,
            val_result,
            best_miou,
            class_info,
        )
        state["dataset"] = {
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "skipped_train": skipped_train,
            "skipped_val": skipped_val,
            "missing_mask_report": args.missing_mask_report,
        }
        atomic_torch_save(state, output_dir / "last.pth")
        if is_best:
            atomic_torch_save(state, output_dir / "best.pth")
            atomic_torch_save(model.state_dict(), output_dir / "best_weights.pth")
            best_result = {
                "epoch": epoch,
                "mIoU": val_result["mIoU"],
                "Dice": val_result["Dice"],
                "pixel_accuracy": val_result["pixel_accuracy"],
                "class_iou": named_class_iou(class_info.names, val_result["class_iou"]),
            }
            atomic_json_save(best_result, output_dir / "best_metrics.json")
            print(f"  New best mIoU={best_miou:.4f}; saved best model and metrics.")

        history.append(
            {
                "epoch": epoch,
                "seconds": elapsed,
                "train": train_result,
                "validation": val_result,
            }
        )
        atomic_json_save(history, history_path)

    print(f"\nTraining complete. Best validation mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()

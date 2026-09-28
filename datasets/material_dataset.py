"""Dataset and paired transforms for the material segmentation data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .modalities import (
    MODALITIES, canonical_modality, load_modality_image, modality_paths,
    normalization_stats, preprocess_image,
)


@dataclass(frozen=True)
class ClassInfo:
    names: tuple[str, ...]
    colors: tuple[tuple[int, int, int], ...]

    @property
    def num_classes(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class MissingSample:
    split: str
    sample_id: str
    reason: str
    image_path: Path
    mask_path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "split": self.split,
            "sample_id": self.sample_id,
            "reason": self.reason,
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path),
        }


def load_class_info(info_file: str | Path) -> ClassInfo:
    """Read foreground class IDs/colors and prepend black background class 0."""
    info_file = Path(info_file)
    with info_file.open("r", encoding="utf-8") as handle:
        raw: dict[str, dict[str, Any]] = json.load(handle)
    ids = sorted(int(key) for key in raw)
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError(f"Class IDs in {info_file} must be contiguous and begin at 1; got {ids}")

    names = ["background"]
    colors = [(0, 0, 0)]
    for class_id in ids:
        entry = raw[str(class_id)]
        color = tuple(int(value) for value in entry["color"])
        if len(color) != 3 or any(value < 0 or value > 255 for value in color):
            raise ValueError(f"Invalid RGB color for class {class_id}: {color}")
        names.append(str(entry["class_name"]))
        colors.append(color)
    if len(set(colors)) != len(colors):
        raise ValueError(f"Duplicate colors found in {info_file}")
    return ClassInfo(tuple(names), tuple(colors))


def discover_samples(
    data_root: str | Path,
    split_file: str | Path,
    split: str,
    mask_dirname: str = "grayscale",
    modality: str = "rgb",
) -> tuple[list[Sample], list[MissingSample]]:
    """Resolve light-on modality NumPy images and prepared single-channel masks."""
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    modality = canonical_modality(modality)
    data_root = Path(data_root)
    with Path(split_file).open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    key = f"{split}_sets"
    if key not in split_data:
        raise KeyError(f"Missing {key!r} in {split_file}")

    samples: list[Sample] = []
    missing: list[MissingSample] = []
    for set_name in split_data[key]:
        set_dir = data_root / set_name
        if not set_dir.is_dir():
            raise FileNotFoundError(f"Split references a missing dataset set: {set_dir}")
        object_dirs = sorted(path for path in set_dir.iterdir() if path.is_dir())
        for object_dir in object_dirs:
            source_paths = modality_paths(object_dir, modality)
            image_path = source_paths[0]
            mask_path = object_dir / "segmentation_ground_truth" / mask_dirname / (
                f"{object_dir.name}_rgb_mask.png"
            )
            if all(path.is_file() for path in source_paths) and mask_path.is_file():
                samples.append(
                    Sample(
                        sample_id=f"{set_name}/{object_dir.name}",
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )
            else:
                absent = []
                for path in source_paths:
                    if not path.is_file():
                        absent.append(str(path.relative_to(object_dir)))
                if not mask_path.is_file():
                    absent.append("prepared grayscale mask")
                missing.append(
                    MissingSample(
                        split=split,
                        sample_id=f"{set_name}/{object_dir.name}",
                        reason="missing " + " and ".join(absent),
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )
    return samples, missing


class PairedTransform:
    """Transform a CHW modality tensor and its mask with synchronized geometry."""

    def __init__(
        self,
        height: int,
        width: int,
        training: bool,
        mean: Sequence[float],
        std: Sequence[float],
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        modality: str = "rgb",
    ) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("Image height and width must be positive")
        self.modality = canonical_modality(modality)
        mean, std = normalization_stats(self.modality, mean, std)
        if not 0.0 <= hue <= 0.5:
            raise ValueError("hue must be in [0, 0.5]")
        self.size = (height, width)
        self.training = training
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        if self.modality in {"aolp", "dolp", "polar", "thermal_static"}:
            brightness = contrast = saturation = hue = 0.0
        elif self.modality in {"nir", "pol"}:
            saturation = hue = 0.0
        self.jitter = (brightness, contrast, saturation, hue)

    @staticmethod
    def _factor(amount: float, center: float = 1.0) -> float:
        return random.uniform(max(0.0, center - amount), center + amount)

    def _color_jitter(self, image: Tensor) -> Tensor:
        brightness, contrast, saturation, hue = self.jitter
        operations = []
        if brightness > 0.0:
            operations.append(lambda img: TF.adjust_brightness(img, self._factor(brightness)))
        if contrast > 0.0:
            operations.append(lambda img: TF.adjust_contrast(img, self._factor(contrast)))
        if saturation > 0.0:
            operations.append(lambda img: TF.adjust_saturation(img, self._factor(saturation)))
        if hue > 0.0:
            operations.append(lambda img: TF.adjust_hue(img, random.uniform(-hue, hue)))
        random.shuffle(operations)
        for operation in operations:
            image = operation(image)
        return image

    def __call__(self, image: Tensor, mask: Image.Image) -> tuple[Tensor, Image.Image]:
        reflected = False
        if self.training and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            reflected = not reflected
        if self.training and random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            reflected = not reflected
        if reflected and self.modality in {"aolp", "polar"}:
            # Reflecting the polarization angle negates sin(2*theta), preserving cos.
            image = image.clone()
            image[1].neg_()
        if self.training:
            image = self._color_jitter(image)

        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        image_tensor = preprocess_image(image, self.size, self.mean, self.std)
        return image_tensor, mask


class MaterialSegmentationDataset(Dataset[dict[str, Any]]):
    """Aligned modality images paired with prepared class-ID masks."""

    def __init__(
        self,
        data_root: str | Path,
        split_file: str | Path,
        info_file: str | Path,
        split: str,
        transform: PairedTransform,
        mask_dirname: str = "grayscale",
        ignore_index: int = 255,
        strict_labels: bool = True,
        modality: str = "rgb",
    ) -> None:
        super().__init__()
        self.samples, self.missing_samples = discover_samples(
            data_root, split_file, split, mask_dirname, modality
        )
        self.modality = canonical_modality(modality)
        self.in_channels = MODALITIES[self.modality].channels
        self.class_info = load_class_info(info_file)
        self.transform = transform
        self.ignore_index = int(ignore_index)
        self.strict_labels = strict_labels

    def __len__(self) -> int:
        return len(self.samples)

    def _decode_mask(self, mask: Image.Image, sample_id: str) -> Tensor:
        labels = np.asarray(mask, dtype=np.int64)
        valid = (labels >= 0) & (labels < self.class_info.num_classes)
        if self.strict_labels and not bool(valid.all()):
            unknown = np.unique(labels[~valid])
            raise ValueError(
                f"Prepared mask for {sample_id} contains invalid class IDs: "
                f"{unknown[:20].tolist()}"
            )
        if not bool(valid.all()):
            labels = labels.copy()
            labels[~valid] = self.ignore_index
        return torch.from_numpy(labels.copy())

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = load_modality_image(sample.image_path, self.modality)

        with Image.open(sample.mask_path) as mask_file:
            if mask_file.mode not in {"L", "I", "P"}:
                raise ValueError(
                    f"Prepared mask must be single-channel for {sample.sample_id}; "
                    f"found mode {mask_file.mode!r}"
                )
            mask = mask_file.convert("L")
        image_size = (image.shape[2], image.shape[1])
        if image_size != mask.size:
            raise ValueError(
                f"Image/mask dimensions differ for {sample.sample_id}: {image_size} vs {mask.size}"
            )
        image_tensor, mask = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": self._decode_mask(mask, sample.sample_id),
            "sample_id": sample.sample_id,
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(
    data_root: str | Path,
    split_file: str | Path,
    info_file: str | Path,
    mask_dirname: str,
    missing_mask_report: str | Path,
    skip_missing_masks: bool,
    height: int,
    width: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
    hflip_prob: float,
    vflip_prob: float,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    ignore_index: int,
    strict_labels: bool,
    pin_memory: bool,
    persistent_workers: bool,
    modality: str = "rgb",
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], ClassInfo]:
    modality = canonical_modality(modality)
    mean, std = normalization_stats(modality, mean, std)
    train_transform = PairedTransform(
        height,
        width,
        True,
        mean,
        std,
        hflip_prob,
        vflip_prob,
        brightness,
        contrast,
        saturation,
        hue,
        modality=modality,
    )
    val_transform = PairedTransform(height, width, False, mean, std, modality=modality)
    common = dict(
        data_root=data_root,
        split_file=split_file,
        info_file=info_file,
        mask_dirname=mask_dirname,
        ignore_index=ignore_index,
        strict_labels=strict_labels,
        modality=modality,
    )
    train_dataset = MaterialSegmentationDataset(
        split="train", transform=train_transform, **common
    )
    val_dataset = MaterialSegmentationDataset(split="val", transform=val_transform, **common)
    if train_dataset.class_info != val_dataset.class_info:
        raise RuntimeError("Training and validation class metadata differ")

    report_path = Path(missing_mask_report)
    existing: dict[str, Any] = {}
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            existing = loaded
    train_missing = [record.as_dict() for record in train_dataset.missing_samples]
    val_missing = [record.as_dict() for record in val_dataset.missing_samples]
    existing["dataloader"] = {
        "modality": modality,
        "policy": "skip" if skip_missing_masks else "fail",
        "missing_count": len(train_missing) + len(val_missing),
        "train_missing_count": len(train_missing),
        "val_missing_count": len(val_missing),
        "missing_samples": train_missing + val_missing,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
        handle.write("\n")
    temporary.replace(report_path)

    total_missing = len(train_missing) + len(val_missing)
    if total_missing and not skip_missing_masks:
        raise FileNotFoundError(
            f"Found {total_missing} incomplete samples; details were saved to {report_path}"
        )
    if not train_dataset.samples:
        raise RuntimeError(f"No usable training samples remain; see {report_path}")
    if not val_dataset.samples:
        raise RuntimeError(f"No usable validation samples remain; see {report_path}")

    generator = torch.Generator().manual_seed(seed)
    loader_options = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=_seed_worker,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    return train_loader, val_loader, train_dataset.class_info


__all__ = [
    "ClassInfo",
    "MissingSample",
    "MaterialSegmentationDataset",
    "PairedTransform",
    "build_dataloaders",
    "discover_samples",
    "load_class_info",
]

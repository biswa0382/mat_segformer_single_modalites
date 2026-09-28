"""Modality paths, channel layouts, and NumPy loading shared by train and infer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision.transforms import functional as TF

from .aolp_cos_sin import aolp_cos_sin


@dataclass(frozen=True)
class Modality:
    files: tuple[str, ...]
    channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


_NPY = "light_on/aligned/npy"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
MODALITIES = {
    "rgb": Modality((f"{_NPY}/rgb.npy",), 3, _IMAGENET_MEAN, _IMAGENET_STD),
    "nir": Modality((f"{_NPY}/nir.npy",), 3, _IMAGENET_MEAN, _IMAGENET_STD),
    "aolp": Modality((f"{_NPY}/polar_aolp.npy",), 2, (0.0, 0.0), (1.0, 1.0)),
    "dolp": Modality((f"{_NPY}/dolp.npy",), 1, (0.0,), (1.0,)),
    "pol": Modality((f"{_NPY}/polar.npy",), 1, (0.0,), (1.0,)),
    "polar": Modality(
        (f"{_NPY}/polar_aolp.npy", f"{_NPY}/dolp.npy"),
        3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
    ),
    "thermal_static": Modality(
        ("thermal/thermal_first_frame_aligned.npy",), 3, _IMAGENET_MEAN, _IMAGENET_STD,
    ),
}
# Accept the previous AoLP selection spelling in existing commands/checkpoints.
MODALITY_CHOICES = (*MODALITIES, "polar_aolp", "poler")


def canonical_modality(name: str) -> str:
    name = name.lower()
    if name == "polar_aolp":
        name = "aolp"
    elif name == "poler":
        name = "polar"
    if name not in MODALITIES:
        raise ValueError(f"Unknown modality {name!r}; choose from {tuple(MODALITIES)}")
    return name


def normalization_stats(
    modality: str,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    spec = MODALITIES[canonical_modality(modality)]
    mean = tuple(spec.mean if mean is None else mean)
    std = tuple(spec.std if std is None else std)
    if len(mean) != spec.channels or len(std) != spec.channels:
        raise ValueError(f"{modality} needs {spec.channels} values for both mean and std")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or any(v <= 0 for v in std):
        raise ValueError("Normalization values must be finite and std must be positive")
    return mean, std


def modality_paths(sample_dir: Path, modality: str) -> tuple[Path, ...]:
    paths = tuple(sample_dir / name for name in MODALITIES[canonical_modality(modality)].files)
    return tuple(dolp_path(path.parent) if path.name == "dolp.npy" else path for path in paths)


def dolp_path(directory: Path) -> Path:
    """Prefer dolp.npy; also accept the filename used in existing collected samples."""
    preferred = directory / "dolp.npy"
    existing = directory / "polar_dolp.npy"
    return existing if not preferred.is_file() and existing.is_file() else preferred


def _load_array(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    numeric = np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    if not numeric or not array.size or not np.isfinite(array).all():
        raise ValueError(f"Expected a nonempty, finite numeric image in {path}")
    return array


def _single_channel(array: np.ndarray, path: Path) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected HxW or HxWx1 in {path}; got {array.shape}")
    return array


def _unit_range(array: np.ndarray, path: Path) -> np.ndarray:
    """Scale stored intensities to [0, 1] without quantizing through an 8-bit image."""
    values = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.unsignedinteger):
        return values / np.iinfo(array.dtype).max
    if values.min() >= 0 and values.max() <= 1:
        return values
    if values.min() >= 0 and values.max() <= 255:
        return values / 255.0
    raise ValueError(f"Expected normalized [0, 1] or [0, 255] image values in {path}")


def load_modality_image(image_path: str | Path, modality: str) -> Tensor:
    """Load a CHW float32 tensor; AoLP channels keep their signed [-1, 1] values."""
    modality = canonical_modality(modality)
    path = Path(image_path)
    # Retain support for individual RGB PNG/JPEG files during inference.
    if path.suffix.lower() != ".npy":
        if modality != "rgb":
            raise ValueError(f"{modality} input must be a NumPy file: {path}")
        with Image.open(path) as image:
            return TF.pil_to_tensor(image.convert("RGB")).float() / 255.0

    array = _load_array(path)
    if modality in {"aolp", "polar"}:
        angle = _single_channel(array, path)
        cosine, sine = aolp_cos_sin(angle)
        channels = [cosine, sine]
        if modality == "polar":
            source = dolp_path(path.parent)
            dolp = _unit_range(_single_channel(_load_array(source), source), source)
            if dolp.shape != angle.shape:
                raise ValueError(f"AoLP/DoLP dimensions differ: {angle.shape} vs {dolp.shape}")
            channels.append(dolp)
        return torch.from_numpy(np.stack(channels))

    if modality in {"dolp", "pol"}:
        values = _unit_range(_single_channel(array, path), path)
        return torch.from_numpy(values[None])

    # RGB and thermal are stored as BGR images; NIR is repeated for RGB pretraining.
    if modality == "nir" and (array.ndim == 2 or (array.ndim == 3 and array.shape[-1] == 1)):
        array = np.repeat(_single_channel(array, path)[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 BGR data in {path}; got {array.shape}")
    values = _unit_range(array, path)[..., ::-1]
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))


def preprocess_image(
    image: Tensor, size: tuple[int, int], mean: Sequence[float], std: Sequence[float]
) -> Tensor:
    image = TF.resize(image, size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    return TF.normalize(image, mean, std)


def preview_image(image: Tensor, modality: str) -> Image.Image:
    """Make an overlay background without changing the tensor sent to the model."""
    modality = canonical_modality(modality)
    if modality == "aolp":
        image = (torch.atan2(image[1:2], image[0:1]) % (2 * torch.pi)) / (2 * torch.pi)
    elif modality == "polar":
        image = image[2:3]  # Show DoLP behind the predicted mask.
    return TF.to_pil_image(image.clamp(0, 1)).convert("RGB")

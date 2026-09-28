"""SegFormer model assembly and pretrained-backbone loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbone import MIT_CONFIGS, build_mit_backbone
from .head import SegFormerHead


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a state dictionary")
    for key in ("model", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return checkpoint


class SegFormer(nn.Module):
    def __init__(
        self,
        variant: str,
        num_classes: int,
        decoder_dim: int | None = None,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        variant = variant.lower().removeprefix("mit_")
        if variant not in MIT_CONFIGS:
            raise ValueError(f"Unknown MiT variant {variant!r}; choose from {tuple(MIT_CONFIGS)}")
        self.variant = variant
        config = MIT_CONFIGS[variant]
        self.backbone = build_mit_backbone(
            variant,
            in_channels,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
        )
        self.head = SegFormerHead(
            config.embed_dims,
            decoder_dim if decoder_dim is not None else config.decoder_dim,
            num_classes,
            head_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        logits = self.head(self.backbone(x))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def load_pretrained_backbone(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Load official MiT weights or the backbone portion of a full checkpoint."""
        checkpoint = _extract_state_dict(_torch_load(checkpoint_path))
        target = self.backbone.state_dict()
        compatible: dict[str, Tensor] = {}
        skipped: list[str] = []
        adapted: list[str] = []
        for raw_key, value in checkpoint.items():
            key = raw_key.removeprefix("module.")
            key = key.removeprefix("model.")
            key = key.removeprefix("backbone.")
            if (
                key == "patch_embed1.proj.weight"
                and isinstance(value, Tensor)
                and value.ndim == 4
                and value.shape[1] == 3
                and target[key].shape[1] != 3
                and value.shape[0] == target[key].shape[0]
                and value.shape[2:] == target[key].shape[2:]
            ):
                channels = target[key].shape[1]
                # Preserve the summed RGB response while distributing it over C inputs.
                value = value.sum(dim=1, keepdim=True).repeat(1, channels, 1, 1) / channels
                adapted.append(key)
            if key in target and isinstance(value, Tensor) and target[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped.append(raw_key)
        if not compatible:
            raise RuntimeError(f"No compatible MiT weights found in {checkpoint_path}")
        incompatible = self.backbone.load_state_dict(compatible, strict=False)
        return {
            "loaded": len(compatible),
            "missing": list(incompatible.missing_keys),
            "skipped": skipped,
            "adapted": adapted,
        }


def build_segformer(variant: str, num_classes: int, **kwargs: Any) -> SegFormer:
    return SegFormer(variant=variant, num_classes=num_classes, **kwargs)


__all__ = ["SegFormer", "build_segformer"]

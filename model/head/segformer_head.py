"""All-MLP decoder head from SegFormer."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SegFormerHead(nn.Module):
    """Project four MiT feature maps, fuse them, and predict class logits."""

    def __init__(
        self,
        in_channels: Sequence[int],
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("SegFormerHead requires four backbone feature maps")
        self.projections = nn.ModuleList(
            nn.Linear(channels, embed_dim) for channels in in_channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.predict = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != 4:
            raise ValueError(f"Expected four feature maps, received {len(features)}")
        output_size = features[0].shape[-2:]
        projected: list[Tensor] = []
        for feature, projection in zip(features, self.projections):
            batch, _, height, width = feature.shape
            x = projection(feature.flatten(2).transpose(1, 2))
            x = x.transpose(1, 2).reshape(batch, -1, height, width)
            if (height, width) != output_size:
                x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
            projected.append(x)
        # Preserve the C4, C3, C2, C1 concatenation order of the original head.
        return self.predict(self.dropout(self.fuse(torch.cat(projected[::-1], dim=1))))


__all__ = ["SegFormerHead"]

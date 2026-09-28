"""Mix Transformer (MiT) backbones used by SegFormer.

Attribute names match NVIDIA's original implementation, allowing direct use
of the official ImageNet ``mit_b0.pth`` through ``mit_b5.pth`` checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep_prob


class DWConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        return self.dwconv(x).flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.drop(self.act(x))
        return self.drop(self.fc2(x))


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {dim} is not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            reduced = x.permute(0, 2, 1).reshape(batch, channels, height, width)
            reduced = self.sr(reduced).reshape(batch, channels, -1).permute(0, 2, 1)
            reduced = self.norm(reduced)
        else:
            reduced = x
        kv = self.kv(reduced).reshape(
            batch, -1, 2, self.num_heads, channels // self.num_heads
        )
        key, value = kv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = self.attn_drop(((q @ key.transpose(-2, -1)) * self.scale).softmax(-1))
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientSelfAttention(
            dim, num_heads, qkv_bias, attn_drop, drop, sr_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.mlp(self.norm2(x), height, width))


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self, in_channels: int, embed_dim: int, patch_size: int, stride: int
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


@dataclass(frozen=True)
class MiTConfig:
    embed_dims: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    decoder_dim: int


MIT_CONFIGS: dict[str, MiTConfig] = {
    "b0": MiTConfig((32, 64, 160, 256), (2, 2, 2, 2), 256),
    "b1": MiTConfig((64, 128, 320, 512), (2, 2, 2, 2), 256),
    "b2": MiTConfig((64, 128, 320, 512), (3, 4, 6, 3), 768),
    "b3": MiTConfig((64, 128, 320, 512), (3, 4, 18, 3), 768),
    "b4": MiTConfig((64, 128, 320, 512), (3, 8, 27, 3), 768),
    "b5": MiTConfig((64, 128, 320, 512), (3, 6, 40, 3), 768),
}


def init_mit_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MixVisionTransformer(nn.Module):
    """Four-stage hierarchical MiT encoder."""

    def __init__(
        self,
        embed_dims: Sequence[int],
        depths: Sequence[int],
        in_channels: int = 3,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dims = tuple(embed_dims)
        self.depths = tuple(depths)
        heads = (1, 2, 5, 8)
        sr_ratios = (8, 4, 2, 1)

        self.patch_embed1 = OverlapPatchEmbed(in_channels, embed_dims[0], 7, 4)
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        for stage in range(4):
            blocks = nn.ModuleList(
                TransformerBlock(
                    dim=embed_dims[stage],
                    num_heads=heads[stage],
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=rates[offset + index],
                    sr_ratio=sr_ratios[stage],
                )
                for index in range(depths[stage])
            )
            setattr(self, f"block{stage + 1}", blocks)
            setattr(self, f"norm{stage + 1}", nn.LayerNorm(embed_dims[stage], eps=1e-6))
            offset += depths[stage]
        self.apply(init_mit_weights)

    def forward(self, x: Tensor) -> list[Tensor]:
        batch = x.shape[0]
        outputs: list[Tensor] = []
        for stage in range(1, 5):
            x, height, width = getattr(self, f"patch_embed{stage}")(x)
            for block in getattr(self, f"block{stage}"):
                x = block(x, height, width)
            x = getattr(self, f"norm{stage}")(x)
            x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs


def build_mit_backbone(
    variant: str,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
) -> MixVisionTransformer:
    variant = variant.lower().removeprefix("mit_")
    if variant not in MIT_CONFIGS:
        raise ValueError(f"Unknown MiT variant {variant!r}; choose from {tuple(MIT_CONFIGS)}")
    config = MIT_CONFIGS[variant]
    return MixVisionTransformer(
        config.embed_dims,
        config.depths,
        in_channels,
        drop_rate,
        attn_drop_rate,
        drop_path_rate,
    )


__all__ = ["MIT_CONFIGS", "MiTConfig", "MixVisionTransformer", "build_mit_backbone"]

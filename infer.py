"""Run SegFormer inference on aligned NumPy modality images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from .datasets import ClassInfo, load_class_info
from .datasets.modalities import (
    MODALITIES, MODALITY_CHOICES, canonical_modality, load_modality_image,
    normalization_stats, preprocess_image, preview_image,
)
from .model import MIT_CONFIGS, SegFormer, build_segformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Material SegFormer inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--input", required=True, help="Modality NumPy file or sample/dataset directory")
    paths.add_argument(
        "--input-pattern",
        default="",
        help="Directory glob; empty selects the chosen modality's source files",
    )
    paths.add_argument("--checkpoint", required=True)
    paths.add_argument("--info-file", required=True)
    paths.add_argument("--output-dir", required=True)

    model = parser.add_argument_group("model and input")
    model.add_argument("--modality", choices=("auto", *MODALITY_CHOICES), default="auto")
    model.add_argument("--variant", choices=("auto", *MIT_CONFIGS), default="auto")
    model.add_argument("--decoder-dim", type=int, default=0)
    model.add_argument("--image-height", type=int, default=512)
    model.add_argument("--image-width", type=int, default=512)
    model.add_argument("--batch-size", type=int, default=4)
    model.add_argument("--mean", type=float, nargs="+", default=None)
    model.add_argument("--std", type=float, nargs="+", default=None)
    model.add_argument("--device", default="cuda", help="cuda, cuda:N, cpu, or auto")
    model.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    output = parser.add_argument_group("outputs")
    output.add_argument("--overlay-alpha", type=float, default=0.5)
    output.add_argument("--save-class-ids", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--save-color-mask", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--save-confidence", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_height <= 0 or args.image_width <= 0 or args.batch_size <= 0:
        raise ValueError("Image dimensions and batch size must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")
    if not any(
        (args.save_class_ids, args.save_color_mask, args.save_overlay, args.save_confidence)
    ):
        raise ValueError("At least one output type must be enabled")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary")
    for key in ("model", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break
    state_dict = {}
    for key, value in checkpoint.items():
        if isinstance(value, Tensor):
            state_dict[str(key).removeprefix("module.").removeprefix("model.")] = value
    if not state_dict:
        raise ValueError("No model tensors found in checkpoint")
    return state_dict


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    requested_variant: str,
    requested_decoder_dim: int,
    class_info: ClassInfo,
    requested_modality: str = "auto",
) -> tuple[SegFormer, dict[str, Any]]:
    checkpoint = torch_load(checkpoint_path)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, Mapping) else {}
    saved_modality = checkpoint_args.get("modality")
    modality = canonical_modality(
        (saved_modality or "rgb") if requested_modality == "auto" else requested_modality
    )
    if saved_modality is not None and canonical_modality(saved_modality) != modality:
        raise ValueError(f"Checkpoint modality is {saved_modality}, but {modality} was requested")
    state_dict = extract_state_dict(checkpoint)
    in_channels = state_dict["backbone.patch_embed1.proj.weight"].shape[1]
    if in_channels != MODALITIES[modality].channels:
        raise ValueError(
            f"Checkpoint expects {in_channels} input channels, but {modality} has "
            f"{MODALITIES[modality].channels}. Set --modality for weights-only checkpoints."
        )
    mean, std = normalization_stats(modality, checkpoint_args.get("mean"), checkpoint_args.get("std"))
    if requested_variant == "auto":
        variant = str(checkpoint_args.get("variant", "b0"))
    else:
        variant = requested_variant
    decoder_dim = requested_decoder_dim
    if decoder_dim == 0:
        decoder_dim = int(checkpoint_args.get("decoder_dim", 0))

    saved_names = checkpoint.get("class_names") if isinstance(checkpoint, Mapping) else None
    if saved_names is not None and list(saved_names) != list(class_info.names):
        raise ValueError("Checkpoint classes differ from the supplied info.json")
    model = build_segformer(
        variant=variant,
        num_classes=class_info.num_classes,
        in_channels=in_channels,
        decoder_dim=decoder_dim or None,
    )
    model.load_state_dict(state_dict, strict=True)
    return model, {
        "variant": variant, "decoder_dim": decoder_dim, "modality": modality,
        "in_channels": in_channels, "mean": mean, "std": std,
    }


def discover_images(
    input_path: str | Path, pattern: str, modality: str = "rgb"
) -> tuple[Path, list[Path]]:
    input_path = Path(input_path)
    if input_path.is_file():
        return input_path.parent, [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Inference input does not exist: {input_path}")
    use_default = not pattern
    pattern = pattern or f"**/{MODALITIES[canonical_modality(modality)].files[0]}"
    images = sorted(path for path in input_path.glob(pattern) if path.is_file())
    if use_default and modality == "dolp":
        # Older samples use polar_dolp.npy. Prefer dolp.npy if both names exist.
        images.extend(
            path for path in input_path.glob("**/light_on/aligned/npy/polar_dolp.npy")
            if path.is_file() and not path.with_name("dolp.npy").is_file()
        )
        images.sort()
    if not images:
        raise FileNotFoundError(f"No images matching {pattern!r} under {input_path}")
    return input_path, images


def prediction_paths(output_dir: Path, relative_path: Path) -> dict[str, Path]:
    parent = output_dir / relative_path.parent
    stem = relative_path.stem
    return {
        "class_ids": parent / f"{stem}_class_ids.png",
        "color_mask": parent / f"{stem}_color_mask.png",
        "overlay": parent / f"{stem}_overlay.png",
        "confidence": parent / f"{stem}_confidence.png",
    }


def save_prediction(
    image: Image.Image,
    prediction: np.ndarray,
    confidence: np.ndarray,
    colors: np.ndarray,
    paths: Mapping[str, Path],
    args: argparse.Namespace,
) -> dict[str, str]:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    class_image = Image.fromarray(prediction.astype(np.uint8), mode="L")
    color_array = colors[prediction]
    color_image = Image.fromarray(color_array.astype(np.uint8), mode="RGB")
    saved: dict[str, str] = {}
    if args.save_class_ids:
        class_image.save(paths["class_ids"])
        saved["class_ids"] = str(paths["class_ids"])
    if args.save_color_mask:
        color_image.save(paths["color_mask"])
        saved["color_mask"] = str(paths["color_mask"])
    if args.save_overlay:
        overlay = Image.blend(image, color_image, args.overlay_alpha)
        overlay.save(paths["overlay"])
        saved["overlay"] = str(paths["overlay"])
    if args.save_confidence:
        confidence_image = Image.fromarray(
            np.clip(confidence * 255.0, 0, 255).astype(np.uint8), mode="L"
        )
        confidence_image.save(paths["confidence"])
        saved["confidence"] = str(paths["confidence"])
    return saved


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    class_info = load_class_info(args.info_file)
    model, model_config = build_model_from_checkpoint(
        args.checkpoint, args.variant, args.decoder_dim, class_info, args.modality
    )
    args.modality = model_config["modality"]
    args.mean, args.std = normalization_stats(
        args.modality,
        model_config["mean"] if args.mean is None else args.mean,
        model_config["std"] if args.std is None else args.std,
    )
    model.to(device).eval()
    amp = args.amp and device.type == "cuda"

    input_root, images = discover_images(args.input, args.input_pattern, args.modality)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(class_info.colors, dtype=np.uint8)
    records: list[dict[str, Any]] = []

    print(
        f"Checkpoint={args.checkpoint} variant=MiT-{model_config['variant'].upper()} "
        f"modality={args.modality} channels={model_config['in_channels']} "
        f"device={device} images={len(images)} AMP={amp}"
    )
    for start in range(0, len(images), args.batch_size):
        batch_paths = images[start : start + args.batch_size]
        originals: list[Image.Image] = []
        tensors: list[Tensor] = []
        for image_path in batch_paths:
            image = load_modality_image(image_path, args.modality)
            originals.append(preview_image(image, args.modality))
            tensors.append(
                preprocess_image(
                    image,
                    (args.image_height, args.image_width),
                    args.mean,
                    args.std,
                )
            )
        inputs = torch.stack(tensors).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            probabilities = model(inputs).softmax(dim=1)
        confidence_batch, prediction_batch = probabilities.max(dim=1)

        for image_path, image, prediction, confidence in zip(
            batch_paths, originals, prediction_batch, confidence_batch
        ):
            original_size = image.size
            prediction_image = Image.fromarray(prediction.cpu().numpy().astype(np.uint8), mode="L")
            prediction_image = prediction_image.resize(original_size, Image.Resampling.NEAREST)
            confidence_image = Image.fromarray(confidence.cpu().numpy().astype(np.float32), mode="F")
            confidence_image = confidence_image.resize(original_size, Image.Resampling.BILINEAR)
            prediction_array = np.asarray(prediction_image, dtype=np.uint8)
            confidence_array = np.asarray(confidence_image, dtype=np.float32)
            relative = image_path.relative_to(input_root)
            paths = prediction_paths(output_dir, relative)
            saved = save_prediction(
                image,
                prediction_array,
                confidence_array,
                colors,
                paths,
                args,
            )
            present_ids = sorted(int(value) for value in np.unique(prediction_array))
            records.append(
                {
                    "input": str(image_path),
                    "width": original_size[0],
                    "height": original_size[1],
                    "present_class_ids": present_ids,
                    "present_classes": [class_info.names[value] for value in present_ids],
                    "mean_confidence": float(confidence_array.mean()),
                    "outputs": saved,
                }
            )
            print(f"[{len(records):04d}/{len(images):04d}] {image_path}")

    manifest = {
        "checkpoint": str(args.checkpoint),
        "variant": model_config["variant"],
        "modality": args.modality,
        "in_channels": model_config["in_channels"],
        "mean": list(args.mean),
        "std": list(args.std),
        "num_classes": class_info.num_classes,
        "class_names": list(class_info.names),
        "class_colors": [list(color) for color in class_info.colors],
        "input_size": [args.image_height, args.image_width],
        "records": records,
    }
    manifest_path = output_dir / "inference_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    temporary.replace(manifest_path)
    print(f"Saved {len(records)} predictions and manifest to {output_dir}")


if __name__ == "__main__":
    main()

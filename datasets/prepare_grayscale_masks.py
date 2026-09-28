"""Convert color semantic masks to single-channel class-ID PNG masks.

Each output is validated against the global class/color metadata and the
sample's COCO categories, annotations, dimensions, and segmentation pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .material_dataset import ClassInfo, load_class_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare grayscale class-ID masks from color masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--output-dirname", default="grayscale")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--missing-mask-report", required=True)
    parser.add_argument("--splits", choices=("all", "train", "val"), default="all")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-missing-masks", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--validate-coco-pixels", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--max-coco-mismatch-ratio",
        type=float,
        default=0.001,
        help="Maximum fraction of visible foreground pixels not covered by same-class COCO masks",
    )
    return parser.parse_args()


def selected_sets(split_file: str | Path, splits: str) -> list[str]:
    with Path(split_file).open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    keys = ("train_sets", "val_sets") if splits == "all" else (f"{splits}_sets",)
    names: list[str] = []
    for key in keys:
        if key not in split_data:
            raise KeyError(f"Missing {key!r} in {split_file}")
        names.extend(str(value) for value in split_data[key])
    if len(names) != len(set(names)):
        raise ValueError("A dataset set appears in more than one selected split")
    return names


def validate_categories(coco: Mapping[str, Any], classes: ClassInfo, coco_path: Path) -> None:
    categories = {int(entry["id"]): str(entry["name"]) for entry in coco["categories"]}
    expected = {class_id: classes.names[class_id] for class_id in range(1, classes.num_classes)}
    if categories != expected:
        raise ValueError(
            f"COCO categories in {coco_path} differ from info.json: "
            f"expected={expected}, found={categories}"
        )


def color_to_labels(color_mask: np.ndarray, classes: ClassInfo, sample_id: str) -> np.ndarray:
    if color_mask.ndim != 3 or color_mask.shape[2] != 3:
        raise ValueError(f"Color mask for {sample_id} is not RGB: {color_mask.shape}")
    labels = np.full(color_mask.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(classes.colors):
        labels[np.all(color_mask == color, axis=2)] = class_id
    unknown = labels == 255
    if bool(unknown.any()):
        colors = np.unique(color_mask[unknown].reshape(-1, 3), axis=0)
        raise ValueError(
            f"Color mask for {sample_id} contains colors absent from info.json: "
            f"{colors[:20].tolist()}"
        )
    return labels


def decode_uncompressed_rle(segmentation: Mapping[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in segmentation["size"])
    counts = segmentation["counts"]
    if not isinstance(counts, list):
        raise TypeError(
            "Compressed COCO RLE requires pycocotools; this dataset is expected to use "
            "uncompressed integer counts"
        )
    flat = np.zeros(height * width, dtype=bool)
    cursor = 0
    foreground = False
    for raw_run in counts:
        run = int(raw_run)
        if run < 0 or cursor + run > flat.size:
            raise ValueError("Invalid COCO RLE run length")
        if foreground:
            flat[cursor : cursor + run] = True
        cursor += run
        foreground = not foreground
    if cursor != flat.size:
        raise ValueError(f"COCO RLE decodes to {cursor} pixels, expected {flat.size}")
    return flat.reshape((height, width), order="F")


def decode_polygons(
    polygons: Sequence[Sequence[float]], height: int, width: int
) -> np.ndarray:
    canvas = Image.new("1", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError("Invalid COCO polygon")
        points = list(zip(polygon[0::2], polygon[1::2]))
        drawer.polygon(points, fill=1)
    return np.asarray(canvas, dtype=bool)


def coco_annotation_ids(
    coco: Mapping[str, Any], classes: ClassInfo, coco_path: Path
) -> set[int]:
    annotated_ids: set[int] = set()
    for annotation in coco.get("annotations", []):
        category_id = int(annotation["category_id"])
        class_id = int(annotation.get("class_id", category_id))
        if class_id != category_id or not 0 < class_id < classes.num_classes:
            raise ValueError(f"Invalid class/category ID in {coco_path}: {annotation}")
        if annotation.get("class_name", classes.names[class_id]) != classes.names[class_id]:
            raise ValueError(f"Class name mismatch for ID {class_id} in {coco_path}")
        annotated_ids.add(class_id)
    return annotated_ids


def coco_class_coverage(
    coco: Mapping[str, Any], classes: ClassInfo, shape: tuple[int, int], coco_path: Path
) -> dict[int, np.ndarray]:
    height, width = shape
    coverage: dict[int, np.ndarray] = {}
    for annotation in coco.get("annotations", []):
        category_id = int(annotation["category_id"])
        class_id = int(annotation.get("class_id", category_id))
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, Mapping):
            instance = decode_uncompressed_rle(segmentation)
        elif isinstance(segmentation, list):
            instance = decode_polygons(segmentation, height, width)
        else:
            raise TypeError(f"Unsupported COCO segmentation in {coco_path}")
        if instance.shape != shape:
            raise ValueError(
                f"COCO segmentation shape {instance.shape} differs from mask {shape} in {coco_path}"
            )
        if class_id not in coverage:
            coverage[class_id] = np.zeros(shape, dtype=bool)
        coverage[class_id] |= instance
    return coverage


def iter_object_dirs(data_root: Path, set_names: Iterable[str]) -> Iterable[tuple[str, Path]]:
    for set_name in set_names:
        set_dir = data_root / set_name
        if not set_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset set: {set_dir}")
        for object_dir in sorted(path for path in set_dir.iterdir() if path.is_dir()):
            yield f"{set_name}/{object_dir.name}", object_dir


def save_png_atomic(labels: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    Image.fromarray(labels, mode="L").save(temporary, format="PNG", optimize=True)
    temporary.replace(output_path)


def prepare_one(
    sample_id: str,
    object_dir: Path,
    classes: ClassInfo,
    output_dirname: str,
    overwrite: bool,
    validate_coco_pixels: bool,
    max_coco_mismatch_ratio: float,
) -> tuple[Path, list[int], bool, int, float]:
    object_id = object_dir.name
    ground_truth = object_dir / "segmentation_ground_truth"
    color_path = ground_truth / "rgb" / f"{object_id}_rgb_mask.png"
    coco_path = ground_truth / "coco_rgb.json"
    image_path = object_dir / "light_on" / "aligned" / "png" / "rgb.png"
    output_path = ground_truth / output_dirname / f"{object_id}_rgb_mask.png"

    # Normal training startup uses --no-overwrite. A previously prepared mask
    # is already PNG-authoritative, so validate its IDs without reparsing large
    # COCO RLE payloads or requiring the source annotations again.
    if output_path.is_file() and not overwrite:
        with Image.open(output_path) as existing_file:
            if existing_file.mode not in {"L", "I", "P"}:
                raise ValueError(f"Prepared mask is not single-channel: {output_path}")
            existing = np.asarray(existing_file.convert("L"), dtype=np.uint8)
        present_ids = sorted(int(value) for value in np.unique(existing) if value != 0)
        invalid = [value for value in present_ids if value >= classes.num_classes]
        if invalid:
            raise ValueError(f"Prepared mask has invalid class IDs {invalid}: {output_path}")
        return output_path, present_ids, False, 0, 0.0

    for path in (image_path, color_path, coco_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input for {sample_id}: {path}")

    with Image.open(image_path) as image_file:
        image_size = image_file.size
    with Image.open(color_path) as color_file:
        color_mask = np.asarray(color_file.convert("RGB"), dtype=np.uint8)
    if image_size != (color_mask.shape[1], color_mask.shape[0]):
        raise ValueError(f"Image/color-mask size mismatch for {sample_id}")
    labels = color_to_labels(color_mask, classes, sample_id)

    with coco_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    validate_categories(coco, classes, coco_path)
    images = coco.get("images", [])
    if len(images) != 1:
        raise ValueError(f"Expected one COCO image record in {coco_path}")
    if (int(images[0]["width"]), int(images[0]["height"])) != image_size:
        raise ValueError(f"COCO image dimensions differ for {sample_id}")

    annotated_ids = coco_annotation_ids(coco, classes, coco_path)
    color_ids = set(int(value) for value in np.unique(labels) if value != 0)
    if color_ids != annotated_ids:
        raise ValueError(
            f"Color-mask classes and COCO annotation classes differ for {sample_id}: "
            f"color={sorted(color_ids)}, coco={sorted(annotated_ids)}"
        )
    differing = 0
    mismatch_ratio = 0.0
    if validate_coco_pixels:
        coverage = coco_class_coverage(coco, classes, labels.shape, coco_path)
        # Only validate visible foreground. The color PNG remains authoritative
        # for background and holes, which COCO instance geometry may not retain.
        consistent = np.zeros(labels.shape, dtype=bool)
        for class_id, class_coverage in coverage.items():
            selected = labels == class_id
            consistent[selected] = class_coverage[selected]
        consistent[labels == 0] = True
        differing = int(np.count_nonzero(~consistent))
        mismatch_ratio = differing / labels.size
        if mismatch_ratio > max_coco_mismatch_ratio:
            raise ValueError(
                f"The color mask has {differing} visible pixels without same-class COCO "
                f"coverage ({mismatch_ratio:.6%}) for {sample_id}, exceeding the allowed "
                f"{max_coco_mismatch_ratio:.6%}"
            )

    save_png_atomic(labels, output_path)
    return output_path, sorted(color_ids), True, differing, mismatch_ratio


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.max_coco_mismatch_ratio <= 1.0:
        raise ValueError("--max-coco-mismatch-ratio must be in [0, 1]")
    data_root = Path(args.data_root)
    classes = load_class_info(args.info_file)
    if classes.num_classes > 255:
        raise ValueError("Single-channel 8-bit PNG supports at most 255 class IDs")
    set_names = selected_sets(args.split_file, args.splits)
    records = []
    missing_records = []
    written = 0
    for index, (sample_id, object_dir) in enumerate(
        iter_object_dirs(data_root, set_names), start=1
    ):
        try:
            output_path, present_ids, wrote, differing, mismatch_ratio = prepare_one(
                sample_id,
                object_dir,
                classes,
                args.output_dirname,
                args.overwrite,
                args.validate_coco_pixels,
                args.max_coco_mismatch_ratio,
            )
        except FileNotFoundError as error:
            ground_truth = object_dir / "segmentation_ground_truth"
            missing_record = {
                "sample_id": sample_id,
                "reason": str(error),
                "image_path": str(object_dir / "light_on" / "aligned" / "png" / "rgb.png"),
                "color_mask_path": str(
                    ground_truth / "rgb" / f"{object_dir.name}_rgb_mask.png"
                ),
                "coco_path": str(ground_truth / "coco_rgb.json"),
                "prepared_mask_path": str(
                    ground_truth
                    / args.output_dirname
                    / f"{object_dir.name}_rgb_mask.png"
                ),
            }
            missing_records.append(missing_record)
            print(f"[{index:03d}] SKIP {sample_id}: {error}")
            continue
        written += int(wrote)
        records.append(
            {
                "sample_id": sample_id,
                "mask": str(output_path),
                "present_class_ids": present_ids,
                "coco_mismatch_pixels": differing,
                "coco_mismatch_ratio": mismatch_ratio,
            }
        )
        if wrote:
            print(f"[{index:04d}] PREPARED {sample_id} -> {output_path}")

    manifest = {
        "format": "single-channel uint8 PNG; pixel value equals class ID",
        "background_id": 0,
        "output_dirname": args.output_dirname,
        "num_classes": classes.num_classes,
        "class_names": list(classes.names),
        "class_colors": [list(color) for color in classes.colors],
        "samples": records,
        "missing_samples": missing_records,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    temporary.replace(manifest_path)

    missing_report_path = Path(args.missing_mask_report)
    missing_report_path.parent.mkdir(parents=True, exist_ok=True)
    missing_report = {
        "preparation": {
            "policy": "skip" if args.skip_missing_masks else "fail",
            "missing_count": len(missing_records),
            "missing_samples": missing_records,
        }
    }
    report_temporary = missing_report_path.with_suffix(missing_report_path.suffix + ".tmp")
    with report_temporary.open("w", encoding="utf-8") as handle:
        json.dump(missing_report, handle, indent=2)
        handle.write("\n")
    report_temporary.replace(missing_report_path)

    if missing_records and not args.skip_missing_masks:
        raise FileNotFoundError(
            f"Found {len(missing_records)} samples with missing inputs; "
            f"details were saved to {missing_report_path}"
        )
    print(
        f"Prepared {len(records)} masks ({written} written), skipped "
        f"{len(missing_records)} missing samples, and saved manifest to {manifest_path}"
    )


if __name__ == "__main__":
    main()

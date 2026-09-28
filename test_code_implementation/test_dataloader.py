from pathlib import Path
import argparse
import sys

import torch

# When this file is run directly, Python adds test_code_implementation/ to
# sys.path, but not the repository root where the datasets package lives.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import build_dataloaders
from datasets.modalities import MODALITY_CHOICES


def main():
    parser = argparse.ArgumentParser(description="Inspect batches for a selected modality")
    parser.add_argument("--modality", choices=MODALITY_CHOICES, default="rgb")
    args = parser.parse_args()
    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    DATA_ROOT = "/media/voyager/7e280623-1817-45d4-9e64-b992b9c41c77/base_model/dataset/mat_seg_dataset"
    SPLIT_FILE = "/media/voyager/7e280623-1817-45d4-9e64-b992b9c41c77/base_model/dataset/split.json"
    INFO_FILE = "/media/voyager/7e280623-1817-45d4-9e64-b992b9c41c77/base_model/dataset/info.json"

    # ---------------------------------------------------------
    # Dataloader configuration
    # ---------------------------------------------------------
    train_loader, val_loader, class_info = build_dataloaders(
        data_root=DATA_ROOT,
        split_file=SPLIT_FILE,
        info_file=INFO_FILE,
        modality=args.modality,

        mask_dirname="grayscale",
        missing_mask_report="missing_samples.json",
        skip_missing_masks=False,

        height=512,
        width=512,

        batch_size=4,
        num_workers=2,
        seed=42,

        mean=None,
        std=None,

        hflip_prob=0.5,
        vflip_prob=0.0,

        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.0,

        ignore_index=255,
        strict_labels=True,

        pin_memory=True,
        persistent_workers=True,
    )

    # ---------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Number of classes : {class_info.num_classes}")
    print(f"Modality          : {train_loader.dataset.modality}")
    print(f"Input channels    : {train_loader.dataset.in_channels}")
    print(f"Class names       : {class_info.names}")

    print(f"\nTraining samples   : {len(train_loader.dataset)}")
    print(f"Validation samples : {len(val_loader.dataset)}")

    print(f"\nTraining batches   : {len(train_loader)}")
    print(f"Validation batches : {len(val_loader)}")

    # ---------------------------------------------------------
    # Missing samples
    # ---------------------------------------------------------
    train_missing = train_loader.dataset.missing_samples
    val_missing = val_loader.dataset.missing_samples

    print(f"\nMissing train samples : {len(train_missing)}")
    print(f"Missing val samples   : {len(val_missing)}")

    # ---------------------------------------------------------
    # Check one training batch
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING BATCH TEST")
    print("=" * 70)

    train_batch = next(iter(train_loader))

    images = train_batch["image"]
    masks = train_batch["mask"]
    sample_ids = train_batch["sample_id"]

    print(f"Image shape       : {images.shape}")
    print(f"Mask shape        : {masks.shape}")
    print(f"Image dtype       : {images.dtype}")
    print(f"Mask dtype        : {masks.dtype}")

    print(f"Image min         : {images.min().item():.4f}")
    print(f"Image max         : {images.max().item():.4f}")
    print(f"Image mean        : {images.mean().item():.4f}")
    print(f"Image std         : {images.std().item():.4f}")

    print(f"\nSample IDs:")
    for sample_id in sample_ids:
        print(f"  {sample_id}")

    # ---------------------------------------------------------
    # Check mask labels
    # ---------------------------------------------------------
    unique_labels = torch.unique(masks)

    print("\nUnique mask labels in batch:")
    print(unique_labels.tolist())

    valid_labels = set(range(class_info.num_classes))
    invalid_labels = [
        int(x)
        for x in unique_labels.tolist()
        if int(x) not in valid_labels and int(x) != 255
    ]

    if invalid_labels:
        print(f"ERROR: Invalid labels found: {invalid_labels}")
    else:
        print("Mask labels: OK")

    # ---------------------------------------------------------
    # Check validation batch
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION BATCH TEST")
    print("=" * 70)

    val_batch = next(iter(val_loader))

    val_images = val_batch["image"]
    val_masks = val_batch["mask"]
    val_sample_ids = val_batch["sample_id"]

    print(f"Image shape       : {val_images.shape}")
    print(f"Mask shape        : {val_masks.shape}")
    print(f"Image dtype       : {val_images.dtype}")
    print(f"Mask dtype        : {val_masks.dtype}")

    print(f"Image min         : {val_images.min().item():.4f}")
    print(f"Image max         : {val_images.max().item():.4f}")
    print(f"Image mean        : {val_images.mean().item():.4f}")
    print(f"Image std         : {val_images.std().item():.4f}")

    print("\nSample IDs:")
    for sample_id in val_sample_ids:
        print(f"  {sample_id}")

    print("\nUnique validation mask labels:")
    print(torch.unique(val_masks).tolist())

    # ---------------------------------------------------------
    # Final shape checks
    # ---------------------------------------------------------
    expected_image_shape = (
        images.shape[0],
        train_loader.dataset.in_channels,
        512,
        512,
    )

    expected_mask_shape = (
        masks.shape[0],
        512,
        512,
    )

    assert images.shape == expected_image_shape, (
        f"Unexpected image shape: {images.shape}"
    )

    assert masks.shape == expected_mask_shape, (
        f"Unexpected mask shape: {masks.shape}"
    )

    assert images.dtype == torch.float32
    assert masks.dtype == torch.int64

    print("\n" + "=" * 70)
    print("ALL DATALOADER TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()

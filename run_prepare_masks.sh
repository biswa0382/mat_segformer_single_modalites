#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="dataset/mat_seg_dataset"
SPLIT_FILE="dataset/split.json"
INFO_FILE="dataset/info.json"
OUTPUT_DIRNAME="grayscale"
MANIFEST="dataset/grayscale_masks_manifest.json"
MISSING_MASK_REPORT="dataset/missing_masks.json"
SPLITS="all"
OVERWRITE=true
VALIDATE_COCO_PIXELS=false
MAX_COCO_MISMATCH_RATIO=0.001
SKIP_MISSING_MASKS=true

OVERWRITE_FLAG="--no-overwrite"
VALIDATE_FLAG="--no-validate-coco-pixels"
SKIP_MISSING_FLAG="--no-skip-missing-masks"
[[ "${OVERWRITE}" == "true" ]] && OVERWRITE_FLAG="--overwrite"
[[ "${VALIDATE_COCO_PIXELS}" == "true" ]] && VALIDATE_FLAG="--validate-coco-pixels"
[[ "${SKIP_MISSING_MASKS}" == "true" ]] && SKIP_MISSING_FLAG="--skip-missing-masks"

"${PYTHON_BIN}" -m mat_segformer.datasets.prepare_grayscale_masks \
    --data-root "${DATA_ROOT}" \
    --split-file "${SPLIT_FILE}" \
    --info-file "${INFO_FILE}" \
    --output-dirname "${OUTPUT_DIRNAME}" \
    --manifest "${MANIFEST}" \
    --missing-mask-report "${MISSING_MASK_REPORT}" \
    --splits "${SPLITS}" \
    "${OVERWRITE_FLAG}" \
    "${SKIP_MISSING_FLAG}" \
    "${VALIDATE_FLAG}" \
    --max-coco-mismatch-ratio "${MAX_COCO_MISMATCH_RATIO}" \
    "$@"

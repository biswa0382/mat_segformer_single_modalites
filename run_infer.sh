#!/usr/bin/env bash
set -euo pipefail

# All inference settings are controlled here.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Paths -----------------------------------------------------------------------
INPUT_PATH="dataset/mat_seg_dataset/set_17/0721"
INPUT_PATTERN=""                 # Empty: use the selected modality's NumPy files
CHECKPOINT="checkpoints/mat_segformer/best.pth"
INFO_FILE="dataset/info.json"
OUTPUT_DIR="inference/mat_segformer"

# Model/input -----------------------------------------------------------------
MODALITY="auto"                  # checkpoint setting; or rgb | nir | aolp | dolp | polar | pol | thermal_static
VARIANT="auto"                   # auto reads a full checkpoint; or b0 ... b5
DECODER_DIM=0                     # 0 reads checkpoint/default configuration
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
BATCH_SIZE=4
MEAN=()                          # Empty: checkpoint values or modality defaults
STD=()
DEVICE="cuda"                    # cuda | cuda:N | cpu | auto
AMP=true

# Outputs ---------------------------------------------------------------------
OVERLAY_ALPHA=0.5
SAVE_CLASS_IDS=true
SAVE_COLOR_MASK=true
SAVE_OVERLAY=true
SAVE_CONFIDENCE=true

bool_flag() {
    local value="$1"
    local positive="$2"
    local negative="$3"
    if [[ "${value}" == "true" ]]; then
        printf '%s' "${positive}"
    else
        printf '%s' "${negative}"
    fi
}

NORMALIZATION_ARGS=()
if (( ${#MEAN[@]} )); then NORMALIZATION_ARGS+=(--mean "${MEAN[@]}"); fi
if (( ${#STD[@]} )); then NORMALIZATION_ARGS+=(--std "${STD[@]}"); fi

"${PYTHON_BIN}" -m mat_segformer.infer \
    --input "${INPUT_PATH}" \
    --input-pattern "${INPUT_PATTERN}" \
    --checkpoint "${CHECKPOINT}" \
    --info-file "${INFO_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant "${VARIANT}" \
    --modality "${MODALITY}" \
    --decoder-dim "${DECODER_DIM}" \
    --image-height "${IMAGE_HEIGHT}" \
    --image-width "${IMAGE_WIDTH}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    "$(bool_flag "${AMP}" --amp --no-amp)" \
    --overlay-alpha "${OVERLAY_ALPHA}" \
    "$(bool_flag "${SAVE_CLASS_IDS}" --save-class-ids --no-save-class-ids)" \
    "$(bool_flag "${SAVE_COLOR_MASK}" --save-color-mask --no-save-color-mask)" \
    "$(bool_flag "${SAVE_OVERLAY}" --save-overlay --no-save-overlay)" \
    "$(bool_flag "${SAVE_CONFIDENCE}" --save-confidence --no-save-confidence)" \
    "${NORMALIZATION_ARGS[@]}" "$@"

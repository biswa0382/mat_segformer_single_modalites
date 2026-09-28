#!/usr/bin/env bash
set -euo pipefail

# Every training/data/model hyperparameter is configured in this file.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_ID="${GPU_ID:-1}"             # Select one GPU; e.g. GPU_ID=2 bash run_train.sh

# Paths -----------------------------------------------------------------------
DATA_ROOT="dataset/mat_seg_dataset"
SPLIT_FILE="dataset/split.json"
INFO_FILE="dataset/info.json"
MASK_DIRNAME="grayscale"
MASK_MANIFEST="dataset/grayscale_masks_manifest.json"
MISSING_MASK_REPORT="dataset/missing_masks.json"
OUTPUT_DIR="checkpoints/mat_seg_polar"
PRETRAINED_DIR="checkpoints/pretrained/segformer"
RESUME=""  # resumes at epoch 6



# Prepare class-ID masks before training --------------------------------------
PREPARE_MASKS=true
PREPARE_SPLITS="all"              # all | train | val
OVERWRITE_MASKS=false
VALIDATE_COCO_PIXELS=false         # COCO does not preserve all PNG holes
MAX_COCO_MISMATCH_RATIO=0.001      # PNG stays authoritative; COCO only validates
SKIP_MISSING_MASKS=true            # log incomplete samples and continue

# Model -----------------------------------------------------------------------
VARIANT="b0"                      # b0 | b1 | b2 | b3 | b4 | b5
DECODER_DIM=0                     # 0 selects the official variant default
DROP_RATE=0.0
ATTN_DROP_RATE=0.0
DROP_PATH_RATE=0.1
HEAD_DROPOUT=0.1

# Input and augmentation ------------------------------------------------------
MODALITY="pol"                  # rgb | nir | aolp | dolp | polar (aolp.npy,dolp.npy) | pol (pol.npy) | thermal_static
# aolp: [cos(2*theta), sin(2*theta)]; polar adds DoLP as channel 3.
# dolp: single-channel DoLP; pol: polarization intensity from polar.npy.
# thermal_static: saved first thermal frame.
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
BATCH_SIZE=8
NUM_WORKERS=4
MEAN=()                          # Empty: choose defaults for MODALITY
STD=()                           # Custom arrays need one value per input channel
HFLIP_PROB=0.5
VFLIP_PROB=0.0
BRIGHTNESS=0.2
CONTRAST=0.2
SATURATION=0.2
HUE=0.05
IGNORE_INDEX=255
STRICT_LABELS=true
PIN_MEMORY=true
PERSISTENT_WORKERS=true

# Optimization ----------------------------------------------------------------
EPOCHS=200
SEED=42
DEVICE="cuda:0"                  # Selected GPU is remapped to cuda:0; also cpu | auto
DETERMINISTIC=false
AMP=true
ACCUMULATION_STEPS=1
BACKBONE_LR=0.00006
HEAD_LR=0.0006
WEIGHT_DECAY=0.01
BETA1=0.9
BETA2=0.999
LABEL_SMOOTHING=0.0
GRAD_CLIP=1.0                    # 0 disables gradient clipping
SCHEDULER="poly"                 # poly | cosine | constant
WARMUP_EPOCHS=2.0
WARMUP_START_RATIO=0.01
MIN_LR_RATIO=0.01
POLY_POWER=1.0

# Metrics, logging, and checkpoints ------------------------------------------
INCLUDE_BACKGROUND=false         # false: mIoU/Dice average material classes only
LOG_INTERVAL=5

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    printf 'GPU_ID must be a single non-negative integer, got: %s\n' "${GPU_ID}" >&2
    exit 1
fi
# Set visibility before either Python process imports PyTorch or initializes CUDA.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

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

if [[ "${PREPARE_MASKS}" == "true" ]]; then
    "${PYTHON_BIN}" -m mat_segformer.datasets.prepare_grayscale_masks \
        --data-root "${DATA_ROOT}" \
        --split-file "${SPLIT_FILE}" \
        --info-file "${INFO_FILE}" \
        --output-dirname "${MASK_DIRNAME}" \
        --manifest "${MASK_MANIFEST}" \
        --missing-mask-report "${MISSING_MASK_REPORT}" \
        --splits "${PREPARE_SPLITS}" \
        "$(bool_flag "${OVERWRITE_MASKS}" --overwrite --no-overwrite)" \
        "$(bool_flag "${SKIP_MISSING_MASKS}" --skip-missing-masks --no-skip-missing-masks)" \
        "$(bool_flag "${VALIDATE_COCO_PIXELS}" --validate-coco-pixels --no-validate-coco-pixels)" \
        --max-coco-mismatch-ratio "${MAX_COCO_MISMATCH_RATIO}"
fi

PRETRAINED="${PRETRAINED_DIR}/mit_${VARIANT}.pth"
TRAIN_ARGS=(
    --data-root "${DATA_ROOT}"
    --split-file "${SPLIT_FILE}"
    --info-file "${INFO_FILE}"
    --mask-dirname "${MASK_DIRNAME}"
    --missing-mask-report "${MISSING_MASK_REPORT}"
    "$(bool_flag "${SKIP_MISSING_MASKS}" --skip-missing-masks --no-skip-missing-masks)"
    --output-dir "${OUTPUT_DIR}"
    --pretrained "${PRETRAINED}"
    --variant "${VARIANT}"
    --decoder-dim "${DECODER_DIM}"
    --drop-rate "${DROP_RATE}"
    --attn-drop-rate "${ATTN_DROP_RATE}"
    --drop-path-rate "${DROP_PATH_RATE}"
    --head-dropout "${HEAD_DROPOUT}"
    --modality "${MODALITY}"
    --image-height "${IMAGE_HEIGHT}"
    --image-width "${IMAGE_WIDTH}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --hflip-prob "${HFLIP_PROB}"
    --vflip-prob "${VFLIP_PROB}"
    --brightness "${BRIGHTNESS}"
    --contrast "${CONTRAST}"
    --saturation "${SATURATION}"
    --hue "${HUE}"
    --ignore-index "${IGNORE_INDEX}"
    "$(bool_flag "${STRICT_LABELS}" --strict-labels --no-strict-labels)"
    "$(bool_flag "${PIN_MEMORY}" --pin-memory --no-pin-memory)"
    "$(bool_flag "${PERSISTENT_WORKERS}" --persistent-workers --no-persistent-workers)"
    --epochs "${EPOCHS}"
    --seed "${SEED}"
    --device "${DEVICE}"
    "$(bool_flag "${DETERMINISTIC}" --deterministic --no-deterministic)"
    "$(bool_flag "${AMP}" --amp --no-amp)"
    --accumulation-steps "${ACCUMULATION_STEPS}"
    --backbone-lr "${BACKBONE_LR}"
    --head-lr "${HEAD_LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --beta1 "${BETA1}"
    --beta2 "${BETA2}"
    --label-smoothing "${LABEL_SMOOTHING}"
    --grad-clip "${GRAD_CLIP}"
    --scheduler "${SCHEDULER}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --warmup-start-ratio "${WARMUP_START_RATIO}"
    --min-lr-ratio "${MIN_LR_RATIO}"
    --poly-power "${POLY_POWER}"
    "$(bool_flag "${INCLUDE_BACKGROUND}" --include-background --no-include-background)"
    --log-interval "${LOG_INTERVAL}"
)

if (( ${#MEAN[@]} )); then TRAIN_ARGS+=(--mean "${MEAN[@]}"); fi
if (( ${#STD[@]} )); then TRAIN_ARGS+=(--std "${STD[@]}"); fi

if [[ -n "${RESUME}" ]]; then
    TRAIN_ARGS+=(--resume "${RESUME}")
fi

printf 'CUDA_VISIBLE_DEVICES=%s (selected GPU maps to cuda:0)\n' "${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" -m mat_segformer.train "${TRAIN_ARGS[@]}" "$@"

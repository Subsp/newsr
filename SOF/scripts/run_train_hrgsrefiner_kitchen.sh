#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PRIORS_DIR="${PRIORS_DIR:-${SCENE_ROOT}/priors}"
VGGT_ROOT="${VGGT_ROOT:-${WORK_ROOT}/vggt}"

GS_MODEL_PATH="${GS_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images2_v1/sof30k}"
ORACLE_ROOT="${ORACLE_ROOT:-${SCENE_ASSET_ROOT}/oracle/formal_oracle_sof30k}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_train_formal}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SCENE_NAME}_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/cache}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"

NUM_VIEWS="${NUM_VIEWS:-4}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-32}"
CAMERA_RESOLUTION="${CAMERA_RESOLUTION:-2}"
MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
EVAL_EVERY="${EVAL_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-20}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SEED="${SEED:-0}"

LOSS_DEPTH_W="${LOSS_DEPTH_W:-1.0}"
LOSS_NORMAL_W="${LOSS_NORMAL_W:-0.2}"
LOSS_MV_W="${LOSS_MV_W:-0.0}"
LOSS_DELTA_W="${LOSS_DELTA_W:-0.02}"
LOSS_SURFACE_W="${LOSS_SURFACE_W:-0.2}"
LOSS_UPDATE_W="${LOSS_UPDATE_W:-0.1}"
LOSS_DETAIL_W="${LOSS_DETAIL_W:-0.1}"
LOSS_CONF_W="${LOSS_CONF_W:-0.1}"
LOSS_PRIOR_COLOR_W="${LOSS_PRIOR_COLOR_W:-0.05}"

VGGT_CONF_THRESHOLD="${VGGT_CONF_THRESHOLD:-0.3}"
GS_ALPHA_THRESHOLD="${GS_ALPHA_THRESHOLD:-0.05}"
DETAIL_GRAD_THRESHOLD="${DETAIL_GRAD_THRESHOLD:-0.08}"
REPROJECTION_TOLERANCE="${REPROJECTION_TOLERANCE:-0.05}"

REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

for path in "${SCENE_ROOT}" "${PRIORS_DIR}" "${VGGT_ROOT}" "${GS_MODEL_PATH}" "${ORACLE_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[train-hrgs-kitchen] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[train-hrgs-kitchen] scene root         : ${SCENE_ROOT}"
echo "[train-hrgs-kitchen] asset root         : ${SCENE_ASSET_ROOT}"
echo "[train-hrgs-kitchen] priors dir         : ${PRIORS_DIR}"
echo "[train-hrgs-kitchen] oracle root        : ${ORACLE_ROOT}"
echo "[train-hrgs-kitchen] vggt root          : ${VGGT_ROOT}"
echo "[train-hrgs-kitchen] gs model path      : ${GS_MODEL_PATH}"
echo "[train-hrgs-kitchen] output dir         : ${OUTPUT_DIR}"
echo "[train-hrgs-kitchen] cache dir          : ${CACHE_DIR}"
echo "[train-hrgs-kitchen] num views          : ${NUM_VIEWS}"
echo "[train-hrgs-kitchen] samples per scene  : ${SAMPLES_PER_SCENE}"
echo "[train-hrgs-kitchen] max steps          : ${MAX_STEPS}"
echo "[train-hrgs-kitchen] lr                 : ${LR}"
echo "[train-hrgs-kitchen] mv loss weight     : ${LOSS_MV_W}"
if [[ -n "${REFINER_CHECKPOINT}" ]]; then
  echo "[train-hrgs-kitchen] resume checkpoint  : ${REFINER_CHECKPOINT}"
fi

CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/train_hrgsrefiner_v0.py"
  --scene_root "${SCENE_ROOT}"
  --gs_model_path "${GS_MODEL_PATH}"
  --oracle_root "${ORACLE_ROOT}"
  --priors_dir "${PRIORS_DIR}"
  --vggt_root "${VGGT_ROOT}"
  --source_images_subdir "${SOURCE_IMAGES_SUBDIR}"
  --target_images_subdir "${TARGET_IMAGES_SUBDIR}"
  --output_dir "${OUTPUT_DIR}"
  --num_views "${NUM_VIEWS}"
  --samples_per_scene "${SAMPLES_PER_SCENE}"
  --camera_resolution "${CAMERA_RESOLUTION}"
  --require_priors
  --cache_samples
  --cache_dir "${CACHE_DIR}"
  --max_steps "${MAX_STEPS}"
  --save_every "${SAVE_EVERY}"
  --eval_every "${EVAL_EVERY}"
  --log_every "${LOG_EVERY}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
  --seed "${SEED}"
  --loss_depth_w "${LOSS_DEPTH_W}"
  --loss_normal_w "${LOSS_NORMAL_W}"
  --loss_mv_w "${LOSS_MV_W}"
  --loss_delta_w "${LOSS_DELTA_W}"
  --loss_surface_w "${LOSS_SURFACE_W}"
  --loss_update_w "${LOSS_UPDATE_W}"
  --loss_detail_w "${LOSS_DETAIL_W}"
  --loss_conf_w "${LOSS_CONF_W}"
  --loss_prior_color_w "${LOSS_PRIOR_COLOR_W}"
  --vggt_conf_threshold "${VGGT_CONF_THRESHOLD}"
  --gs_alpha_threshold "${GS_ALPHA_THRESHOLD}"
  --detail_grad_threshold "${DETAIL_GRAD_THRESHOLD}"
  --reprojection_tolerance "${REPROJECTION_TOLERANCE}"
)

if [[ -n "${REFINER_CHECKPOINT}" ]]; then
  CMD+=(--refiner_checkpoint "${REFINER_CHECKPOINT}")
fi

"${CMD[@]}"

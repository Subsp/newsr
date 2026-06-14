#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
IMAGES_DIR="${IMAGES_DIR:-${SCENE_ROOT}/${IMAGES_SUBDIR}}"

ITERATIONS="${ITERATIONS:-30000}"
SPLATTING_CONFIG="${SPLATTING_CONFIG:-configs/hierarchical.json}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-${SCENE_NAME}_sof_vanilla_images2_v1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sof30k}"
MODEL_DIR="${MODEL_DIR:-${SCENE_ASSET_ROOT}/${EXPERIMENT_GROUP}/${EXPERIMENT_NAME}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DEVICE="${DATA_DEVICE:-cuda}"
EVAL="${EVAL:-1}"
RUN_RENDER_AFTER="${RUN_RENDER_AFTER:-0}"
RUN_METRICS_AFTER="${RUN_METRICS_AFTER:-0}"

TEST_ITERATIONS="${TEST_ITERATIONS:-${ITERATIONS}}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-${ITERATIONS}}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-${ITERATIONS}}"
START_CHECKPOINT="${START_CHECKPOINT:-}"

mkdir -p "${MODEL_DIR}"

for path in "${SCENE_ROOT}" "${IMAGES_DIR}" "${SCENE_ROOT}/sparse/0"; do
  if [[ ! -e "${path}" ]]; then
    echo "[sof-vanilla-images2] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[sof-vanilla-images2] scene             : ${SCENE_NAME}"
echo "[sof-vanilla-images2] scene root        : ${SCENE_ROOT}"
echo "[sof-vanilla-images2] asset root        : ${SCENE_ASSET_ROOT}"
echo "[sof-vanilla-images2] images subdir     : ${IMAGES_SUBDIR}"
echo "[sof-vanilla-images2] model dir         : ${MODEL_DIR}"
echo "[sof-vanilla-images2] splatting config  : ${SPLATTING_CONFIG}"
echo "[sof-vanilla-images2] iterations        : ${ITERATIONS}"
echo "[sof-vanilla-images2] test iterations   : ${TEST_ITERATIONS}"
echo "[sof-vanilla-images2] save iterations   : ${SAVE_ITERATIONS}"
echo "[sof-vanilla-images2] ckpt iterations   : ${CHECKPOINT_ITERATIONS}"
echo "[sof-vanilla-images2] data device       : ${DATA_DEVICE}"
if [[ -n "${START_CHECKPOINT}" ]]; then
  echo "[sof-vanilla-images2] start checkpoint  : ${START_CHECKPOINT}"
fi

TRAIN_CMD=(
  "${PYTHON_BIN}" train.py
  --splatting_config "${SPLATTING_CONFIG}"
  -s "${SCENE_ROOT}"
  -i "${IMAGES_SUBDIR}"
  -m "${MODEL_DIR}"
  --iterations "${ITERATIONS}"
  --test_iterations "${TEST_ITERATIONS}"
  --save_iterations "${SAVE_ITERATIONS}"
  --checkpoint_iterations "${CHECKPOINT_ITERATIONS}"
  --data_device "${DATA_DEVICE}"
)

if [[ "${EVAL}" == "1" ]]; then
  TRAIN_CMD+=(--eval)
fi

if [[ -n "${START_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--start_checkpoint "${START_CHECKPOINT}")
fi

echo
echo "[1/3] train vanilla SOF on ${IMAGES_SUBDIR}"
(
  cd "${SOF_ROOT}"
  "${TRAIN_CMD[@]}"
)

if [[ "${RUN_RENDER_AFTER}" == "1" ]]; then
  echo
  echo "[2/3] render ${IMAGES_SUBDIR}"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" render.py \
      -m "${MODEL_DIR}" \
      -s "${SCENE_ROOT}" \
      -i "${IMAGES_SUBDIR}" \
      --iteration "${ITERATIONS}" \
      --eval \
      --skip_train \
      --data_device cpu
  )
else
  echo
  echo "[2/3] skip render (RUN_RENDER_AFTER=${RUN_RENDER_AFTER})"
fi

if [[ "${RUN_METRICS_AFTER}" == "1" ]]; then
  echo
  echo "[3/3] metrics"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" metrics.py -m "${MODEL_DIR}"
  )
else
  echo
  echo "[3/3] skip metrics (RUN_METRICS_AFTER=${RUN_METRICS_AFTER})"
fi

echo
echo "[done] model dir : ${MODEL_DIR}"
echo "[done] checkpoint: ${MODEL_DIR}/chkpnt${ITERATIONS}.pth"
echo "[done] config    : ${MODEL_DIR}/config.json"

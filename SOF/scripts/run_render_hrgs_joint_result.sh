#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_eval}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
JOINT_MODEL_PATH="${JOINT_MODEL_PATH:-}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
ITERATION="${ITERATION:-550}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${JOINT_MODEL_PATH}" ]]; then
  if [[ -z "${CHECKPOINT_TAG}" ]]; then
    echo "[render-hrgs-joint] provide JOINT_MODEL_PATH or CHECKPOINT_TAG" >&2
    exit 1
  fi
  JOINT_MODEL_PATH="${OUTPUT_ROOT}/${SCENE_NAME}_joint_${CHECKPOINT_TAG}"
fi

for path in "${SCENE_ROOT}" "${JOINT_MODEL_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[render-hrgs-joint] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[render-hrgs-joint] scene root    : ${SCENE_ROOT}"
echo "[render-hrgs-joint] joint model    : ${JOINT_MODEL_PATH}"
echo "[render-hrgs-joint] images subdir  : ${IMAGES_SUBDIR}"
echo "[render-hrgs-joint] iteration      : ${ITERATION}"

(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" render.py \
    -m "${JOINT_MODEL_PATH}" \
    -s "${SCENE_ROOT}" \
    -i "${IMAGES_SUBDIR}" \
    --iteration "${ITERATION}" \
    --eval \
    --skip_train \
    --data_device cpu
)

echo
echo "[done] renders : ${JOINT_MODEL_PATH}/test/ours_${ITERATION}/renders"

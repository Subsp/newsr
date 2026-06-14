#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
MIP_INPUT_MODEL_PATH="${MIP_INPUT_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
PREPARE_INPUT_IF_MISSING="${PREPARE_INPUT_IF_MISSING:-1}"
SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SCENE_NAME}_mipstart_v0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${PREPARE_INPUT_IF_MISSING}" == "1" && ! -e "${MIP_INPUT_MODEL_PATH}/point_cloud" ]]; then
  echo "[train-hrgs-mipstart] preparing SOF-native mip input field ..."
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  MIP_MODEL_PATH="${MIP_MODEL_PATH}" \
  OUTPUT_MODEL_PATH="${MIP_INPUT_MODEL_PATH}" \
  MIP_ITERATION="${MIP_ITERATION}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SOF_ROOT}/scripts/run_prepare_hrgs_mip_input_scene.sh"
fi

if [[ ! -e "${MIP_INPUT_MODEL_PATH}" ]]; then
  echo "[train-hrgs-mipstart] adapted mip input field not found: ${MIP_INPUT_MODEL_PATH}" >&2
  exit 1
fi

echo "[train-hrgs-mipstart] scene root      : ${SCENE_ROOT}"
echo "[train-hrgs-mipstart] mip model path   : ${MIP_MODEL_PATH}"
echo "[train-hrgs-mipstart] input field path : ${MIP_INPUT_MODEL_PATH}"
echo "[train-hrgs-mipstart] source grid      : ${SOURCE_IMAGES_SUBDIR}"
echo "[train-hrgs-mipstart] target grid      : ${TARGET_IMAGES_SUBDIR}"
echo "[train-hrgs-mipstart] experiment name  : ${EXPERIMENT_NAME}"

SCENE_NAME="${SCENE_NAME}" \
SCENE_ROOT="${SCENE_ROOT}" \
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
GS_MODEL_PATH="${MIP_INPUT_MODEL_PATH}" \
SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR}" \
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${SOF_ROOT}/scripts/run_train_hrgsrefiner_kitchen.sh"

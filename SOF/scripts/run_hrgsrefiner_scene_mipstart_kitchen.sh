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

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
MIP_INPUT_MODEL_PATH="${MIP_INPUT_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
PREPARE_INPUT_IF_MISSING="${PREPARE_INPUT_IF_MISSING:-1}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${SOF_ROOT}/output/hrgs_train_formal/${SCENE_NAME}_mipstart_v0}"
REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-}"
if [[ -z "${REFINER_CHECKPOINT}" ]]; then
  REF_LIST="$(find "${TRAIN_OUTPUT_DIR}/checkpoints" -maxdepth 1 -name 'hrgsrefiner_step_*.pt' | sort)"
  if [[ -z "${REF_LIST}" ]]; then
    echo "[hrgs-scene-mipstart] no refiner checkpoints found under ${TRAIN_OUTPUT_DIR}/checkpoints" >&2
    exit 1
  fi
  REFINER_CHECKPOINT="$(printf '%s\n' "${REF_LIST}" | tail -n 1)"
fi

CHECKPOINT_TAG="${CHECKPOINT_TAG:-$(basename "${REFINER_CHECKPOINT}" .pt)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_eval}"
RUNNER_OUT="${RUNNER_OUT:-${OUTPUT_ROOT}/${SCENE_NAME}_mipstart_${CHECKPOINT_TAG}}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
MAX_VIEWS="${MAX_VIEWS:-2}"
VISIBILITY_DOWNSAMPLE="${VISIBILITY_DOWNSAMPLE:-8}"
VISIBILITY_TOPK="${VISIBILITY_TOPK:-4}"
VISIBILITY_MAX_VISIBLE="${VISIBILITY_MAX_VISIBLE:-30000}"
VISIBILITY_MAX_PATCH_RADIUS="${VISIBILITY_MAX_PATCH_RADIUS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONUNBUFFERED=1

for path in "${SCENE_ROOT}" "${PRIORS_DIR}" "${VGGT_ROOT}" "${MIP_MODEL_PATH}" "${REFINER_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[hrgs-scene-mipstart] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ "${PREPARE_INPUT_IF_MISSING}" == "1" && ! -e "${MIP_INPUT_MODEL_PATH}/point_cloud" ]]; then
  echo "[hrgs-scene-mipstart] preparing SOF-native mip input field ..."
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
  echo "[hrgs-scene-mipstart] adapted mip input field not found: ${MIP_INPUT_MODEL_PATH}" >&2
  exit 1
fi

echo "[hrgs-scene-mipstart] scene root         : ${SCENE_ROOT}"
echo "[hrgs-scene-mipstart] mip model path      : ${MIP_MODEL_PATH}"
echo "[hrgs-scene-mipstart] input field path    : ${MIP_INPUT_MODEL_PATH}"
echo "[hrgs-scene-mipstart] refiner checkpoint  : ${REFINER_CHECKPOINT}"
echo "[hrgs-scene-mipstart] runner out          : ${RUNNER_OUT}"
echo "[hrgs-scene-mipstart] source grid         : ${SOURCE_IMAGES_SUBDIR}"
echo "[hrgs-scene-mipstart] target grid         : ${TARGET_IMAGES_SUBDIR}"
echo "[hrgs-scene-mipstart] visibility ds       : ${VISIBILITY_DOWNSAMPLE}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/run_hrgsrefiner_scene.py" \
  --scene_root "${SCENE_ROOT}" \
  --gs_model_path "${MIP_INPUT_MODEL_PATH}" \
  --output_dir "${RUNNER_OUT}" \
  --refiner_checkpoint "${REFINER_CHECKPOINT}" \
  --vggt_root "${VGGT_ROOT}" \
  --source_images_subdir "${SOURCE_IMAGES_SUBDIR}" \
  --target_images_subdir "${TARGET_IMAGES_SUBDIR}" \
  --priors_dir "${PRIORS_DIR}" \
  --require_priors \
  --max_views "${MAX_VIEWS}" \
  --visibility_downsample "${VISIBILITY_DOWNSAMPLE}" \
  --visibility_topk "${VISIBILITY_TOPK}" \
  --visibility_max_visible "${VISIBILITY_MAX_VISIBLE}" \
  --visibility_max_patch_radius "${VISIBILITY_MAX_PATCH_RADIUS}"

echo
echo "[done] runner out : ${RUNNER_OUT}"

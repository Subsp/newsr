#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k_rerun_v0}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
MIP_ITERATION="${MIP_ITERATION:-30000}"

STATE_RUN_NAME="${STATE_RUN_NAME:-mip30k_rerun_gs2mesh_surface_state_v0}"
STATE_DIR="${STATE_DIR:-${SOF_ROOT}/output/gaussian_surface_state_v0/${SCENE_NAME}/${STATE_RUN_NAME}}"
SURFACE_STATE_PAYLOAD="${SURFACE_STATE_PAYLOAD:-${STATE_DIR}/gaussian_surface_state_v0.pt}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-12}"
SURFACE_STATE_GROUPS="${SURFACE_STATE_GROUPS:-surface_carrier,near_surface_uncertain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STATE_DIR}/renders_by_class_${IMAGES_SUBDIR}_${SPLIT}_max${MAX_VIEWS}}"

if [[ ! -d "${MIP_MODEL_PATH}" ]]; then
  echo "[surface-state-render-v0] missing MIP_MODEL_PATH=${MIP_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SURFACE_STATE_PAYLOAD}" ]]; then
  echo "[surface-state-render-v0] missing SURFACE_STATE_PAYLOAD=${SURFACE_STATE_PAYLOAD}" >&2
  echo "[surface-state-render-v0] run scripts/run_classify_mip_surface_state_v0_kitchen.sh first." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${WHITE_BACKGROUND:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--white_background)
fi
if [[ "${SAVE_ALPHA:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--save_alpha)
fi
if [[ "${SAVE_PREMUL:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--save_premul)
fi
if [[ "${SKIP_EMPTY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_empty)
fi

echo "[surface-state-render-v0] scene      : ${SCENE_ROOT}"
echo "[surface-state-render-v0] model      : ${MIP_MODEL_PATH} iter=${MIP_ITERATION}"
echo "[surface-state-render-v0] payload    : ${SURFACE_STATE_PAYLOAD}"
echo "[surface-state-render-v0] groups     : ${SURFACE_STATE_GROUPS}"
echo "[surface-state-render-v0] views      : ${IMAGES_SUBDIR} split=${SPLIT} max=${MAX_VIEWS}"
echo "[surface-state-render-v0] output     : ${OUTPUT_ROOT}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/render_surface_state_class_groups_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --model_path "${MIP_MODEL_PATH}" \
  --iteration "${MIP_ITERATION}" \
  --surface_state_payload "${SURFACE_STATE_PAYLOAD}" \
  --output_root "${OUTPUT_ROOT}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --split "${SPLIT}" \
  --max_views "${MAX_VIEWS}" \
  --groups "${SURFACE_STATE_GROUPS}" \
  "${EXTRA_ARGS[@]}"

echo "[done] rendered groups: ${OUTPUT_ROOT}"

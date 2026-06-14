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
BEFORE_MODEL_PATH="${BEFORE_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
BEFORE_ITERATION="${BEFORE_ITERATION:-30000}"

AFTER_MODEL_PATH="${AFTER_MODEL_PATH:-}"
AFTER_ITERATION="${AFTER_ITERATION:-32000}"
AFTER_LABEL="${AFTER_LABEL:-after_injection}"
BEFORE_LABEL="${BEFORE_LABEL:-before_injection}"

SURFACE_STATE_PROFILE="${SURFACE_STATE_PROFILE:-conservative_v0}"
STATE_RUN_NAME_DEFAULT="mip30k_rerun_gs2mesh_surface_state_v0"
if [[ "${SURFACE_STATE_PROFILE}" != "conservative_v0" ]]; then
  STATE_RUN_NAME_DEFAULT="${STATE_RUN_NAME_DEFAULT}_${SURFACE_STATE_PROFILE}"
fi
STATE_RUN_NAME="${STATE_RUN_NAME:-${STATE_RUN_NAME_DEFAULT}}"
STATE_DIR="${STATE_DIR:-${SOF_ROOT}/output/gaussian_surface_state_v0/${SCENE_NAME}/${STATE_RUN_NAME}}"
SURFACE_STATE_PAYLOAD="${SURFACE_STATE_PAYLOAD:-${STATE_DIR}/gaussian_surface_state_v0.pt}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-0}"
SURFACE_RENDER_GROUPS="${SURFACE_RENDER_GROUPS:-surface_candidate}"

COMPARE_TAG_DEFAULT="${MIP_EXPERIMENT_NAME}_${SURFACE_RENDER_GROUPS}_${SPLIT}_before_after_v0"
COMPARE_TAG="${COMPARE_TAG:-${COMPARE_TAG_DEFAULT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/surface_layer_before_after_v0/${SCENE_NAME}/${COMPARE_TAG}}"
PAIR_GAP="${PAIR_GAP:-8}"

if [[ -z "${AFTER_MODEL_PATH}" ]]; then
  echo "[surface-layer-compare-v0] AFTER_MODEL_PATH is required." >&2
  exit 1
fi
if [[ ! -d "${BEFORE_MODEL_PATH}" ]]; then
  echo "[surface-layer-compare-v0] missing BEFORE_MODEL_PATH=${BEFORE_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${AFTER_MODEL_PATH}" ]]; then
  echo "[surface-layer-compare-v0] missing AFTER_MODEL_PATH=${AFTER_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SURFACE_STATE_PAYLOAD}" ]]; then
  echo "[surface-layer-compare-v0] missing SURFACE_STATE_PAYLOAD=${SURFACE_STATE_PAYLOAD}" >&2
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

BEFORE_ROOT="${OUTPUT_ROOT}/${BEFORE_LABEL}"
AFTER_ROOT="${OUTPUT_ROOT}/${AFTER_LABEL}"
PAIR_ROOT="${OUTPUT_ROOT}/pairs"

echo "[surface-layer-compare-v0] scene         : ${SCENE_ROOT}"
echo "[surface-layer-compare-v0] before model  : ${BEFORE_MODEL_PATH} iter=${BEFORE_ITERATION}"
echo "[surface-layer-compare-v0] after model   : ${AFTER_MODEL_PATH} iter=${AFTER_ITERATION}"
echo "[surface-layer-compare-v0] payload       : ${SURFACE_STATE_PAYLOAD}"
echo "[surface-layer-compare-v0] groups        : ${SURFACE_RENDER_GROUPS}"
echo "[surface-layer-compare-v0] view setup    : ${IMAGES_SUBDIR} split=${SPLIT} max=${MAX_VIEWS}"
echo "[surface-layer-compare-v0] output root   : ${OUTPUT_ROOT}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/render_surface_state_class_groups_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --model_path "${BEFORE_MODEL_PATH}" \
  --iteration "${BEFORE_ITERATION}" \
  --surface_state_payload "${SURFACE_STATE_PAYLOAD}" \
  --output_root "${BEFORE_ROOT}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --split "${SPLIT}" \
  --max_views "${MAX_VIEWS}" \
  --groups "${SURFACE_RENDER_GROUPS}" \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/render_surface_state_class_groups_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --model_path "${AFTER_MODEL_PATH}" \
  --iteration "${AFTER_ITERATION}" \
  --surface_state_payload "${SURFACE_STATE_PAYLOAD}" \
  --output_root "${AFTER_ROOT}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --split "${SPLIT}" \
  --max_views "${MAX_VIEWS}" \
  --groups "${SURFACE_RENDER_GROUPS}" \
  "${EXTRA_ARGS[@]}"

IFS=',' read -r -a GROUP_ARRAY <<< "${SURFACE_RENDER_GROUPS}"
for raw_group in "${GROUP_ARRAY[@]}"; do
  group="$(printf '%s' "${raw_group}" | xargs)"
  if [[ -z "${group}" ]]; then
    continue
  fi
  before_render_dir="${BEFORE_ROOT}/${group}/${SPLIT}/ours_${BEFORE_ITERATION}/renders"
  after_render_dir="${AFTER_ROOT}/${group}/${SPLIT}/ours_${AFTER_ITERATION}/renders"
  if [[ -d "${before_render_dir}" && -d "${after_render_dir}" ]]; then
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/compose_side_by_side_pairs_v0.py" \
      --left_dir "${before_render_dir}" \
      --right_dir "${after_render_dir}" \
      --output_dir "${PAIR_ROOT}/${group}/renders" \
      --left_label "${BEFORE_LABEL}" \
      --right_label "${AFTER_LABEL}" \
      --gap "${PAIR_GAP}" \
      --add_labels
  fi
  if [[ "${SAVE_ALPHA:-1}" == "1" ]]; then
    before_alpha_dir="${BEFORE_ROOT}/${group}/${SPLIT}/ours_${BEFORE_ITERATION}/alpha"
    after_alpha_dir="${AFTER_ROOT}/${group}/${SPLIT}/ours_${AFTER_ITERATION}/alpha"
    if [[ -d "${before_alpha_dir}" && -d "${after_alpha_dir}" ]]; then
      "${PYTHON_BIN}" -u "${SCRIPT_DIR}/compose_side_by_side_pairs_v0.py" \
        --left_dir "${before_alpha_dir}" \
        --right_dir "${after_alpha_dir}" \
        --output_dir "${PAIR_ROOT}/${group}/alpha" \
        --left_label "${BEFORE_LABEL}" \
        --right_label "${AFTER_LABEL}" \
        --gap "${PAIR_GAP}" \
        --add_labels
    fi
  fi
  if [[ "${SAVE_PREMUL:-1}" == "1" ]]; then
    before_premul_dir="${BEFORE_ROOT}/${group}/${SPLIT}/ours_${BEFORE_ITERATION}/premul"
    after_premul_dir="${AFTER_ROOT}/${group}/${SPLIT}/ours_${AFTER_ITERATION}/premul"
    if [[ -d "${before_premul_dir}" && -d "${after_premul_dir}" ]]; then
      "${PYTHON_BIN}" -u "${SCRIPT_DIR}/compose_side_by_side_pairs_v0.py" \
        --left_dir "${before_premul_dir}" \
        --right_dir "${after_premul_dir}" \
        --output_dir "${PAIR_ROOT}/${group}/premul" \
        --left_label "${BEFORE_LABEL}" \
        --right_label "${AFTER_LABEL}" \
        --gap "${PAIR_GAP}" \
        --add_labels
    fi
  fi
done

echo "[done] before renders : ${BEFORE_ROOT}"
echo "[done] after renders  : ${AFTER_ROOT}"
echo "[done] pair renders   : ${PAIR_ROOT}"

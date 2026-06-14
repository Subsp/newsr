#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
BASE_TAG="${BASE_TAG:-k22}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input_init_repair_v0}"
MIP_ITERATION="${MIP_ITERATION:-30000}"

REG_RUN_NAME="${REG_RUN_NAME:-${BASE_TAG}_reg}"
REG_RUN_ROOT="${REG_RUN_ROOT:-${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${REG_RUN_NAME}}"
REG_MODEL_PATH="${REG_MODEL_PATH:-${REG_RUN_ROOT}/pulled_mip_model}"
REG_ITERATION="${REG_ITERATION:-32000}"
USE_REG="${USE_REG:-1}"

REC_RUN_NAME="${REC_RUN_NAME:-${BASE_TAG}}"
REC_RUN_ROOT="${REC_RUN_ROOT:-${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${REC_RUN_NAME}}"
REC_MODEL_NAME="${REC_MODEL_NAME:-recovered_mip_model_lr_miphr_v1}"
REC_MODEL_PATH="${REC_MODEL_PATH:-${REC_RUN_ROOT}/${REC_MODEL_NAME}}"
REC_ITERATION="${REC_ITERATION:-31600}"
USE_REC="${USE_REC:-1}"

RUN_NAME="${RUN_NAME:-closure_${BASE_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/diagnose_mip_closure_v0/${SCENE_NAME}/${RUN_NAME}}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-16}"
VIEW_INDICES="${VIEW_INDICES:-}"
LOWPASS_KERNEL="${LOWPASS_KERNEL:-25}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.05}"
DEPTH_RELATIVE_MIN="${DEPTH_RELATIVE_MIN:-0.5}"
NUM_DEBUG_VIEWS="${NUM_DEBUG_VIEWS:-4}"
SAVE_VIEW_NPZ="${SAVE_VIEW_NPZ:-0}"
COMPARE_WITH_OWN_SPLAT_SETTINGS="${COMPARE_WITH_OWN_SPLAT_SETTINGS:-0}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-srtest}"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_model_path() {
  local candidate_root="$1"
  local iteration="$2"
  if [[ -z "${candidate_root}" ]]; then
    return 1
  fi
  if [[ -e "${candidate_root}/point_cloud/iteration_${iteration}/point_cloud.ply" ]]; then
    printf '%s\n' "${candidate_root}"
    return 0
  fi
  if [[ -d "${candidate_root}" ]]; then
    local child
    for child in "${candidate_root}"/*; do
      if [[ -d "${child}" && -e "${child}/point_cloud/iteration_${iteration}/point_cloud.ply" ]]; then
        printf '%s\n' "${child}"
        return 0
      fi
    done
  fi
  return 1
}

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

if RESOLVED_MIP_MODEL_PATH="$(resolve_model_path "${MIP_MODEL_PATH}" "${MIP_ITERATION}")"; then
  MIP_MODEL_PATH="${RESOLVED_MIP_MODEL_PATH}"
fi
if [[ "${USE_REG}" == "1" && -n "${REG_MODEL_PATH}" ]]; then
  if RESOLVED_REG_MODEL_PATH="$(resolve_model_path "${REG_MODEL_PATH}" "${REG_ITERATION}")"; then
    REG_MODEL_PATH="${RESOLVED_REG_MODEL_PATH}"
  elif RESOLVED_REG_MODEL_PATH="$(resolve_model_path "${REG_RUN_ROOT}" "${REG_ITERATION}")"; then
    REG_MODEL_PATH="${RESOLVED_REG_MODEL_PATH}"
  fi
fi
if [[ "${USE_REC}" == "1" && -n "${REC_MODEL_PATH}" ]]; then
  if RESOLVED_REC_MODEL_PATH="$(resolve_model_path "${REC_MODEL_PATH}" "${REC_ITERATION}")"; then
    REC_MODEL_PATH="${RESOLVED_REC_MODEL_PATH}"
  elif RESOLVED_REC_MODEL_PATH="$(resolve_model_path "${REC_RUN_ROOT}" "${REC_ITERATION}")"; then
    REC_MODEL_PATH="${RESOLVED_REC_MODEL_PATH}"
  fi
fi

if [[ ! -e "${MIP_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" ]]; then
  echo "[diagnose-mip-closure-v0] missing mip model: ${MIP_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" >&2
  exit 1
fi

ARGS=(
  --scene_root "${SCENE_ROOT}"
  --mip_model_path "${MIP_MODEL_PATH}"
  --output_root "${OUTPUT_ROOT}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
  --view_indices "${VIEW_INDICES}"
  --mip_iteration "${MIP_ITERATION}"
  --lowpass_kernel "${LOWPASS_KERNEL}"
  --alpha_threshold "${ALPHA_THRESHOLD}"
  --depth_relative_min "${DEPTH_RELATIVE_MIN}"
  --num_debug_views "${NUM_DEBUG_VIEWS}"
)

if [[ "${USE_REG}" == "1" && -n "${REG_MODEL_PATH}" ]]; then
  if [[ ! -e "${REG_MODEL_PATH}/point_cloud/iteration_${REG_ITERATION}/point_cloud.ply" ]]; then
    echo "[diagnose-mip-closure-v0] missing reg model: ${REG_MODEL_PATH}/point_cloud/iteration_${REG_ITERATION}/point_cloud.ply" >&2
    exit 1
  fi
  ARGS+=(--reg_model_path "${REG_MODEL_PATH}" --reg_iteration "${REG_ITERATION}")
fi

if [[ "${USE_REC}" == "1" && -n "${REC_MODEL_PATH}" ]]; then
  if [[ ! -e "${REC_MODEL_PATH}/point_cloud/iteration_${REC_ITERATION}/point_cloud.ply" ]]; then
    echo "[diagnose-mip-closure-v0] missing rec model: ${REC_MODEL_PATH}/point_cloud/iteration_${REC_ITERATION}/point_cloud.ply" >&2
    exit 1
  fi
  ARGS+=(--rec_model_path "${REC_MODEL_PATH}" --rec_iteration "${REC_ITERATION}")
fi

if [[ "${USE_REG}" != "1" && "${USE_REC}" != "1" ]]; then
  echo "[diagnose-mip-closure-v0] both USE_REG and USE_REC are disabled; nothing to compare." >&2
  exit 1
fi

if [[ "${SAVE_VIEW_NPZ}" == "1" ]]; then
  ARGS+=(--save_view_npz)
fi
if [[ "${COMPARE_WITH_OWN_SPLAT_SETTINGS}" == "1" ]]; then
  ARGS+=(--compare_with_own_splat_settings)
fi
if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
  ARGS+=(--white_background)
fi

echo "[diagnose-mip-closure-v0] scene root  : ${SCENE_ROOT}"
echo "[diagnose-mip-closure-v0] mip model   : ${MIP_MODEL_PATH} iter=${MIP_ITERATION}"
if [[ "${USE_REG}" == "1" && -n "${REG_MODEL_PATH}" ]]; then
  echo "[diagnose-mip-closure-v0] reg model   : ${REG_MODEL_PATH} iter=${REG_ITERATION}"
else
  echo "[diagnose-mip-closure-v0] reg model   : disabled"
fi
if [[ "${USE_REC}" == "1" && -n "${REC_MODEL_PATH}" ]]; then
  echo "[diagnose-mip-closure-v0] rec model   : ${REC_MODEL_PATH} iter=${REC_ITERATION}"
else
  echo "[diagnose-mip-closure-v0] rec model   : disabled"
fi
echo "[diagnose-mip-closure-v0] output root : ${OUTPUT_ROOT}"
echo "[diagnose-mip-closure-v0] views       : split=${SPLIT} max=${MAX_VIEWS} indices=${VIEW_INDICES:-<uniform>}"
echo "[diagnose-mip-closure-v0] lowpass     : kernel=${LOWPASS_KERNEL} alpha>${ALPHA_THRESHOLD} depth_rel_min=${DEPTH_RELATIVE_MIN}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/diagnose_mip_closure_v0.py" "${ARGS[@]}"

echo "[done] summary      : ${OUTPUT_ROOT}/closure_summary.json"
echo "[done] debug views  : ${OUTPUT_ROOT}/debug_views"
if [[ "${SAVE_VIEW_NPZ}" == "1" ]]; then
  echo "[done] npz payloads : ${OUTPUT_ROOT}"
fi

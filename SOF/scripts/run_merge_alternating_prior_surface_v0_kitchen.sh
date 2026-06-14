#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-srtest}"

BASE_MODEL_NAME="${BASE_MODEL_NAME:-soflr30k}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images8_v1/${BASE_MODEL_NAME}}"
BASE_ITERATION="${BASE_ITERATION:--1}"

ALTERNATING_RUN_NAME="${ALTERNATING_RUN_NAME:-alternating_prior_surface_mv16_anchor_v1}"
ALTERNATING_MODEL_PATH="${ALTERNATING_MODEL_PATH:-${SOF_ROOT}/output/alternating_prior_surface_v0/${SCENE_NAME}/${ALTERNATING_RUN_NAME}}"
ALTERNATING_ITERATION="${ALTERNATING_ITERATION:--1}"

RUN_NAME="${RUN_NAME:-${ALTERNATING_RUN_NAME}_merged_${BASE_MODEL_NAME}}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/alternating_prior_surface_merge_v0/${SCENE_NAME}/${RUN_NAME}}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-0}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
RENDER_SPLIT="${RENDER_SPLIT:-test}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
DATA_DEVICE="${DATA_DEVICE:-cpu}"

resolve_point_cloud_ply() {
  local model_path="$1"
  local requested_iter="$2"
  local point_root="${model_path}/point_cloud"
  if [[ ! -d "${point_root}" ]]; then
    return 1
  fi
  if [[ "${requested_iter}" != "-1" ]]; then
    local ply="${point_root}/iteration_${requested_iter}/point_cloud.ply"
    if [[ -f "${ply}" ]]; then
      printf '%s|%s\n' "${requested_iter}" "${ply}"
      return 0
    fi
    return 1
  fi

  local best_iter=""
  local best_ply=""
  local dir
  for dir in "${point_root}"/iteration_*; do
    [[ -d "${dir}" ]] || continue
    local name="${dir##*/iteration_}"
    [[ "${name}" =~ ^[0-9]+$ ]] || continue
    local ply="${dir}/point_cloud.ply"
    [[ -f "${ply}" ]] || continue
    if [[ -z "${best_iter}" || "${name}" -gt "${best_iter}" ]]; then
      best_iter="${name}"
      best_ply="${ply}"
    fi
  done
  [[ -n "${best_ply}" ]] || return 1
  printf '%s|%s\n' "${best_iter}" "${best_ply}"
}

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[merge-alternating-prior-surface-v0] missing scene root: ${SCENE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
  echo "[merge-alternating-prior-surface-v0] missing base model path: ${BASE_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${ALTERNATING_MODEL_PATH}" ]]; then
  echo "[merge-alternating-prior-surface-v0] missing alternating model path: ${ALTERNATING_MODEL_PATH}" >&2
  exit 1
fi

BASE_INFO="$(resolve_point_cloud_ply "${BASE_MODEL_PATH}" "${BASE_ITERATION}")" || {
  echo "[merge-alternating-prior-surface-v0] failed to resolve base model ply under: ${BASE_MODEL_PATH}" >&2
  exit 1
}
ALT_INFO="$(resolve_point_cloud_ply "${ALTERNATING_MODEL_PATH}" "${ALTERNATING_ITERATION}")" || {
  echo "[merge-alternating-prior-surface-v0] failed to resolve alternating model ply under: ${ALTERNATING_MODEL_PATH}" >&2
  exit 1
}

BASE_RESOLVED_ITER="${BASE_INFO%%|*}"
BASE_PLY="${BASE_INFO#*|}"
ALT_RESOLVED_ITER="${ALT_INFO%%|*}"
ALT_PLY="${ALT_INFO#*|}"

mkdir -p "${OUTPUT_MODEL_PATH}"

echo "[merge-alternating-prior-surface-v0] scene root         : ${SCENE_ROOT}"
echo "[merge-alternating-prior-surface-v0] base model path    : ${BASE_MODEL_PATH}"
echo "[merge-alternating-prior-surface-v0] base ply           : ${BASE_PLY}"
echo "[merge-alternating-prior-surface-v0] alternating path   : ${ALTERNATING_MODEL_PATH}"
echo "[merge-alternating-prior-surface-v0] alternating ply    : ${ALT_PLY}"
echo "[merge-alternating-prior-surface-v0] output model path  : ${OUTPUT_MODEL_PATH}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/merge_gaussian_plys_v0.py" \
  --base_ply "${BASE_PLY}" \
  --extra_ply "${ALT_PLY}" \
  --output_model_path "${OUTPUT_MODEL_PATH}" \
  --output_iteration "${OUTPUT_ITERATION}" \
  --copy_config_from "${BASE_MODEL_PATH}"

if [[ "${RUN_RENDER}" == "1" ]]; then
  RENDER_CMD=(
    "${PYTHON_BIN}" -u "${SOF_ROOT}/render.py"
    -m "${OUTPUT_MODEL_PATH}"
    -s "${SCENE_ROOT}"
    -i "${IMAGES_SUBDIR}"
    --iteration "${OUTPUT_ITERATION}"
    --eval
    --data_device "${DATA_DEVICE}"
    --skip_train
  )
  if [[ "${RENDER_SPLIT}" == "train" ]]; then
    RENDER_CMD=(
      "${PYTHON_BIN}" -u "${SOF_ROOT}/render.py"
      -m "${OUTPUT_MODEL_PATH}"
      -s "${SCENE_ROOT}"
      -i "${IMAGES_SUBDIR}"
      --iteration "${OUTPUT_ITERATION}"
      --eval
      --data_device "${DATA_DEVICE}"
      --skip_test
    )
  elif [[ "${RENDER_SPLIT}" != "both" ]]; then
    echo "[merge-alternating-prior-surface-v0] unsupported RENDER_SPLIT=${RENDER_SPLIT}, expected train/test/both" >&2
    exit 1
  fi
  "${RENDER_CMD[@]}"
fi

if [[ "${RUN_METRICS}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/metrics.py" -m "${OUTPUT_MODEL_PATH}"
fi

echo "[done] base iter      : ${BASE_RESOLVED_ITER}"
echo "[done] alternating iter: ${ALT_RESOLVED_ITER}"
echo "[done] merged model   : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
if [[ "${RUN_RENDER}" == "1" ]]; then
  if [[ "${RENDER_SPLIT}" == "train" ]]; then
    echo "[done] renders       : ${OUTPUT_MODEL_PATH}/train/ours_${OUTPUT_ITERATION}/renders"
  elif [[ "${RENDER_SPLIT}" == "test" ]]; then
    echo "[done] renders       : ${OUTPUT_MODEL_PATH}/test/ours_${OUTPUT_ITERATION}/renders"
  else
    echo "[done] renders train : ${OUTPUT_MODEL_PATH}/train/ours_${OUTPUT_ITERATION}/renders"
    echo "[done] renders test  : ${OUTPUT_MODEL_PATH}/test/ours_${OUTPUT_ITERATION}/renders"
  fi
fi
if [[ "${RUN_METRICS}" == "1" ]]; then
  echo "[done] metrics       : ${OUTPUT_MODEL_PATH}/results_full.json"
fi

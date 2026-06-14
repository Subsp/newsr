#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-${STAGE_NAME}_geometry_only_v0}"
MESH_BOUNDED_RUN_NAME="${MESH_BOUNDED_RUN_NAME:-${SOURCE_RUN_NAME}_mesh_bounded_v0}"
INPUT_MODEL_PATH="${INPUT_MODEL_PATH:-${SOF_ROOT}/output/mesh_bounded_gaussians_v0/${SCENE_NAME}/${MESH_BOUNDED_RUN_NAME}}"
ITERATION="${ITERATION:-34000}"
PAYLOAD_PATH="${PAYLOAD_PATH:-${INPUT_MODEL_PATH}/mesh_bounded_gaussians_v0.pt}"

OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${MESH_BOUNDED_RUN_NAME}_strict_gate_v0}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/mesh_bounded_gaussians_v0/${SCENE_NAME}/${OUTPUT_RUN_NAME}}"

MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.65}"
MIN_MIP_SUPPORT="${MIN_MIP_SUPPORT:-0.80}"
MAX_D_NORM="${MAX_D_NORM:-0.85}"
MAX_SIGMA_NORMAL_NORM="${MAX_SIGMA_NORMAL_NORM:-0.95}"
MAX_PROXY_COUNT="${MAX_PROXY_COUNT:-35000}"
OPACITY_SCALE="${OPACITY_SCALE:-0.20}"
ALPHA_MAX="${ALPHA_MAX:-0.006}"

if [[ ! -f "${INPUT_MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
  echo "[filter-mesh-bounded-v0] missing input ply: ${INPUT_MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" >&2
  exit 1
fi
if [[ ! -f "${PAYLOAD_PATH}" ]]; then
  echo "[filter-mesh-bounded-v0] missing payload: ${PAYLOAD_PATH}" >&2
  exit 1
fi

echo "[filter-mesh-bounded-v0] input : ${INPUT_MODEL_PATH} iter=${ITERATION}"
echo "[filter-mesh-bounded-v0] output: ${OUTPUT_MODEL_PATH}"
echo "[filter-mesh-bounded-v0] thresholds: conf>=${MIN_CONFIDENCE} mip>=${MIN_MIP_SUPPORT} d<=${MAX_D_NORM} sigma<=${MAX_SIGMA_NORMAL_NORM} max=${MAX_PROXY_COUNT}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/filter_mesh_bounded_gaussians_v0.py" \
  --input_model_path "${INPUT_MODEL_PATH}" \
  --output_model_path "${OUTPUT_MODEL_PATH}" \
  --iteration "${ITERATION}" \
  --payload_path "${PAYLOAD_PATH}" \
  --min_confidence "${MIN_CONFIDENCE}" \
  --min_mip_support "${MIN_MIP_SUPPORT}" \
  --max_d_norm "${MAX_D_NORM}" \
  --max_sigma_normal_norm "${MAX_SIGMA_NORMAL_NORM}" \
  --max_proxy_count "${MAX_PROXY_COUNT}" \
  --opacity_scale "${OPACITY_SCALE}" \
  --alpha_max "${ALPHA_MAX}"

echo "[done] filtered model  : ${OUTPUT_MODEL_PATH}"
echo "[done] filtered payload: ${OUTPUT_MODEL_PATH}/mesh_bounded_gaussians_v0_filtered.pt"
echo "[done] summary         : ${OUTPUT_MODEL_PATH}/filter_mesh_bounded_gaussians_v0_summary.json"

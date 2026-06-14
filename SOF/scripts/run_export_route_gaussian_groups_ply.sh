#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_eval}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
DETAIL_MODEL_PATH="${DETAIL_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input}"

CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
ROUTE_PAYLOAD="${ROUTE_PAYLOAD:-}"
EXPORT_OUTPUT_DIR="${EXPORT_OUTPUT_DIR:-}"
ITERATION="${ITERATION:-30000}"
ROUTE_GROUPS="${ROUTE_GROUPS:-detail,suppress}"
GROUP_MODE="${GROUP_MODE:-threshold}"
THRESHOLD="${THRESHOLD:-0.2}"
SH_DEGREE="${SH_DEGREE:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${ROUTE_PAYLOAD}" ]]; then
  if [[ -z "${CHECKPOINT_TAG}" ]]; then
    echo "[export-route-ply] provide ROUTE_PAYLOAD or CHECKPOINT_TAG" >&2
    exit 1
  fi
  ROUTE_PAYLOAD="${OUTPUT_ROOT}/${SCENE_NAME}_${CHECKPOINT_TAG}/route_payload_v0.pt"
fi

if [[ -z "${EXPORT_OUTPUT_DIR}" ]]; then
  if [[ -z "${CHECKPOINT_TAG}" ]]; then
    CHECKPOINT_TAG="manual"
  fi
  EXPORT_OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE_NAME}_route_plys_${CHECKPOINT_TAG}"
fi

for path in "${DETAIL_MODEL_PATH}" "${ROUTE_PAYLOAD}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[export-route-ply] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[export-route-ply] detail model     : ${DETAIL_MODEL_PATH}"
echo "[export-route-ply] route payload    : ${ROUTE_PAYLOAD}"
echo "[export-route-ply] output dir       : ${EXPORT_OUTPUT_DIR}"
echo "[export-route-ply] groups           : ${ROUTE_GROUPS}"
echo "[export-route-ply] group mode       : ${GROUP_MODE}"
echo "[export-route-ply] threshold        : ${THRESHOLD}"
echo "[export-route-ply] iteration        : ${ITERATION}"

CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/export_route_gaussian_groups_ply.py"
  --detail_model_path "${DETAIL_MODEL_PATH}"
  --route_payload "${ROUTE_PAYLOAD}"
  --output_dir "${EXPORT_OUTPUT_DIR}"
  --iteration "${ITERATION}"
  --groups "${ROUTE_GROUPS}"
  --group_mode "${GROUP_MODE}"
  --threshold "${THRESHOLD}"
  --sh_degree "${SH_DEGREE}"
)

"${CMD[@]}"

echo
echo "[done] route group ply exports: ${EXPORT_OUTPUT_DIR}"

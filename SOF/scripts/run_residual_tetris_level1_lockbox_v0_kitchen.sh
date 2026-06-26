#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"

STATIC_NAME="${STATIC_NAME:-${BASE_EXPERIMENT_NAME}_residual_tetris_oracle_v0_static_v1}"
STATIC_V1_DIR="${STATIC_V1_DIR:-${SOF_ROOT}/output/residual_tetris_static_v1/${STATIC_NAME}}"
ORACLE_NAME="${ORACLE_NAME:-${BASE_EXPERIMENT_NAME}_residual_tetris_oracle_v0}"
ORACLE_DIR="${ORACLE_DIR:-${SOF_ROOT}/output/residual_tetris_oracle_v0/${ORACLE_NAME}}"

LOCKBOX_NAME="${LOCKBOX_NAME:-level1_lockbox_v0}"
LOCKBOX_PRIMITIVE_DIR="${LOCKBOX_PRIMITIVE_DIR:-}"
LOCKBOX_BASE_RENDER_DIR="${LOCKBOX_BASE_RENDER_DIR:-}"
LOCKBOX_SR_DIR="${LOCKBOX_SR_DIR:-}"
LOCKBOX_WEIGHT_DIR="${LOCKBOX_WEIGHT_DIR:-}"
LOCKBOX_Q_PARENT_DIR="${LOCKBOX_Q_PARENT_DIR:-}"
LOCKBOX_CARRIER_RGB_DIR="${LOCKBOX_CARRIER_RGB_DIR:-}"
LOCKBOX_CARRIER_RENDER_DIR="${LOCKBOX_CARRIER_RENDER_DIR:-}"

OUTPUT_NAME="${OUTPUT_NAME:-${STATIC_NAME}_${LOCKBOX_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/residual_tetris_level1_lockbox_v0/${OUTPUT_NAME}}"
CHECK_DIR="${CHECK_DIR:-${WORK_ROOT}/check/residual_tetris_level1_lockbox_v0/${OUTPUT_NAME}}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
BOUNDED_DELTA_CLIP="${BOUNDED_DELTA_CLIP:--1}"
BOUNDED_MODE="${BOUNDED_MODE:-from_config}"
VISUAL_SIGNED_SCALE="${VISUAL_SIGNED_SCALE:-4.0}"
ERROR_SCALE="${ERROR_SCALE:-8.0}"
LP_SCALE="${LP_SCALE:-16.0}"
LEAK_SCALE="${LEAK_SCALE:-24.0}"
OUT_OF_RANGE_SCALE="${OUT_OF_RANGE_SCALE:-1.0}"
CHANGED_THRESHOLD="${CHANGED_THRESHOLD:-0.00392156862745098}"
CAMERA_INDEX_OFFSET="${CAMERA_INDEX_OFFSET:-0}"
WRITE_BUFFERS="${WRITE_BUFFERS:-1}"

for required in \
  "${STATIC_V1_DIR}/v1_manifest.json" \
  "${STATIC_V1_DIR}/renderer_config.json" \
  "${STATIC_V1_DIR}/cells_deploy_top40_raw.json" \
  "${STATIC_V1_DIR}/cells_core28.json" \
  "${STATIC_V1_DIR}/cells_minimal_clean_dev.json" \
  "${STATIC_V1_DIR}/frozen_cells_3d.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[level1-lockbox-v0] required frozen static V1 path not found: ${required}" >&2
    echo "[level1-lockbox-v0] rerun scripts/run_residual_tetris_static_v1_kitchen.sh after pulling latest code." >&2
    exit 1
  fi
done

for name in LOCKBOX_PRIMITIVE_DIR LOCKBOX_BASE_RENDER_DIR LOCKBOX_SR_DIR LOCKBOX_WEIGHT_DIR; do
  value="${!name:-}"
  if [[ -z "${value}" ]]; then
    echo "[level1-lockbox-v0] ${name} is required. Do not default to development views for lockbox." >&2
    exit 1
  fi
  if [[ ! -e "${value}" ]]; then
    echo "[level1-lockbox-v0] required lockbox path not found: ${value}" >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_DIR}" "${CHECK_DIR}"
fi
mkdir -p "${CHECK_DIR}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[level1-lockbox-v0] static   : ${STATIC_V1_DIR}"
echo "[level1-lockbox-v0] primitive: ${LOCKBOX_PRIMITIVE_DIR}"
echo "[level1-lockbox-v0] base     : ${LOCKBOX_BASE_RENDER_DIR}"
echo "[level1-lockbox-v0] sr       : ${LOCKBOX_SR_DIR}"
echo "[level1-lockbox-v0] weight   : ${LOCKBOX_WEIGHT_DIR}"
echo "[level1-lockbox-v0] output   : ${OUTPUT_DIR}"
echo "[level1-lockbox-v0] check    : ${CHECK_DIR}"

ARGS=(
  --static_v1_dir "${STATIC_V1_DIR}"
  --oracle_dir "${ORACLE_DIR}"
  --primitive_dir "${LOCKBOX_PRIMITIVE_DIR}"
  --base_render_dir "${LOCKBOX_BASE_RENDER_DIR}"
  --sr_dir "${LOCKBOX_SR_DIR}"
  --weight_dir "${LOCKBOX_WEIGHT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --check_dir "${CHECK_DIR}"
  --match_policy "${MATCH_POLICY}"
  --limit "${LIMIT}"
  --bounded_delta_clip "${BOUNDED_DELTA_CLIP}"
  --bounded_mode "${BOUNDED_MODE}"
  --visual_signed_scale "${VISUAL_SIGNED_SCALE}"
  --error_scale "${ERROR_SCALE}"
  --lp_scale "${LP_SCALE}"
  --leak_scale "${LEAK_SCALE}"
  --out_of_range_scale "${OUT_OF_RANGE_SCALE}"
  --changed_threshold "${CHANGED_THRESHOLD}"
  --camera_index_offset "${CAMERA_INDEX_OFFSET}"
  --write_buffers "${WRITE_BUFFERS}"
)

if [[ -n "${LOCKBOX_Q_PARENT_DIR}" ]]; then
  ARGS+=(--q_parent_dir "${LOCKBOX_Q_PARENT_DIR}")
fi
if [[ -n "${LOCKBOX_CARRIER_RGB_DIR}" ]]; then
  ARGS+=(--carrier_rgb_dir "${LOCKBOX_CARRIER_RGB_DIR}")
fi
if [[ -n "${LOCKBOX_CARRIER_RENDER_DIR}" ]]; then
  ARGS+=(--carrier_render_dir "${LOCKBOX_CARRIER_RENDER_DIR}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/render_residual_tetris_level1_lockbox_v0.py" "${ARGS[@]}"

echo "[level1-lockbox-v0] shallow outputs:"
echo "  ${CHECK_DIR}/summary.json"
echo "  ${CHECK_DIR}/level1_lockbox_metrics.json"
echo "  ${CHECK_DIR}/per_view_metrics.json"
echo "  ${CHECK_DIR}/lockbox_manifest.json"
echo "  ${CHECK_DIR}/visuals/deploy_top40_raw/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/deploy_top40_bounded/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/deploy_top40_minimal_clean_dev/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/core28/base_plus_residual"

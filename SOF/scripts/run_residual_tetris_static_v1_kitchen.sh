#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"

ORACLE_NAME="${ORACLE_NAME:-${BASE_EXPERIMENT_NAME}_residual_tetris_oracle_v0}"
ORACLE_DIR="${ORACLE_DIR:-${SOF_ROOT}/output/residual_tetris_oracle_v0/${ORACLE_NAME}}"
OUTPUT_NAME="${OUTPUT_NAME:-${ORACLE_NAME}_static_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/residual_tetris_static_v1/${OUTPUT_NAME}}"
CHECK_DIR="${CHECK_DIR:-${WORK_ROOT}/check/residual_tetris_static_v1/${OUTPUT_NAME}}"

OVERWRITE="${OVERWRITE:-0}"
BOUNDED_DELTA_CLIP="${BOUNDED_DELTA_CLIP:-0.08}"
BOUNDED_MODE="${BOUNDED_MODE:-per_cell_clip}"
MINIMAL_CLEAN_DROP_CLUSTER_IDS="${MINIMAL_CLEAN_DROP_CLUSTER_IDS:-165}"
DOSE_COUNTS="${DOSE_COUNTS:-5,10,20,40}"
FOCUS_CLUSTER_IDS="${FOCUS_CLUSTER_IDS:-64,174,70,133,156,223}"
VISUAL_SIGNED_SCALE="${VISUAL_SIGNED_SCALE:-4.0}"
ERROR_SCALE="${ERROR_SCALE:-8.0}"
LP_SCALE="${LP_SCALE:-16.0}"
LEAK_SCALE="${LEAK_SCALE:-24.0}"
WRITE_BUFFERS="${WRITE_BUFFERS:-1}"
WRITE_PER_CELL_BUFFERS="${WRITE_PER_CELL_BUFFERS:-1}"
PER_CELL_BUFFER_VARIANT="${PER_CELL_BUFFER_VARIANT:-deploy_top40_raw}"
CLOSURE_SMALL_COUNT="${CLOSURE_SMALL_COUNT:-5}"
CLOSURE_SINGLE_CLUSTER_ID="${CLOSURE_SINGLE_CLUSTER_ID:--1}"

for required in "${ORACLE_DIR}/summary.json" "${ORACLE_DIR}/deploy_selected_rows.json" "${ORACLE_DIR}/core_selected_rows.json" "${ORACLE_DIR}/rows.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[residual-static-v1] required path not found: ${required}" >&2
    echo "[residual-static-v1] run scripts/run_residual_tetris_oracle_v0_kitchen.sh first or set ORACLE_DIR." >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_DIR}" "${CHECK_DIR}"
fi
mkdir -p "${CHECK_DIR}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[residual-static-v1] oracle : ${ORACLE_DIR}"
echo "[residual-static-v1] output : ${OUTPUT_DIR}"
echo "[residual-static-v1] check  : ${CHECK_DIR}"

ARGS=(
  --oracle_dir "${ORACLE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --check_dir "${CHECK_DIR}"
  --bounded_delta_clip "${BOUNDED_DELTA_CLIP}"
  --bounded_mode "${BOUNDED_MODE}"
  --minimal_clean_drop_cluster_ids "${MINIMAL_CLEAN_DROP_CLUSTER_IDS}"
  --dose_counts "${DOSE_COUNTS}"
  --focus_cluster_ids "${FOCUS_CLUSTER_IDS}"
  --visual_signed_scale "${VISUAL_SIGNED_SCALE}"
  --error_scale "${ERROR_SCALE}"
  --lp_scale "${LP_SCALE}"
  --leak_scale "${LEAK_SCALE}"
  --write_buffers "${WRITE_BUFFERS}"
  --write_per_cell_buffers "${WRITE_PER_CELL_BUFFERS}"
  --per_cell_buffer_variant "${PER_CELL_BUFFER_VARIANT}"
  --closure_small_count "${CLOSURE_SMALL_COUNT}"
  --closure_single_cluster_id "${CLOSURE_SINGLE_CLUSTER_ID}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/render_residual_tetris_static_v1.py" "${ARGS[@]}"

echo "[residual-static-v1] shallow outputs:"
echo "  ${CHECK_DIR}/v1_manifest.json"
echo "  ${CHECK_DIR}/renderer_config.json"
echo "  ${CHECK_DIR}/cells_deploy_top40_raw.json"
echo "  ${CHECK_DIR}/cells_core28.json"
echo "  ${CHECK_DIR}/cells_minimal_clean_dev.json"
echo "  ${CHECK_DIR}/static_v1_metrics.json"
echo "  ${CHECK_DIR}/numeric_closure.json"
echo "  ${CHECK_DIR}/visuals/deploy_top40_raw/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/deploy_top40_bounded/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/deploy_top40_minimal_clean_dev/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/core28/base_plus_residual"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"

ORACLE_NAME="${ORACLE_NAME:-${BASE_EXPERIMENT_NAME}_residual_tetris_oracle_v0}"
ORACLE_DIR="${ORACLE_DIR:-${SOF_ROOT}/output/residual_tetris_oracle_v0/${ORACLE_NAME}}"
OUTPUT_NAME="${OUTPUT_NAME:-${ORACLE_NAME}_deploy_residual_preview_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/residual_tetris_preview_v0/${OUTPUT_NAME}}"
CHECK_DIR="${CHECK_DIR:-${WORK_ROOT}/check/residual_tetris_preview_v0/${OUTPUT_NAME}}"

OVERWRITE="${OVERWRITE:-0}"
BOUNDED_DELTA_CLIP="${BOUNDED_DELTA_CLIP:-0.08}"
DOSE_COUNTS="${DOSE_COUNTS:-5,10,20,40}"
FOCUS_CLUSTER_IDS="${FOCUS_CLUSTER_IDS:-64,174,70,133,156,223}"
VISUAL_SIGNED_SCALE="${VISUAL_SIGNED_SCALE:-4.0}"
ERROR_SCALE="${ERROR_SCALE:-8.0}"
LP_SCALE="${LP_SCALE:-16.0}"
LEAK_SCALE="${LEAK_SCALE:-24.0}"
CLEAN_NEGATIVE_VIEW_RATIO="${CLEAN_NEGATIVE_VIEW_RATIO:-0.5}"
CLEAN_LP_DRIFT_MARGINAL_MIN="${CLEAN_LP_DRIFT_MARGINAL_MIN:-0.0}"
CLEAN_LEAKAGE_MARGINAL_MIN="${CLEAN_LEAKAGE_MARGINAL_MIN:-0.0}"
MINIMAL_CLEAN_DROP_CLUSTER_IDS="${MINIMAL_CLEAN_DROP_CLUSTER_IDS:-165}"

for required in "${ORACLE_DIR}/summary.json" "${ORACLE_DIR}/deploy_selected_rows.json" "${ORACLE_DIR}/core_selected_rows.json" "${ORACLE_DIR}/rows.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[residual-preview-v0] required path not found: ${required}" >&2
    echo "[residual-preview-v0] run scripts/run_residual_tetris_oracle_v0_kitchen.sh first or set ORACLE_DIR." >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_DIR}" "${CHECK_DIR}"
fi
mkdir -p "${CHECK_DIR}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[residual-preview-v0] oracle : ${ORACLE_DIR}"
echo "[residual-preview-v0] output : ${OUTPUT_DIR}"
echo "[residual-preview-v0] check  : ${CHECK_DIR}"

ARGS=(
  --oracle_dir "${ORACLE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --check_dir "${CHECK_DIR}"
  --bounded_delta_clip "${BOUNDED_DELTA_CLIP}"
  --dose_counts "${DOSE_COUNTS}"
  --focus_cluster_ids "${FOCUS_CLUSTER_IDS}"
  --visual_signed_scale "${VISUAL_SIGNED_SCALE}"
  --error_scale "${ERROR_SCALE}"
  --lp_scale "${LP_SCALE}"
  --leak_scale "${LEAK_SCALE}"
  --clean_negative_view_ratio "${CLEAN_NEGATIVE_VIEW_RATIO}"
  --clean_lp_drift_marginal_min "${CLEAN_LP_DRIFT_MARGINAL_MIN}"
  --clean_leakage_marginal_min "${CLEAN_LEAKAGE_MARGINAL_MIN}"
  --minimal_clean_drop_cluster_ids "${MINIMAL_CLEAN_DROP_CLUSTER_IDS}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/render_residual_tetris_preview_v0.py" "${ARGS[@]}"

echo "[residual-preview-v0] shallow outputs:"
echo "  ${CHECK_DIR}/summary.json"
echo "  ${CHECK_DIR}/joint_metrics.json"
echo "  ${CHECK_DIR}/dose_curve.json"
echo "  ${CHECK_DIR}/leave_one_cell_out.json"
echo "  ${CHECK_DIR}/per_cell_marginals.json"
echo "  ${CHECK_DIR}/minimal_clean_selected_rows.json"
echo "  ${CHECK_DIR}/clean_selected_rows.json"
echo "  ${CHECK_DIR}/clean_rejected_rows.json"
echo "  ${CHECK_DIR}/negative_view_diagnostics.json"
echo "  ${CHECK_DIR}/visuals/deploy_top40_raw/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/deploy_top40_bounded/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/deploy_top40_minimal_clean_dev/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/deploy_top40_clean29/base_plus_residual"
echo "  ${CHECK_DIR}/visuals/core28/base_plus_residual"
echo "  ${CHECK_DIR}/negative_view_diagnostics"
echo "  ${CHECK_DIR}/cell_sheet"

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

ATTRIB_NAME="${ATTRIB_NAME:-failure_attr_v2_gt_qproxy_scaled}"
OUTPUT_NAME="${OUTPUT_NAME:-${STATIC_NAME}_${ATTRIB_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/residual_tetris_failure_attribution_v2/${OUTPUT_NAME}}"
CHECK_DIR="${CHECK_DIR:-${WORK_ROOT}/check/residual_tetris_failure_attribution_v2/${OUTPUT_NAME}}"

LOCKBOX_PRIMITIVE_DIR="${LOCKBOX_PRIMITIVE_DIR:-}"
LOCKBOX_BASE_RENDER_DIR="${LOCKBOX_BASE_RENDER_DIR:-}"
LOCKBOX_TARGET_DIR="${LOCKBOX_TARGET_DIR:-}"
LOCKBOX_WEIGHT_DIR="${LOCKBOX_WEIGHT_DIR:-}"
LOCKBOX_Q_PARENT_DIR="${LOCKBOX_Q_PARENT_DIR:-}"
LOCKBOX_ALT_TARGET_DIR="${LOCKBOX_ALT_TARGET_DIR:-}"

TARGET_TYPE="${TARGET_TYPE:-}"
ALT_TARGET_TYPE="${ALT_TARGET_TYPE:-}"
MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-0}"
CAMERA_INDEX_OFFSET="${CAMERA_INDEX_OFFSET:-0}"
CELL_SET="${CELL_SET:-deploy_top40_raw}"
Q_MODES="${Q_MODES:-proxy,proxy_scaled,true,unit_visibility}"
LAMBDAS="${LAMBDAS:-0.125,0.25,0.5,1.0}"
SIGNS="${SIGNS:-plus,minus}"
DEV_Q_REFERENCE="${DEV_Q_REFERENCE:-0.10}"
Q_SCALE_STAT="${Q_SCALE_STAT:-median}"
SHIFT_GRID="${SHIFT_GRID:--2,-1,0,1,2}"
SHIFT_Q_MODES="${SHIFT_Q_MODES:-proxy_scaled,true}"
SHIFT_SIGNS="${SHIFT_SIGNS:-plus}"
SHIFT_LAMBDAS="${SHIFT_LAMBDAS:-1.0}"
WRITE_PER_CELL_REPORT="${WRITE_PER_CELL_REPORT:-1}"
TARGET_ACTIVE_THRESHOLD="${TARGET_ACTIVE_THRESHOLD:-0.01}"
WRITE_VISUALS="${WRITE_VISUALS:-1}"
VISUAL_VARIANT_LIMIT="${VISUAL_VARIANT_LIMIT:-8}"
OVERWRITE="${OVERWRITE:-0}"

for required in \
  "${STATIC_V1_DIR}/v1_manifest.json" \
  "${STATIC_V1_DIR}/renderer_config.json" \
  "${STATIC_V1_DIR}/cells_deploy_top40_raw.json" \
  "${STATIC_V1_DIR}/cells_core28.json" \
  "${STATIC_V1_DIR}/cells_minimal_clean_dev.json" \
  "${STATIC_V1_DIR}/frozen_cells_3d.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[failure-attribution-v2] required frozen static V1 path not found: ${required}" >&2
    exit 1
  fi
done

for name in LOCKBOX_PRIMITIVE_DIR LOCKBOX_BASE_RENDER_DIR LOCKBOX_TARGET_DIR LOCKBOX_WEIGHT_DIR; do
  value="${!name:-}"
  if [[ -z "${value}" ]]; then
    echo "[failure-attribution-v2] ${name} is required." >&2
    exit 1
  fi
  if [[ ! -e "${value}" ]]; then
    echo "[failure-attribution-v2] required lockbox path not found: ${value}" >&2
    exit 1
  fi
done
if [[ -n "${LOCKBOX_ALT_TARGET_DIR}" && ! -e "${LOCKBOX_ALT_TARGET_DIR}" ]]; then
  echo "[failure-attribution-v2] required alt target path not found: ${LOCKBOX_ALT_TARGET_DIR}" >&2
  exit 1
fi
if [[ -n "${LOCKBOX_Q_PARENT_DIR}" && ! -e "${LOCKBOX_Q_PARENT_DIR}" ]]; then
  echo "[failure-attribution-v2] required q_parent path not found: ${LOCKBOX_Q_PARENT_DIR}" >&2
  exit 1
fi

lower_name="$(printf '%s %s %s' "${ATTRIB_NAME}" "${OUTPUT_NAME}" "${Q_MODES}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "${LOCKBOX_Q_PARENT_DIR}" && ( "${lower_name}" == *"qtrue"* || "${lower_name}" == *"true_q"* || "${lower_name}" == *"true-donor"* || "${lower_name}" == *"true_donor"* ) ]]; then
  echo "[failure-attribution-v2] experiment requests true donor q but LOCKBOX_Q_PARENT_DIR is empty; refusing fallback." >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_DIR}" "${CHECK_DIR}"
fi
mkdir -p "${CHECK_DIR}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[failure-attribution-v2] static    : ${STATIC_V1_DIR}"
echo "[failure-attribution-v2] primitive : ${LOCKBOX_PRIMITIVE_DIR}"
echo "[failure-attribution-v2] base      : ${LOCKBOX_BASE_RENDER_DIR}"
echo "[failure-attribution-v2] target    : ${LOCKBOX_TARGET_DIR}"
echo "[failure-attribution-v2] alt target: ${LOCKBOX_ALT_TARGET_DIR:-<none>}"
echo "[failure-attribution-v2] q_parent  : ${LOCKBOX_Q_PARENT_DIR:-<none>}"
echo "[failure-attribution-v2] output    : ${OUTPUT_DIR}"

ARGS=(
  --static_v1_dir "${STATIC_V1_DIR}"
  --oracle_dir "${ORACLE_DIR}"
  --primitive_dir "${LOCKBOX_PRIMITIVE_DIR}"
  --base_render_dir "${LOCKBOX_BASE_RENDER_DIR}"
  --target_dir "${LOCKBOX_TARGET_DIR}"
  --weight_dir "${LOCKBOX_WEIGHT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --check_dir "${CHECK_DIR}"
  --target_type "${TARGET_TYPE}"
  --alt_target_type "${ALT_TARGET_TYPE}"
  --match_policy "${MATCH_POLICY}"
  --limit "${LIMIT}"
  --camera_index_offset "${CAMERA_INDEX_OFFSET}"
  --cell_set "${CELL_SET}"
  --q_modes "${Q_MODES}"
  --lambdas "${LAMBDAS}"
  --signs "${SIGNS}"
  --dev_q_reference "${DEV_Q_REFERENCE}"
  --q_scale_stat "${Q_SCALE_STAT}"
  "--shift_grid=${SHIFT_GRID}"
  --shift_q_modes "${SHIFT_Q_MODES}"
  --shift_signs "${SHIFT_SIGNS}"
  --shift_lambdas "${SHIFT_LAMBDAS}"
  --write_per_cell_report "${WRITE_PER_CELL_REPORT}"
  --target_active_threshold "${TARGET_ACTIVE_THRESHOLD}"
  --write_visuals "${WRITE_VISUALS}"
  --visual_variant_limit "${VISUAL_VARIANT_LIMIT}"
)
if [[ -n "${LOCKBOX_Q_PARENT_DIR}" ]]; then
  ARGS+=(--q_parent_dir "${LOCKBOX_Q_PARENT_DIR}")
fi
if [[ -n "${LOCKBOX_ALT_TARGET_DIR}" ]]; then
  ARGS+=(--alt_target_dir "${LOCKBOX_ALT_TARGET_DIR}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_residual_tetris_failure_attribution_v0.py" "${ARGS[@]}"

echo "[failure-attribution-v2] shallow outputs:"
echo "  ${CHECK_DIR}/summary.json"
echo "  ${CHECK_DIR}/metrics.json"
echo "  ${CHECK_DIR}/per_view_metrics.json"
echo "  ${CHECK_DIR}/q_distribution_report.json"
echo "  ${CHECK_DIR}/target_similarity_report.json"
echo "  ${CHECK_DIR}/per_cell_failure_report.json"
echo "  ${CHECK_DIR}/visuals"

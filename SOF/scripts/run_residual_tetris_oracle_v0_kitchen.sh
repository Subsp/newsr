#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting}"

BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/${BASE_EXPERIMENT_NAME}}"
BASE_ITERATION="${BASE_ITERATION:-30000}"
BASE_RENDER_DIR="${BASE_RENDER_DIR:-${BASE_MODEL_DIR}/train/ours_${BASE_ITERATION}/test_preds_1}"

SR_PRIOR_NAME="${SR_PRIOR_NAME:-qwen_steps1_seed42_rcgm_aligned_images2_train244_v0}"
SR_DIR="${SR_DIR:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${SR_PRIOR_NAME}/fused_priors}"

EVIDENCE_NAME="${EVIDENCE_NAME:-qwen_vosr_sr_hf_effective_8view_v0}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${SCENE_ASSET_ROOT}/sr_hf_evidence/${EVIDENCE_NAME}}"
WEIGHT_DIR="${WEIGHT_DIR:-${EVIDENCE_ROOT}/effective_hf_weight}"

CARRIER_NAME="${CARRIER_NAME:-qwen_vosr_effective_hf_2dgs_one_v0}"
CARRIER_ROOT="${CARRIER_ROOT:-${SOF_ROOT}/output/2dgs_sr_hf_evidence_carrier/${CARRIER_NAME}}"
PRIMITIVE_DIR="${PRIMITIVE_DIR:-${CARRIER_ROOT}/primitives}"
CARRIER_RGB_DIR="${CARRIER_RGB_DIR:-${CARRIER_ROOT}/evidence_target}"
CARRIER_RENDER_DIR="${CARRIER_RENDER_DIR:-${CARRIER_ROOT}/evidence_render}"

Q_PARENT_DIR="${Q_PARENT_DIR:-}"
OUTPUT_NAME="${OUTPUT_NAME:-${BASE_EXPERIMENT_NAME}_residual_tetris_oracle_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/residual_tetris_oracle_v0/${OUTPUT_NAME}}"
CHECK_DIR="${CHECK_DIR:-${WORK_ROOT}/check/residual_tetris_oracle_v0/${OUTPUT_NAME}}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-8}"
OVERWRITE="${OVERWRITE:-0}"
MAX_CLUSTERS="${MAX_CLUSTERS:-256}"
MAX_PRIMITIVES_PER_VIEW="${MAX_PRIMITIVES_PER_VIEW:-32768}"
MIN_Q="${MIN_Q:-0.01}"

MIN_CLUSTER_VIEWS="${MIN_CLUSTER_VIEWS:-3}"
MIN_FIT_VIEWS="${MIN_FIT_VIEWS:-1}"
MIN_SELECTION_VIEWS="${MIN_SELECTION_VIEWS:-1}"
MIN_TEST_VIEWS="${MIN_TEST_VIEWS:-1}"
MIN_TARGET_ENERGY="${MIN_TARGET_ENERGY:-0.00001}"
MIN_ACTIVE_AREA="${MIN_ACTIVE_AREA:-24}"

PIECE_TYPES="${PIECE_TYPES:-signed_single,dipole,dog,split}"
PIECE_SCALES="${PIECE_SCALES:-0.75,1.0,1.5}"
ORIENTATION_DEGS="${ORIENTATION_DEGS:--10,0,10}"
PHASES="${PHASES:--1,1}"

BETA_MAX="${BETA_MAX:-0.35}"
LAMBDA_OFF="${LAMBDA_OFF:-0.25}"
LAMBDA_LP="${LAMBDA_LP:-0.05}"
LAMBDA_DC="${LAMBDA_DC:-0.05}"
NUM_WRONG_SLOTS="${NUM_WRONG_SLOTS:-5}"
NUM_SHUFFLED_Q="${NUM_SHUFFLED_Q:-5}"
NULL_PERCENTILE="${NULL_PERCENTILE:-95}"
SELECTOR_CORR_THRESHOLD="${SELECTOR_CORR_THRESHOLD:-0.20}"
SELECTOR_EE_THRESHOLD="${SELECTOR_EE_THRESHOLD:-0.05}"
SELECTOR_LEAK_MAX="${SELECTOR_LEAK_MAX:-0.30}"
SELECTOR_LP_DRIFT_MAX="${SELECTOR_LP_DRIFT_MAX:-1.0}"
SELECTOR_DELIVERY_RETENTION_MIN="${SELECTOR_DELIVERY_RETENTION_MIN:-0.50}"
SELECTOR_BETA_SATURATION_MAX="${SELECTOR_BETA_SATURATION_MAX:-0.50}"
SELECTOR_BETA_MAX_FRACTION_MAX="${SELECTOR_BETA_MAX_FRACTION_MAX:-0.90}"
SELECTOR_NULL_MARGIN_MIN="${SELECTOR_NULL_MARGIN_MIN:-0.0}"
SELECTOR_CURVE_COUNTS="${SELECTOR_CURVE_COUNTS:-8,16,28,40,56,74,120}"
SELECTOR_CURVE_RATIOS="${SELECTOR_CURVE_RATIOS:-0.05,0.10,0.15,0.20,0.30,0.50}"
SELECTOR_CURVE_RANK_KEYS="${SELECTOR_CURVE_RANK_KEYS:-selector_score,C_selection_explained_energy,A_selection_explained_energy,cluster_target_energy,C_selection_gain}"
TEST_GOOD_CORR_THRESHOLD="${TEST_GOOD_CORR_THRESHOLD:-0.20}"
TEST_GOOD_EE_THRESHOLD="${TEST_GOOD_EE_THRESHOLD:-0.05}"
TEST_GOOD_LEAK_MAX="${TEST_GOOD_LEAK_MAX:-0.30}"
TEST_GOOD_LP_DRIFT_MAX="${TEST_GOOD_LP_DRIFT_MAX:-1.0}"
RESPONSIBILITY_BG_TAU="${RESPONSIBILITY_BG_TAU:-0.25}"
CORE_WEIGHT_THRESHOLD="${CORE_WEIGHT_THRESHOLD:-0.015}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.03}"
TOLERANCE_RADIUS="${TOLERANCE_RADIUS:-3}"

DEBUG_LIMIT="${DEBUG_LIMIT:-12}"

BASE_PLY="${BASE_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply"
for required in "${BASE_MODEL_DIR}" "${BASE_PLY}" "${BASE_RENDER_DIR}" "${SR_DIR}" "${WEIGHT_DIR}" "${PRIMITIVE_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[residual-tetris-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_DIR}" "${CHECK_DIR}"
fi
mkdir -p "${OUTPUT_DIR}" "${CHECK_DIR}"

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[residual-tetris-v0] base      : ${BASE_MODEL_DIR}"
echo "[residual-tetris-v0] base render: ${BASE_RENDER_DIR}"
echo "[residual-tetris-v0] sr        : ${SR_DIR}"
echo "[residual-tetris-v0] weight    : ${WEIGHT_DIR}"
echo "[residual-tetris-v0] primitive : ${PRIMITIVE_DIR}"
echo "[residual-tetris-v0] output    : ${OUTPUT_DIR}"
echo "[residual-tetris-v0] check     : ${CHECK_DIR}"

ARGS=(
  --base_model_dir "${BASE_MODEL_DIR}"
  --base_iteration "${BASE_ITERATION}"
  --primitive_dir "${PRIMITIVE_DIR}"
  --base_render_dir "${BASE_RENDER_DIR}"
  --sr_dir "${SR_DIR}"
  --weight_dir "${WEIGHT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --match_policy "${MATCH_POLICY}"
  --limit "${LIMIT}"
  --max_clusters "${MAX_CLUSTERS}"
  --max_primitives_per_view "${MAX_PRIMITIVES_PER_VIEW}"
  --min_q "${MIN_Q}"
  --min_cluster_views "${MIN_CLUSTER_VIEWS}"
  --min_fit_views "${MIN_FIT_VIEWS}"
  --min_selection_views "${MIN_SELECTION_VIEWS}"
  --min_test_views "${MIN_TEST_VIEWS}"
  --min_target_energy "${MIN_TARGET_ENERGY}"
  --min_active_area "${MIN_ACTIVE_AREA}"
  --piece_types "${PIECE_TYPES}"
  --piece_scales "${PIECE_SCALES}"
  "--orientation_degs=${ORIENTATION_DEGS}"
  "--phases=${PHASES}"
  --beta_max "${BETA_MAX}"
  --lambda_off "${LAMBDA_OFF}"
  --lambda_lp "${LAMBDA_LP}"
  --lambda_dc "${LAMBDA_DC}"
  --num_wrong_slots "${NUM_WRONG_SLOTS}"
  --num_shuffled_q "${NUM_SHUFFLED_Q}"
  --null_percentile "${NULL_PERCENTILE}"
  --selector_corr_threshold "${SELECTOR_CORR_THRESHOLD}"
  --selector_ee_threshold "${SELECTOR_EE_THRESHOLD}"
  --selector_leak_max "${SELECTOR_LEAK_MAX}"
  --selector_lp_drift_max "${SELECTOR_LP_DRIFT_MAX}"
  --selector_delivery_retention_min "${SELECTOR_DELIVERY_RETENTION_MIN}"
  --selector_beta_saturation_max "${SELECTOR_BETA_SATURATION_MAX}"
  --selector_beta_max_fraction_max "${SELECTOR_BETA_MAX_FRACTION_MAX}"
  --selector_null_margin_min "${SELECTOR_NULL_MARGIN_MIN}"
  --selector_curve_counts "${SELECTOR_CURVE_COUNTS}"
  --selector_curve_ratios "${SELECTOR_CURVE_RATIOS}"
  --selector_curve_rank_keys "${SELECTOR_CURVE_RANK_KEYS}"
  --test_good_corr_threshold "${TEST_GOOD_CORR_THRESHOLD}"
  --test_good_ee_threshold "${TEST_GOOD_EE_THRESHOLD}"
  --test_good_leak_max "${TEST_GOOD_LEAK_MAX}"
  --test_good_lp_drift_max "${TEST_GOOD_LP_DRIFT_MAX}"
  --responsibility_bg_tau "${RESPONSIBILITY_BG_TAU}"
  --core_weight_threshold "${CORE_WEIGHT_THRESHOLD}"
  --support_threshold "${SUPPORT_THRESHOLD}"
  --tolerance_radius "${TOLERANCE_RADIUS}"
  --debug_limit "${DEBUG_LIMIT}"
)

if [[ -n "${Q_PARENT_DIR}" ]]; then
  ARGS+=(--q_parent_dir "${Q_PARENT_DIR}")
fi
if [[ -d "${CARRIER_RGB_DIR}" ]]; then
  ARGS+=(--carrier_rgb_dir "${CARRIER_RGB_DIR}")
fi
if [[ -d "${CARRIER_RENDER_DIR}" ]]; then
  ARGS+=(--carrier_render_dir "${CARRIER_RENDER_DIR}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/evaluate_residual_tetris_oracle_v0.py" "${ARGS[@]}"

cp "${OUTPUT_DIR}/summary.json" "${CHECK_DIR}/summary.json"
cp "${OUTPUT_DIR}/rows.json" "${CHECK_DIR}/rows.json"
cp "${OUTPUT_DIR}/selected_rows.json" "${CHECK_DIR}/selected_rows.json"
cp "${OUTPUT_DIR}/selector_ablation.json" "${CHECK_DIR}/selector_ablation.json"
cp "${OUTPUT_DIR}/selector_curve.json" "${CHECK_DIR}/selector_curve.json"
cp "${OUTPUT_DIR}/README.txt" "${CHECK_DIR}/README.txt"
if [[ -d "${OUTPUT_DIR}/debug" ]]; then
  mkdir -p "${CHECK_DIR}/debug"
  find "${OUTPUT_DIR}/debug" -type f -name '*.png' | head -80 | while read -r image_path; do
    rel="${image_path#${OUTPUT_DIR}/debug/}"
    mkdir -p "${CHECK_DIR}/debug/$(dirname "${rel}")"
    cp "${image_path}" "${CHECK_DIR}/debug/${rel}"
  done
fi

echo "[residual-tetris-v0] shallow outputs:"
echo "  ${CHECK_DIR}/summary.json"
echo "  ${CHECK_DIR}/rows.json"
echo "  ${CHECK_DIR}/selected_rows.json"
echo "  ${CHECK_DIR}/selector_ablation.json"
echo "  ${CHECK_DIR}/selector_curve.json"
echo "  ${CHECK_DIR}/README.txt"
echo "  ${CHECK_DIR}/debug"

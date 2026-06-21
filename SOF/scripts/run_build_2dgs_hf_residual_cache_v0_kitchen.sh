#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTERNAL_REPO_ROOT="${EXTERNAL_REPO_ROOT:-${WORK_ROOT}/external/GaussianImage}"

NPSE_ROOT="${NPSE_ROOT:-${SCENE_ASSET_ROOT}/npse_cache/render_x1_restormer_depthprior_npse_yellow_fidelity_nogate_full_v0}"
TARGET_DIR="${TARGET_DIR:-${NPSE_ROOT}/edge_target}"
MASK_DIR="${MASK_DIR:-${NPSE_ROOT}/trust_edge}"
ANCHOR_DIR="${ANCHOR_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0/train/ours_30000/test_preds_1}"

OUTPUT_NAME="${OUTPUT_NAME:-render_x1_restormer_gaussianimage_hf_residual_v0_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/2dgs_hf_residual/${OUTPUT_NAME}}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-8}"
DEBUG_LIMIT="${DEBUG_LIMIT:-24}"
OVERWRITE="${OVERWRITE:-0}"

HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-9}"
DETAIL_ALPHA="${DETAIL_ALPHA:-0.8}"
RESIDUAL_CLIP="${RESIDUAL_CLIP:-0.08}"
CONFIDENCE_POWER="${CONFIDENCE_POWER:-1.5}"
MASK_POWER="${MASK_POWER:-1.0}"

NUM_GAUSSIANS="${NUM_GAUSSIANS:-4096}"
GAUSSIANIMAGE_MODEL="${GAUSSIANIMAGE_MODEL:-cholesky}"
GAUSSIANIMAGE_OPTIMIZER="${GAUSSIANIMAGE_OPTIMIZER:-adam}"
ITERATIONS="${ITERATIONS:-700}"
LR="${LR:-0.001}"
LOSS="${LOSS:-l1_l2}"
LAMBDA_L1="${LAMBDA_L1:-0.5}"
LAMBDA_L2="${LAMBDA_L2:-0.5}"
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-0.005}"
FIT_TARGET_MODE="${FIT_TARGET_MODE:-hf_residual}"
RGB_LOSS_WEIGHT_MODE="${RGB_LOSS_WEIGHT_MODE:-full}"
OUTPUT_PROFILE="${OUTPUT_PROFILE:-full}"
NEUTRAL_OUTSIDE_MASK="${NEUTRAL_OUTSIDE_MASK:-0}"
INIT_RANDOM="${INIT_RANDOM:-0}"
INIT_MIN_SCORE="${INIT_MIN_SCORE:-0.035}"
INIT_NMS_RADIUS_PX="${INIT_NMS_RADIUS_PX:-2}"
INIT_MAX_CANDIDATES="${INIT_MAX_CANDIDATES:-0}"
INIT_WEIGHT_POWER="${INIT_WEIGHT_POWER:-0.5}"
INIT_ORIENTATION_RADIUS_PX="${INIT_ORIENTATION_RADIUS_PX:-5}"
INIT_SIGMA_LONG_PX="${INIT_SIGMA_LONG_PX:-5.0}"
INIT_SIGMA_SHORT_PX="${INIT_SIGMA_SHORT_PX:-0.8}"
INIT_COHERENCE_LONG_BOOST="${INIT_COHERENCE_LONG_BOOST:-0.75}"
SEGMENT_INIT="${SEGMENT_INIT:-1}"
SEGMENT_SAMPLES_PER_SEED="${SEGMENT_SAMPLES_PER_SEED:-7}"
SEGMENT_STEP_PX="${SEGMENT_STEP_PX:-2.0}"
SEGMENT_SEED_NMS_RADIUS_PX="${SEGMENT_SEED_NMS_RADIUS_PX:-8}"
SEGMENT_TRACE_SEARCH_RADIUS_PX="${SEGMENT_TRACE_SEARCH_RADIUS_PX:-2}"
SEGMENT_TURN_MIN_COS="${SEGMENT_TURN_MIN_COS:-0.45}"
SEGMENT_MIN_SCORE="${SEGMENT_MIN_SCORE:--1.0}"
SEGMENT_ANCHOR_WEIGHT="${SEGMENT_ANCHOR_WEIGHT:-0.015}"
SEGMENT_PAIR_WEIGHT="${SEGMENT_PAIR_WEIGHT:-0.030}"
SEGMENT_SHAPE_WEIGHT="${SEGMENT_SHAPE_WEIGHT:-0.001}"
SEGMENT_COLOR_SMOOTH_WEIGHT="${SEGMENT_COLOR_SMOOTH_WEIGHT:-0.002}"
LIGHT_VIS="${LIGHT_VIS:-0}"
LIGHT_VIS_STRENGTH="${LIGHT_VIS_STRENGTH:-0.75}"
SAVE_PT="${SAVE_PT:-0}"

for required in "${EXTERNAL_REPO_ROOT}" "${TARGET_DIR}" "${MASK_DIR}" "${ANCHOR_DIR}"; do
  if [[ ! -d "${required}" ]]; then
    echo "[2dgs-hf-v0] required path not found: ${required}" >&2
    if [[ "${required}" == "${EXTERNAL_REPO_ROOT}" ]]; then
      echo "[2dgs-hf-v0] clone official GaussianImage first:" >&2
      echo "  mkdir -p ${WORK_ROOT}/external && git clone https://github.com/Xinjie-Q/GaussianImage.git ${EXTERNAL_REPO_ROOT}" >&2
    fi
    exit 1
  fi
done

ARGS=(
  --external_repo_root "${EXTERNAL_REPO_ROOT}"
  --target_dir "${TARGET_DIR}"
  --anchor_dir "${ANCHOR_DIR}"
  --mask_dir "${MASK_DIR}"
  --output_dir "${OUTPUT_ROOT}"
  --match_policy "${MATCH_POLICY}"
  --highpass_kernel "${HIGHPASS_KERNEL}"
  --detail_alpha "${DETAIL_ALPHA}"
  --residual_clip "${RESIDUAL_CLIP}"
  --confidence_power "${CONFIDENCE_POWER}"
  --mask_power "${MASK_POWER}"
  --num_gaussians "${NUM_GAUSSIANS}"
  --model "${GAUSSIANIMAGE_MODEL}"
  --optimizer "${GAUSSIANIMAGE_OPTIMIZER}"
  --iterations "${ITERATIONS}"
  --lr "${LR}"
  --loss "${LOSS}"
  --lambda_l1 "${LAMBDA_L1}"
  --lambda_l2 "${LAMBDA_L2}"
  --background_weight "${BACKGROUND_WEIGHT}"
  --fit_target_mode "${FIT_TARGET_MODE}"
  --rgb_loss_weight_mode "${RGB_LOSS_WEIGHT_MODE}"
  --output_profile "${OUTPUT_PROFILE}"
  --init_min_score "${INIT_MIN_SCORE}"
  --init_nms_radius_px "${INIT_NMS_RADIUS_PX}"
  --init_max_candidates "${INIT_MAX_CANDIDATES}"
  --init_weight_power "${INIT_WEIGHT_POWER}"
  --init_orientation_radius_px "${INIT_ORIENTATION_RADIUS_PX}"
  --init_sigma_long_px "${INIT_SIGMA_LONG_PX}"
  --init_sigma_short_px "${INIT_SIGMA_SHORT_PX}"
  --init_coherence_long_boost "${INIT_COHERENCE_LONG_BOOST}"
  --segment_samples_per_seed "${SEGMENT_SAMPLES_PER_SEED}"
  --segment_step_px "${SEGMENT_STEP_PX}"
  --segment_seed_nms_radius_px "${SEGMENT_SEED_NMS_RADIUS_PX}"
  --segment_trace_search_radius_px "${SEGMENT_TRACE_SEARCH_RADIUS_PX}"
  --segment_turn_min_cos "${SEGMENT_TURN_MIN_COS}"
  --segment_min_score "${SEGMENT_MIN_SCORE}"
  --segment_anchor_weight "${SEGMENT_ANCHOR_WEIGHT}"
  --segment_pair_weight "${SEGMENT_PAIR_WEIGHT}"
  --segment_shape_weight "${SEGMENT_SHAPE_WEIGHT}"
  --segment_color_smooth_weight "${SEGMENT_COLOR_SMOOTH_WEIGHT}"
  --light_visual_strength "${LIGHT_VIS_STRENGTH}"
  --limit "${LIMIT}"
  --debug_limit "${DEBUG_LIMIT}"
)

if [[ "${INIT_RANDOM}" == "1" ]]; then
  ARGS+=(--init_random)
fi

if [[ "${SEGMENT_INIT}" == "1" ]]; then
  ARGS+=(--segment_init)
else
  ARGS+=(--no_segment_init)
fi

if [[ "${NEUTRAL_OUTSIDE_MASK}" == "1" ]]; then
  ARGS+=(--neutral_outside_mask)
else
  ARGS+=(--no_neutral_outside_mask)
fi

if [[ "${LIGHT_VIS}" == "1" ]]; then
  ARGS+=(--light_visuals)
fi

if [[ "${SAVE_PT}" == "1" ]]; then
  ARGS+=(--save_pt)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "[gaussianimage-hf-v0] external: ${EXTERNAL_REPO_ROOT}"
echo "[gaussianimage-hf-v0] target  : ${TARGET_DIR}"
echo "[gaussianimage-hf-v0] anchor  : ${ANCHOR_DIR}"
echo "[gaussianimage-hf-v0] mask    : ${MASK_DIR}"
echo "[gaussianimage-hf-v0] output  : ${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_gaussianimage_hf_residual_cache_v0.py" "${ARGS[@]}"

echo "[gaussianimage-hf-v0] inspect dirs:"
echo "  ${OUTPUT_ROOT}/gs_delta_signed"
echo "  ${OUTPUT_ROOT}/gs_delta_rgb_signed"
echo "  ${OUTPUT_ROOT}/gs_delta_rgb_pos"
echo "  ${OUTPUT_ROOT}/gs_delta_rgb_neg"
echo "  ${OUTPUT_ROOT}/gs_delta_color_render"
echo "  ${OUTPUT_ROOT}/gs_delta_abs"
echo "  ${OUTPUT_ROOT}/gs_delta_alpha"
echo "  ${OUTPUT_ROOT}/gs_delta_primitives"
echo "  ${OUTPUT_ROOT}/primitives"
if [[ "${OUTPUT_PROFILE}" == "carrier_only" ]]; then
  exit 0
fi
echo "  ${OUTPUT_ROOT}/sheet"
echo "  ${OUTPUT_ROOT}/overlay"
echo "  ${OUTPUT_ROOT}/primitive_overlay"
echo "  ${OUTPUT_ROOT}/recon_hf"
echo "  ${OUTPUT_ROOT}/target_hf"
echo "  ${OUTPUT_ROOT}/rgb_recon_hf"
echo "  ${OUTPUT_ROOT}/rgb_recon_hf_weighted"
echo "  ${OUTPUT_ROOT}/rgb_recon"
echo "  ${OUTPUT_ROOT}/rgb_recon_error"
echo "  ${OUTPUT_ROOT}/rgb_delta_recon"
echo "  ${OUTPUT_ROOT}/rgb_delta_apply"
echo "  ${OUTPUT_ROOT}/rgb_delta_apply_error"
echo "  ${OUTPUT_ROOT}/rgb_delta_recon_trust"
echo "  ${OUTPUT_ROOT}/rgb_delta_apply_trust"
echo "  ${OUTPUT_ROOT}/rgb_delta_extra_outside"
echo "  ${OUTPUT_ROOT}/rgb_error_overlay"
echo "  ${OUTPUT_ROOT}/rgb_sheet"
echo "  ${OUTPUT_ROOT}/carrier_rgb_target"
echo "  ${OUTPUT_ROOT}/carrier_rgb_anchor"
echo "  ${OUTPUT_ROOT}/carrier_rgb_target_over_anchor"
echo "  ${OUTPUT_ROOT}/carrier_alpha"
if [[ "${LIGHT_VIS}" == "1" ]]; then
  echo "  ${OUTPUT_ROOT}/recon_abs_light"
  echo "  ${OUTPUT_ROOT}/target_abs_light"
  echo "  ${OUTPUT_ROOT}/primitive_overlay_light"
  echo "  ${OUTPUT_ROOT}/overlay_light"
fi

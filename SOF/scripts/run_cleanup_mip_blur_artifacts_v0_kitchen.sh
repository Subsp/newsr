#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-${MIP_ITERATION}}"

RUN_NAME="${RUN_NAME:-mip_blur_delete_v0}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/cleanup_mip_blur_artifacts_v0/${SCENE_NAME}/${RUN_NAME}}"
OUTPUT_MODEL="${OUTPUT_MODEL:-${RUN_ROOT}/cleaned_mip_model_delete_v1}"
RENDER_DIR="${RENDER_DIR:-${RUN_ROOT}/cleaned_mip_renders_no_gt_delete_v1}"

LR_IMAGES_SUBDIR="${LR_IMAGES_SUBDIR:-images_8}"
SR_IMAGES_SUBDIR="${SR_IMAGES_SUBDIR:-images_2}"
SR_MASK_THRESHOLD="${SR_MASK_THRESHOLD:-0.12}"
SR_MASK_MODE="${SR_MASK_MODE:-soft}"
SR_PRIOR_ROOT="${SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/sof_surface_v0_${LR_IMAGES_SUBDIR}_to_${SR_IMAGES_SUBDIR}_mask${SR_MASK_THRESHOLD}_${SR_MASK_MODE}}"
SR_PRIOR_SUBDIR="${SR_PRIOR_SUBDIR:-fused_priors}"
SR_PRIOR_MASK_SUBDIR="${SR_PRIOR_MASK_SUBDIR:-usable_masks}"
SR_ANCHOR_SUBDIR="${SR_ANCHOR_SUBDIR:-aligned_references}"
SR_PRIOR_CONSISTENCY_THRESHOLD="${SR_PRIOR_CONSISTENCY_THRESHOLD:-0.12}"

MAX_LR_VIEWS="${MAX_LR_VIEWS:-16}"
MAX_SR_VIEWS="${MAX_SR_VIEWS:-16}"
VISIBILITY_DOWNSAMPLE="${VISIBILITY_DOWNSAMPLE:-8}"
VISIBILITY_TOPK="${VISIBILITY_TOPK:-4}"
VISIBILITY_MAX_PATCH_RADIUS="${VISIBILITY_MAX_PATCH_RADIUS:-2}"

LOWPASS_KERNEL="${LOWPASS_KERNEL:-31}"
VEIL_WEIGHT="${VEIL_WEIGHT:-0.65}"
DETAIL_GAP_WEIGHT="${DETAIL_GAP_WEIGHT:-0.35}"
DETAIL_NORM_PERCENTILE="${DETAIL_NORM_PERCENTILE:-0.92}"
RESIDUAL_NORM_PERCENTILE="${RESIDUAL_NORM_PERCENTILE:-0.95}"
LOWFREQ_NORM_PERCENTILE="${LOWFREQ_NORM_PERCENTILE:-0.93}"
RADIUS_RISK_PX="${RADIUS_RISK_PX:-24}"
DELETE_QUANTILE="${DELETE_QUANTILE:-0.970}"
MIN_BLUR_SCORE="${MIN_BLUR_SCORE:-0.08}"
MIN_SR_SUPPORT="${MIN_SR_SUPPORT:-0.18}"
MIN_PRIOR_DETAIL="${MIN_PRIOR_DETAIL:-0.05}"
MIN_DETAIL_GAP="${MIN_DETAIL_GAP:-0.05}"
MIN_LOWFREQ_SCORE="${MIN_LOWFREQ_SCORE:-0.08}"
MIN_SR_RESIDUAL="${MIN_SR_RESIDUAL:-0.05}"
MIN_FOOTPRINT_RISK="${MIN_FOOTPRINT_RISK:-0.35}"
MIN_RADIUS_PX="${MIN_RADIUS_PX:-20}"
PRUNE_MAX_OPACITY="${PRUNE_MAX_OPACITY:-0.30}"
PRUNE_MAX_VISIBLE_FRACTION="${PRUNE_MAX_VISIBLE_FRACTION:-0.45}"
LR_BAD_RESIDUAL_THRESHOLD="${LR_BAD_RESIDUAL_THRESHOLD:-0.12}"
MAX_PRUNE_FRACTION="${MAX_PRUNE_FRACTION:-0.02}"
MAX_PRUNE_COUNT="${MAX_PRUNE_COUNT:-0}"
SOFT_MAX_OPACITY="${SOFT_MAX_OPACITY:-0.70}"
MAX_SOFT_FRACTION="${MAX_SOFT_FRACTION:-0.04}"
MAX_SOFT_COUNT="${MAX_SOFT_COUNT:-0}"
SOFT_OPACITY_SCALE="${SOFT_OPACITY_SCALE:-0.75}"
SOFT_SCALE_SHRINK="${SOFT_SCALE_SHRINK:-0.90}"
NUM_DEBUG_VIEWS="${NUM_DEBUG_VIEWS:-4}"

RUN_RENDER="${RUN_RENDER:-1}"
RENDER_SPLIT="${RENDER_SPLIT:-test}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -e "${MIP_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" ]]; then
  echo "[mip-blur-cleanup-v0] missing mip model: ${MIP_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" >&2
  exit 1
fi

if [[ ! -d "${SR_PRIOR_ROOT}/${SR_PRIOR_SUBDIR}" || ! -d "${SR_PRIOR_ROOT}/${SR_PRIOR_MASK_SUBDIR}" || ! -d "${SR_PRIOR_ROOT}/${SR_ANCHOR_SUBDIR}" ]]; then
  echo "[mip-blur-cleanup-v0] missing prepared SR prior cache: ${SR_PRIOR_ROOT}" >&2
  echo "[mip-blur-cleanup-v0] run run_mip_to_sof_surface_v0_kitchen.sh once, or pass SR_PRIOR_ROOT to an existing fused_priors/usable_masks/aligned_references cache." >&2
  exit 1
fi

echo "[mip-blur-cleanup-v0] scene       : ${SCENE_ROOT}"
echo "[mip-blur-cleanup-v0] mip model   : ${MIP_MODEL_PATH} iter=${MIP_ITERATION}"
echo "[mip-blur-cleanup-v0] SR cache    : ${SR_PRIOR_ROOT}"
echo "[mip-blur-cleanup-v0] output      : ${OUTPUT_MODEL}"
echo "[mip-blur-cleanup-v0] prune/soft  : quantile=${DELETE_QUANTILE} prune_cap=${MAX_PRUNE_FRACTION} soft_cap=${MAX_SOFT_FRACTION} min_radius=${MIN_RADIUS_PX}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/cleanup_mip_blur_artifacts_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --mip_model_path "${MIP_MODEL_PATH}" \
  --output_model_path "${OUTPUT_MODEL}" \
  --iteration "${MIP_ITERATION}" \
  --output_iteration "${OUTPUT_ITERATION}" \
  --lr_images_subdir "${LR_IMAGES_SUBDIR}" \
  --sr_images_subdir "${SR_IMAGES_SUBDIR}" \
  --sr_prior_root "${SR_PRIOR_ROOT}" \
  --sr_prior_subdir "${SR_PRIOR_SUBDIR}" \
  --sr_prior_mask_subdir "${SR_PRIOR_MASK_SUBDIR}" \
  --sr_anchor_subdir "${SR_ANCHOR_SUBDIR}" \
  --sr_prior_consistency_threshold "${SR_PRIOR_CONSISTENCY_THRESHOLD}" \
  --max_lr_views "${MAX_LR_VIEWS}" \
  --max_sr_views "${MAX_SR_VIEWS}" \
  --visibility_downsample "${VISIBILITY_DOWNSAMPLE}" \
  --visibility_topk "${VISIBILITY_TOPK}" \
  --visibility_max_patch_radius "${VISIBILITY_MAX_PATCH_RADIUS}" \
  --lowpass_kernel "${LOWPASS_KERNEL}" \
  --veil_weight "${VEIL_WEIGHT}" \
  --detail_gap_weight "${DETAIL_GAP_WEIGHT}" \
  --detail_norm_percentile "${DETAIL_NORM_PERCENTILE}" \
  --residual_norm_percentile "${RESIDUAL_NORM_PERCENTILE}" \
  --lowfreq_norm_percentile "${LOWFREQ_NORM_PERCENTILE}" \
  --radius_risk_px "${RADIUS_RISK_PX}" \
  --delete_quantile "${DELETE_QUANTILE}" \
  --min_blur_score "${MIN_BLUR_SCORE}" \
  --min_sr_support "${MIN_SR_SUPPORT}" \
  --min_prior_detail "${MIN_PRIOR_DETAIL}" \
  --min_detail_gap "${MIN_DETAIL_GAP}" \
  --min_lowfreq_score "${MIN_LOWFREQ_SCORE}" \
  --min_sr_residual "${MIN_SR_RESIDUAL}" \
  --min_footprint_risk "${MIN_FOOTPRINT_RISK}" \
  --min_radius_px "${MIN_RADIUS_PX}" \
  --prune_max_opacity "${PRUNE_MAX_OPACITY}" \
  --prune_max_visible_fraction "${PRUNE_MAX_VISIBLE_FRACTION}" \
  --lr_bad_residual_threshold "${LR_BAD_RESIDUAL_THRESHOLD}" \
  --max_prune_fraction "${MAX_PRUNE_FRACTION}" \
  --max_prune_count "${MAX_PRUNE_COUNT}" \
  --soft_max_opacity "${SOFT_MAX_OPACITY}" \
  --max_soft_fraction "${MAX_SOFT_FRACTION}" \
  --max_soft_count "${MAX_SOFT_COUNT}" \
  --soft_opacity_scale "${SOFT_OPACITY_SCALE}" \
  --soft_scale_shrink "${SOFT_SCALE_SHRINK}" \
  --num_debug_views "${NUM_DEBUG_VIEWS}" \
  --export_deleted_model

if [[ "${RUN_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${OUTPUT_MODEL}" \
    --output_dir "${RENDER_DIR}" \
    --images_subdir "${SR_IMAGES_SUBDIR}" \
    --iteration "${OUTPUT_ITERATION}" \
    --split "${RENDER_SPLIT}"
fi

echo "[done] cleaned mip model : ${OUTPUT_MODEL}"
echo "[done] output ply        : ${OUTPUT_MODEL}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
echo "[done] deleted subset    : ${OUTPUT_MODEL}_deleted_blur_artifacts"
echo "[done] summary           : ${OUTPUT_MODEL}/summary.json"
echo "[done] payload           : ${OUTPUT_MODEL}/mip_blur_cleanup_payload.pt"
if [[ "${RUN_RENDER}" == "1" ]]; then
  echo "[done] renders           : ${RENDER_DIR}/${RENDER_SPLIT}/ours_${OUTPUT_ITERATION}/renders"
fi

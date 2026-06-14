#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
BASE_TAG="${BASE_TAG:-k22}"

IMAGE22_MIP_MODEL_PATH="${IMAGE22_MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input_init_repair_v0}"
IMAGE22_RUN_NAME="${IMAGE22_RUN_NAME:-${BASE_TAG}_clean}"
IMAGE22_RUN_ROOT="${IMAGE22_RUN_ROOT:-${SOF_ROOT}/output/cleanup_mip_view_aligned_volume_artifacts_v0/${SCENE_NAME}/${IMAGE22_RUN_NAME}}"
IMAGE22_CLEANED_MODEL="${IMAGE22_CLEANED_MODEL:-${IMAGE22_RUN_ROOT}/cleaned_mip_model_view_volume_v1}"

POST_REFIT_NAME="${POST_REFIT_NAME:-refit}"
POST_REFIT_MODEL="${POST_REFIT_MODEL:-${IMAGE22_RUN_ROOT}/cleaned_mip_model_view_volume_v1_${POST_REFIT_NAME}}"
POST_REFIT_RENDER_DIR="${POST_REFIT_RENDER_DIR:-${IMAGE22_RUN_ROOT}/cleaned_mip_renders_no_gt_view_volume_v1_${POST_REFIT_NAME}}"

MIP_ITERATION="${MIP_ITERATION:-30000}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-${MIP_ITERATION}}"
SOF_REF_MODEL="${SOF_REF_MODEL:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images2_v1/sof30k}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_BASE_CLEANUP="${RUN_BASE_CLEANUP:-0}"
RUN_POST_REFIT_RENDER="${RUN_POST_REFIT_RENDER:-1}"
RUN_LR_RECOVER="${RUN_LR_RECOVER:-1}"
LR_RECOVER_PROFILE="${LR_RECOVER_PROFILE:-mip_hr_anchor_v0}"
LR_RECOVER_RUN_NAME="${LR_RECOVER_RUN_NAME:-${BASE_TAG}_${POST_REFIT_NAME}}"
LR_RECOVER_RUN_RENDER="${LR_RECOVER_RUN_RENDER:-1}"

POST_REFIT_MODE="${POST_REFIT_MODE:-energy_split_replace}"
POST_REFIT_MAX_FRACTION="${POST_REFIT_MAX_FRACTION:-0.10}"
POST_REFIT_MAX_COUNT="${POST_REFIT_MAX_COUNT:-160000}"
POST_REFIT_MIN_OPACITY="${POST_REFIT_MIN_OPACITY:-0.025}"
POST_REFIT_MIN_EFFECTIVE_SCALE_RATIO="${POST_REFIT_MIN_EFFECTIVE_SCALE_RATIO:-0.0018}"
POST_REFIT_MIN_VOLUME_RADIUS_RATIO="${POST_REFIT_MIN_VOLUME_RADIUS_RATIO:-0.0008}"
POST_REFIT_MIN_FILTER_SCALE_RATIO="${POST_REFIT_MIN_FILTER_SCALE_RATIO:-0.35}"
POST_REFIT_MIN_FULL_ANISOTROPY="${POST_REFIT_MIN_FULL_ANISOTROPY:-2.5}"
POST_REFIT_SPLIT_COUNT="${POST_REFIT_SPLIT_COUNT:-8}"
POST_REFIT_CHILD_SCALE_MULTIPLIER="${POST_REFIT_CHILD_SCALE_MULTIPLIER:-0.70}"
POST_REFIT_CHILD_MAJOR_SCALE_MULTIPLIER="${POST_REFIT_CHILD_MAJOR_SCALE_MULTIPLIER:-0.20}"
POST_REFIT_CHILD_OPACITY_SCALE="${POST_REFIT_CHILD_OPACITY_SCALE:-1.0}"
POST_REFIT_ENERGY_CONSERVE_MODE="${POST_REFIT_ENERGY_CONSERVE_MODE:-area}"
POST_REFIT_FILTER_SCALE="${POST_REFIT_FILTER_SCALE:-0.20}"
POST_REFIT_FILTER_CAP_RATIO="${POST_REFIT_FILTER_CAP_RATIO:-0.0008}"
POST_REFIT_OFFSET_SCALE="${POST_REFIT_OFFSET_SCALE:-1.00}"
POST_REFIT_BRIGHT_MAX_FRACTION="${POST_REFIT_BRIGHT_MAX_FRACTION:-0.01}"
POST_REFIT_BRIGHT_MAX_COUNT="${POST_REFIT_BRIGHT_MAX_COUNT:-15000}"
POST_REFIT_BRIGHT_TARGET="${POST_REFIT_BRIGHT_TARGET:-non_children}"
POST_REFIT_BRIGHT_MIN_OPACITY="${POST_REFIT_BRIGHT_MIN_OPACITY:-0.06}"
POST_REFIT_BRIGHT_MAX_EFFECTIVE_SCALE_RATIO="${POST_REFIT_BRIGHT_MAX_EFFECTIVE_SCALE_RATIO:-0.0025}"
POST_REFIT_BRIGHT_LUMA_QUANTILE="${POST_REFIT_BRIGHT_LUMA_QUANTILE:-0.995}"
POST_REFIT_BRIGHT_MIN_LOCAL_LUMA_RATIO="${POST_REFIT_BRIGHT_MIN_LOCAL_LUMA_RATIO:-1.8}"
POST_REFIT_BRIGHT_MIN_COLOR_DELTA="${POST_REFIT_BRIGHT_MIN_COLOR_DELTA:-0.18}"
POST_REFIT_BRIGHT_NEIGHBOR_K="${POST_REFIT_BRIGHT_NEIGHBOR_K:-8}"
POST_REFIT_BRIGHT_EXPAND_SCALE_MULTIPLIER="${POST_REFIT_BRIGHT_EXPAND_SCALE_MULTIPLIER:-1.35}"
POST_REFIT_BRIGHT_SMALLEST_AXIS_SCALE_MULTIPLIER="${POST_REFIT_BRIGHT_SMALLEST_AXIS_SCALE_MULTIPLIER:-1.0}"
POST_REFIT_BRIGHT_OPACITY_SCALE="${POST_REFIT_BRIGHT_OPACITY_SCALE:-0.6}"
POST_REFIT_BRIGHT_DC_SCALE="${POST_REFIT_BRIGHT_DC_SCALE:-0.82}"
POST_REFIT_BRIGHT_REST_SCALE="${POST_REFIT_BRIGHT_REST_SCALE:-0.4}"
POST_REFIT_BRIGHT_FILTER_SCALE="${POST_REFIT_BRIGHT_FILTER_SCALE:-0.5}"
POST_REFIT_BRIGHT_FILTER_CAP_RATIO="${POST_REFIT_BRIGHT_FILTER_CAP_RATIO:-0.001}"

echo "[image22-post-refit-v1] scene          : ${SCENE_ROOT}"
echo "[image22-post-refit-v1] image22 input  : ${IMAGE22_MIP_MODEL_PATH}"
echo "[image22-post-refit-v1] image22 run    : ${IMAGE22_RUN_NAME}"
echo "[image22-post-refit-v1] cleaned model  : ${IMAGE22_CLEANED_MODEL}"
echo "[image22-post-refit-v1] post refit     : ${POST_REFIT_MODEL}"
echo "[image22-post-refit-v1] post params    : mode=${POST_REFIT_MODE} frac=${POST_REFIT_MAX_FRACTION} split=${POST_REFIT_SPLIT_COUNT} anis=${POST_REFIT_MIN_FULL_ANISOTROPY} major=${POST_REFIT_CHILD_MAJOR_SCALE_MULTIPLIER} energy=${POST_REFIT_ENERGY_CONSERVE_MODE}"
if [[ "${POST_REFIT_MODE}" == *"soften_bright"* ]]; then
  echo "[image22-post-refit-v1] bright soften  : target=${POST_REFIT_BRIGHT_TARGET} frac=${POST_REFIT_BRIGHT_MAX_FRACTION} luma_q=${POST_REFIT_BRIGHT_LUMA_QUANTILE} local_ratio=${POST_REFIT_BRIGHT_MIN_LOCAL_LUMA_RATIO} color_delta=${POST_REFIT_BRIGHT_MIN_COLOR_DELTA}"
fi

if [[ "${RUN_BASE_CLEANUP}" == "1" ]]; then
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP}" \
  MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME}" \
  MIP_MODEL_PATH="${IMAGE22_MIP_MODEL_PATH}" \
  RUN_NAME="${IMAGE22_RUN_NAME}" \
  RUN_RENDER="${RUN_POST_REFIT_RENDER}" \
  RUN_LR_RECOVER=0 \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/run_cleanup_mip_view_aligned_volume_artifacts_initrepair_v0_prune_more_v1_kitchen.sh"
fi

if [[ ! -e "${IMAGE22_CLEANED_MODEL}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" ]]; then
  echo "[image22-post-refit-v1] missing image22 cleaned model: ${IMAGE22_CLEANED_MODEL}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" >&2
  exit 1
fi

"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/prepare_mipsplatting_sof_input_field.py" \
  --mip_model_path "${IMAGE22_CLEANED_MODEL}" \
  --scene_root "${SCENE_ROOT}" \
  --sof_ref_model "${SOF_REF_MODEL}" \
  --output_model_path "${POST_REFIT_MODEL}" \
  --iteration "${MIP_ITERATION}" \
  --output_iteration "${OUTPUT_ITERATION}" \
  --opacity_compensate_scale_shrink area \
  --init_repair_mode "${POST_REFIT_MODE}" \
  --init_repair_max_fraction "${POST_REFIT_MAX_FRACTION}" \
  --init_repair_max_count "${POST_REFIT_MAX_COUNT}" \
  --init_repair_min_opacity "${POST_REFIT_MIN_OPACITY}" \
  --init_repair_min_effective_scale_ratio "${POST_REFIT_MIN_EFFECTIVE_SCALE_RATIO}" \
  --init_repair_min_volume_radius_ratio "${POST_REFIT_MIN_VOLUME_RADIUS_RATIO}" \
  --init_repair_min_filter_scale_ratio "${POST_REFIT_MIN_FILTER_SCALE_RATIO}" \
  --init_repair_min_full_anisotropy "${POST_REFIT_MIN_FULL_ANISOTROPY}" \
  --init_repair_split_count "${POST_REFIT_SPLIT_COUNT}" \
  --init_repair_child_layout major_axis \
  --init_repair_child_scale_multiplier "${POST_REFIT_CHILD_SCALE_MULTIPLIER}" \
  --init_repair_child_major_scale_multiplier "${POST_REFIT_CHILD_MAJOR_SCALE_MULTIPLIER}" \
  --init_repair_child_opacity_scale "${POST_REFIT_CHILD_OPACITY_SCALE}" \
  --init_repair_energy_conserve_mode "${POST_REFIT_ENERGY_CONSERVE_MODE}" \
  --init_repair_filter_scale "${POST_REFIT_FILTER_SCALE}" \
  --init_repair_filter_cap_ratio "${POST_REFIT_FILTER_CAP_RATIO}" \
  --init_repair_offset_scale "${POST_REFIT_OFFSET_SCALE}" \
  --init_repair_bright_max_fraction "${POST_REFIT_BRIGHT_MAX_FRACTION}" \
  --init_repair_bright_max_count "${POST_REFIT_BRIGHT_MAX_COUNT}" \
  --init_repair_bright_target "${POST_REFIT_BRIGHT_TARGET}" \
  --init_repair_bright_min_opacity "${POST_REFIT_BRIGHT_MIN_OPACITY}" \
  --init_repair_bright_max_effective_scale_ratio "${POST_REFIT_BRIGHT_MAX_EFFECTIVE_SCALE_RATIO}" \
  --init_repair_bright_luma_quantile "${POST_REFIT_BRIGHT_LUMA_QUANTILE}" \
  --init_repair_bright_min_local_luma_ratio "${POST_REFIT_BRIGHT_MIN_LOCAL_LUMA_RATIO}" \
  --init_repair_bright_min_color_delta "${POST_REFIT_BRIGHT_MIN_COLOR_DELTA}" \
  --init_repair_bright_neighbor_k "${POST_REFIT_BRIGHT_NEIGHBOR_K}" \
  --init_repair_bright_expand_scale_multiplier "${POST_REFIT_BRIGHT_EXPAND_SCALE_MULTIPLIER}" \
  --init_repair_bright_smallest_axis_scale_multiplier "${POST_REFIT_BRIGHT_SMALLEST_AXIS_SCALE_MULTIPLIER}" \
  --init_repair_bright_opacity_scale "${POST_REFIT_BRIGHT_OPACITY_SCALE}" \
  --init_repair_bright_dc_scale "${POST_REFIT_BRIGHT_DC_SCALE}" \
  --init_repair_bright_rest_scale "${POST_REFIT_BRIGHT_REST_SCALE}" \
  --init_repair_bright_filter_scale "${POST_REFIT_BRIGHT_FILTER_SCALE}" \
  --init_repair_bright_filter_cap_ratio "${POST_REFIT_BRIGHT_FILTER_CAP_RATIO}" \
  --use_aabb_filter

if [[ "${RUN_POST_REFIT_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${POST_REFIT_MODEL}" \
    --output_dir "${POST_REFIT_RENDER_DIR}" \
    --images_subdir images_2 \
    --iteration "${OUTPUT_ITERATION}" \
    --split test
fi

if [[ "${RUN_LR_RECOVER}" == "1" ]]; then
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  START_MODEL_PATH="${POST_REFIT_MODEL}" \
  START_ITERATION="${OUTPUT_ITERATION}" \
  ANCHOR_MODEL_PATH="${IMAGE22_MIP_MODEL_PATH}" \
  ANCHOR_ITERATION="${MIP_ITERATION}" \
  REFINE_PROFILE="${LR_RECOVER_PROFILE}" \
  RUN_NAME="${LR_RECOVER_RUN_NAME}" \
  RUN_RENDER="${LR_RECOVER_RUN_RENDER}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SOF_ROOT}/scripts/run_recover_cleaned_mip_lr_v0_kitchen.sh"
fi

echo "[done] image22 cleaned model : ${IMAGE22_CLEANED_MODEL}"
echo "[done] post refit model      : ${POST_REFIT_MODEL}"
echo "[done] post refit ply        : ${POST_REFIT_MODEL}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
echo "[done] post refit summary    : ${POST_REFIT_MODEL}/mip_input_field_summary.json"
if [[ "${RUN_POST_REFIT_RENDER}" == "1" ]]; then
  echo "[done] post refit renders    : ${POST_REFIT_RENDER_DIR}/test/ours_${OUTPUT_ITERATION}/renders"
fi
if [[ "${RUN_LR_RECOVER}" == "1" ]]; then
  echo "[done] lr recover            : ${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${LR_RECOVER_RUN_NAME}"
fi

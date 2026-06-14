#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
ANCHOR_MODEL_PATH="${ANCHOR_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input}"
ANCHOR_ITERATION="${ANCHOR_ITERATION:-30000}"

START_RUN_NAME="${START_RUN_NAME:-view_aligned_volume_delete_v1}"
START_MODEL_PATH="${START_MODEL_PATH:-${SOF_ROOT}/output/cleanup_mip_view_aligned_volume_artifacts_v0/${SCENE_NAME}/${START_RUN_NAME}/cleaned_mip_model_view_volume_v1}"
START_ITERATION="${START_ITERATION:-30000}"

REFINE_PROFILE="${REFINE_PROFILE:-mip_hr_anchor_v0}"
case "${REFINE_PROFILE}" in
  mip_hr_anchor_v0)
    DEFAULT_OUTPUT_SUFFIX="mip_hr_anchor_v0"
    DEFAULT_ITERATIONS="1600"
    DEFAULT_MAX_VIEWS="20"
    DEFAULT_XYZ_LR="6e-6"
    DEFAULT_OPACITY_LR="7e-4"
    DEFAULT_SCALE_LR="1.5e-4"
    DEFAULT_LAMBDA_LR_RGB="1.0"
    DEFAULT_LAMBDA_ANCHOR_RGB="0.0"
    DEFAULT_LAMBDA_XYZ_ANCHOR="38.0"
    DEFAULT_LAMBDA_OPACITY_ANCHOR="0.04"
    DEFAULT_LAMBDA_SCALE_ANCHOR="0.14"
    DEFAULT_LAMBDA_RISK_OPACITY="0.055"
    DEFAULT_LAMBDA_RISK_SCALE="0.010"
    DEFAULT_LAMBDA_SR_RISK_OPACITY="0.030"
    DEFAULT_LAMBDA_SR_RISK_SCALE="0.006"
    DEFAULT_SR_RISK_BOOST="1.60"
    DEFAULT_SR_PRIOR_L1_WEIGHT="0.05"
    DEFAULT_SR_PRIOR_HF_WEIGHT="0.28"
    DEFAULT_SR_RESIDUAL_ANCHOR="mip_hr"
    DEFAULT_LAMBDA_MIP_HR_LOWFREQ="0.45"
    DEFAULT_MIP_HR_LOWFREQ_KERNEL="25"
    DEFAULT_MIP_HR_LOWFREQ_CONSISTENCY_THRESHOLD="0.24"
    DEFAULT_MIP_HR_LOWFREQ_MASK_FLOOR="0.0"
    DEFAULT_SR_OUTPUT_TAG="miphr_v1"
    DEFAULT_RISK_RADIUS_THRESHOLD="42"
    DEFAULT_RISK_LR_RESIDUAL_THRESHOLD="0.08"
    DEFAULT_RISK_ANISOTROPY_THRESHOLD="18"
    DEFAULT_RISK_MIN_SUPPORT_VIEWS="3"
    DEFAULT_MAX_DISPLACEMENT_RATIO="0.0018"
    ;;
  prior_recovery_v1)
    DEFAULT_OUTPUT_SUFFIX="prior_recovery_v1"
    DEFAULT_ITERATIONS="1400"
    DEFAULT_MAX_VIEWS="20"
    DEFAULT_XYZ_LR="6e-6"
    DEFAULT_OPACITY_LR="7e-4"
    DEFAULT_SCALE_LR="1.5e-4"
    DEFAULT_LAMBDA_LR_RGB="1.0"
    DEFAULT_LAMBDA_ANCHOR_RGB="0.0"
    DEFAULT_LAMBDA_XYZ_ANCHOR="38.0"
    DEFAULT_LAMBDA_OPACITY_ANCHOR="0.04"
    DEFAULT_LAMBDA_SCALE_ANCHOR="0.14"
    DEFAULT_LAMBDA_RISK_OPACITY="0.055"
    DEFAULT_LAMBDA_RISK_SCALE="0.010"
    DEFAULT_LAMBDA_SR_RISK_OPACITY="0.030"
    DEFAULT_LAMBDA_SR_RISK_SCALE="0.006"
    DEFAULT_SR_RISK_BOOST="1.60"
    DEFAULT_SR_PRIOR_L1_WEIGHT="0.06"
    DEFAULT_SR_PRIOR_HF_WEIGHT="0.36"
    DEFAULT_SR_RESIDUAL_ANCHOR="prepared"
    DEFAULT_LAMBDA_MIP_HR_LOWFREQ="0.0"
    DEFAULT_MIP_HR_LOWFREQ_KERNEL="17"
    DEFAULT_MIP_HR_LOWFREQ_CONSISTENCY_THRESHOLD="0.16"
    DEFAULT_MIP_HR_LOWFREQ_MASK_FLOOR="0.0"
    DEFAULT_SR_OUTPUT_TAG="2dsr_v1"
    DEFAULT_RISK_RADIUS_THRESHOLD="42"
    DEFAULT_RISK_LR_RESIDUAL_THRESHOLD="0.08"
    DEFAULT_RISK_ANISOTROPY_THRESHOLD="18"
    DEFAULT_RISK_MIN_SUPPORT_VIEWS="3"
    DEFAULT_MAX_DISPLACEMENT_RATIO="0.0018"
    ;;
  conservative_v0)
    DEFAULT_OUTPUT_SUFFIX="conservative_v0"
    DEFAULT_ITERATIONS="800"
    DEFAULT_MAX_VIEWS="16"
    DEFAULT_XYZ_LR="5e-6"
    DEFAULT_OPACITY_LR="5e-4"
    DEFAULT_SCALE_LR="1e-4"
    DEFAULT_LAMBDA_LR_RGB="1.0"
    DEFAULT_LAMBDA_ANCHOR_RGB="0.0"
    DEFAULT_LAMBDA_XYZ_ANCHOR="45.0"
    DEFAULT_LAMBDA_OPACITY_ANCHOR="0.05"
    DEFAULT_LAMBDA_SCALE_ANCHOR="0.18"
    DEFAULT_LAMBDA_RISK_OPACITY="0.035"
    DEFAULT_LAMBDA_RISK_SCALE="0.005"
    DEFAULT_LAMBDA_SR_RISK_OPACITY="0.0"
    DEFAULT_LAMBDA_SR_RISK_SCALE="0.0"
    DEFAULT_SR_RISK_BOOST="1.00"
    DEFAULT_SR_PRIOR_L1_WEIGHT="0.03"
    DEFAULT_SR_PRIOR_HF_WEIGHT="0.18"
    DEFAULT_SR_RESIDUAL_ANCHOR="prepared"
    DEFAULT_LAMBDA_MIP_HR_LOWFREQ="0.0"
    DEFAULT_MIP_HR_LOWFREQ_KERNEL="17"
    DEFAULT_MIP_HR_LOWFREQ_CONSISTENCY_THRESHOLD="0.16"
    DEFAULT_MIP_HR_LOWFREQ_MASK_FLOOR="0.0"
    DEFAULT_SR_OUTPUT_TAG="2dsr_v1"
    DEFAULT_RISK_RADIUS_THRESHOLD="48"
    DEFAULT_RISK_LR_RESIDUAL_THRESHOLD="0.10"
    DEFAULT_RISK_ANISOTROPY_THRESHOLD="20"
    DEFAULT_RISK_MIN_SUPPORT_VIEWS="4"
    DEFAULT_MAX_DISPLACEMENT_RATIO="0.0015"
    ;;
  strong_v0)
    DEFAULT_OUTPUT_SUFFIX="strong_v0"
    DEFAULT_ITERATIONS="1500"
    DEFAULT_MAX_VIEWS="16"
    DEFAULT_XYZ_LR="8e-6"
    DEFAULT_OPACITY_LR="8e-4"
    DEFAULT_SCALE_LR="2e-4"
    DEFAULT_LAMBDA_LR_RGB="1.0"
    DEFAULT_LAMBDA_ANCHOR_RGB="0.0"
    DEFAULT_LAMBDA_XYZ_ANCHOR="30.0"
    DEFAULT_LAMBDA_OPACITY_ANCHOR="0.035"
    DEFAULT_LAMBDA_SCALE_ANCHOR="0.12"
    DEFAULT_LAMBDA_RISK_OPACITY="0.030"
    DEFAULT_LAMBDA_RISK_SCALE="0.004"
    DEFAULT_LAMBDA_SR_RISK_OPACITY="0.0"
    DEFAULT_LAMBDA_SR_RISK_SCALE="0.0"
    DEFAULT_SR_RISK_BOOST="1.00"
    DEFAULT_SR_PRIOR_L1_WEIGHT="0.05"
    DEFAULT_SR_PRIOR_HF_WEIGHT="0.30"
    DEFAULT_SR_RESIDUAL_ANCHOR="prepared"
    DEFAULT_LAMBDA_MIP_HR_LOWFREQ="0.0"
    DEFAULT_MIP_HR_LOWFREQ_KERNEL="17"
    DEFAULT_MIP_HR_LOWFREQ_CONSISTENCY_THRESHOLD="0.16"
    DEFAULT_MIP_HR_LOWFREQ_MASK_FLOOR="0.0"
    DEFAULT_SR_OUTPUT_TAG="2dsr_v1"
    DEFAULT_RISK_RADIUS_THRESHOLD="48"
    DEFAULT_RISK_LR_RESIDUAL_THRESHOLD="0.10"
    DEFAULT_RISK_ANISOTROPY_THRESHOLD="20"
    DEFAULT_RISK_MIN_SUPPORT_VIEWS="4"
    DEFAULT_MAX_DISPLACEMENT_RATIO="0.0020"
    ;;
  *)
    echo "[recover-cleaned-mip-lr-v0] unknown REFINE_PROFILE: ${REFINE_PROFILE}" >&2
    exit 1
    ;;
esac

USE_SR_PRIOR="${USE_SR_PRIOR:-1}"
SR_OUTPUT_TAG="${SR_OUTPUT_TAG:-${DEFAULT_SR_OUTPUT_TAG}}"
DEFAULT_OUTPUT_MODEL_NAME="recovered_mip_model_lr_v0"
if [[ "${USE_SR_PRIOR}" == "1" ]]; then
  DEFAULT_OUTPUT_SUFFIX="${DEFAULT_OUTPUT_SUFFIX}_${SR_OUTPUT_TAG}"
  DEFAULT_OUTPUT_MODEL_NAME="recovered_mip_model_lr_${SR_OUTPUT_TAG}"
fi

RUN_NAME="${RUN_NAME:-${START_RUN_NAME}_${DEFAULT_OUTPUT_SUFFIX}}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${RUN_NAME}}"
OUTPUT_MODEL="${OUTPUT_MODEL:-${RUN_ROOT}/${DEFAULT_OUTPUT_MODEL_NAME}}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
MAX_VIEWS="${MAX_VIEWS:-${DEFAULT_MAX_VIEWS}}"
ITERATIONS="${ITERATIONS:-${DEFAULT_ITERATIONS}}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-$((START_ITERATION + ITERATIONS))}"
XYZ_LR="${XYZ_LR:-${DEFAULT_XYZ_LR}}"
OPACITY_LR="${OPACITY_LR:-${DEFAULT_OPACITY_LR}}"
SCALE_LR="${SCALE_LR:-${DEFAULT_SCALE_LR}}"
DC_LR="${DC_LR:-0.0}"
REST_LR="${REST_LR:-0.0}"
LAMBDA_LR_RGB="${LAMBDA_LR_RGB:-${DEFAULT_LAMBDA_LR_RGB}}"
LAMBDA_ANCHOR_RGB="${LAMBDA_ANCHOR_RGB:-${DEFAULT_LAMBDA_ANCHOR_RGB}}"
LAMBDA_XYZ_ANCHOR="${LAMBDA_XYZ_ANCHOR:-${DEFAULT_LAMBDA_XYZ_ANCHOR}}"
LAMBDA_OPACITY_ANCHOR="${LAMBDA_OPACITY_ANCHOR:-${DEFAULT_LAMBDA_OPACITY_ANCHOR}}"
LAMBDA_SCALE_ANCHOR="${LAMBDA_SCALE_ANCHOR:-${DEFAULT_LAMBDA_SCALE_ANCHOR}}"
LAMBDA_DC_ANCHOR="${LAMBDA_DC_ANCHOR:-0.0}"
LAMBDA_REST_ANCHOR="${LAMBDA_REST_ANCHOR:-0.0}"
LAMBDA_RISK_OPACITY="${LAMBDA_RISK_OPACITY:-${DEFAULT_LAMBDA_RISK_OPACITY}}"
LAMBDA_RISK_SCALE="${LAMBDA_RISK_SCALE:-${DEFAULT_LAMBDA_RISK_SCALE}}"
LAMBDA_RISK_REST="${LAMBDA_RISK_REST:-0.0}"
LAMBDA_SR_RISK_OPACITY="${LAMBDA_SR_RISK_OPACITY:-${DEFAULT_LAMBDA_SR_RISK_OPACITY}}"
LAMBDA_SR_RISK_SCALE="${LAMBDA_SR_RISK_SCALE:-${DEFAULT_LAMBDA_SR_RISK_SCALE}}"
LAMBDA_SR_RISK_REST="${LAMBDA_SR_RISK_REST:-0.0}"
SR_RISK_BOOST="${SR_RISK_BOOST:-${DEFAULT_SR_RISK_BOOST}}"
RISK_RADIUS_THRESHOLD="${RISK_RADIUS_THRESHOLD:-${DEFAULT_RISK_RADIUS_THRESHOLD}}"
RISK_LR_RESIDUAL_THRESHOLD="${RISK_LR_RESIDUAL_THRESHOLD:-${DEFAULT_RISK_LR_RESIDUAL_THRESHOLD}}"
RISK_ANISOTROPY_THRESHOLD="${RISK_ANISOTROPY_THRESHOLD:-${DEFAULT_RISK_ANISOTROPY_THRESHOLD}}"
RISK_MIN_SUPPORT_VIEWS="${RISK_MIN_SUPPORT_VIEWS:-${DEFAULT_RISK_MIN_SUPPORT_VIEWS}}"
MAX_DISPLACEMENT_RATIO="${MAX_DISPLACEMENT_RATIO:-${DEFAULT_MAX_DISPLACEMENT_RATIO}}"
MAX_DISPLACEMENT_ABS="${MAX_DISPLACEMENT_ABS:-0.0}"
ENABLE_OPACITY_UPDATE="${ENABLE_OPACITY_UPDATE:-1}"
ENABLE_SCALE_UPDATE="${ENABLE_SCALE_UPDATE:-1}"
ENABLE_DC_UPDATE="${ENABLE_DC_UPDATE:-0}"
ENABLE_REST_UPDATE="${ENABLE_REST_UPDATE:-0}"
PHASE_MODE="${PHASE_MODE:-joint}"
PHASE_BLOCK_STEPS="${PHASE_BLOCK_STEPS:-0}"
OPTIMIZE_GAUSSIAN_MASK_PAYLOAD="${OPTIMIZE_GAUSSIAN_MASK_PAYLOAD:-}"
OPTIMIZE_GAUSSIAN_MASK_KEY="${OPTIMIZE_GAUSSIAN_MASK_KEY:-selected_mask}"
GAUSSIAN_UPDATE_SCALE="${GAUSSIAN_UPDATE_SCALE:-1.0}"
GAUSSIAN_SCALE_AXIS_MODE="${GAUSSIAN_SCALE_AXIS_MODE:-all}"
PRIOR_PREFILTER_MODE="${PRIOR_PREFILTER_MODE:-none}"
PRIOR_PREFILTER_MASK_PAYLOAD="${PRIOR_PREFILTER_MASK_PAYLOAD:-}"
PRIOR_PREFILTER_MASK_KEY="${PRIOR_PREFILTER_MASK_KEY:-selected_mask}"
PRIOR_PREFILTER_VIEW_LIMIT="${PRIOR_PREFILTER_VIEW_LIMIT:-0}"
PRIOR_PREFILTER_MIN_TOUCH_VIEWS="${PRIOR_PREFILTER_MIN_TOUCH_VIEWS:-1}"
PRIOR_PREFILTER_MIN_VISIBLE_VIEWS="${PRIOR_PREFILTER_MIN_VISIBLE_VIEWS:-1}"
PRIOR_PREFILTER_MIN_TOUCH_RATIO="${PRIOR_PREFILTER_MIN_TOUCH_RATIO:-0.0}"
PRIOR_PREFILTER_MIN_CANDIDATE_OPACITY="${PRIOR_PREFILTER_MIN_CANDIDATE_OPACITY:-0.0}"
PRIOR_PREFILTER_RADIUS_SCALE="${PRIOR_PREFILTER_RADIUS_SCALE:-0.5}"
PRIOR_PREFILTER_MIN_TOUCH_RADIUS_PX="${PRIOR_PREFILTER_MIN_TOUCH_RADIUS_PX:-2.0}"
PRIOR_PREFILTER_MAX_TOUCH_RADIUS_PX="${PRIOR_PREFILTER_MAX_TOUCH_RADIUS_PX:-16.0}"
PRIOR_PREFILTER_SAVE_PATH="${PRIOR_PREFILTER_SAVE_PATH:-${RUN_ROOT}/prior_prefilter_selected_mask.pt}"
RECOMPUTE_FILTER3D="${RECOMPUTE_FILTER3D:-1}"
SAVE_EVERY="${SAVE_EVERY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAR_QUARANTINE_PAYLOAD_PATH="${STAR_QUARANTINE_PAYLOAD_PATH:-}"
STAR_RELEASE_START_ITER="${STAR_RELEASE_START_ITER:-0}"
STAR_RELEASE_END_ITER="${STAR_RELEASE_END_ITER:-0}"
STAR_RELEASE_MODE="${STAR_RELEASE_MODE:-smoothstep}"
STAR_RELEASE_REST_START_SCALE="${STAR_RELEASE_REST_START_SCALE:--1.0}"
STAR_RELEASE_REST_END_SCALE="${STAR_RELEASE_REST_END_SCALE:-1.0}"
STAR_RELEASE_TAU_START_SCALE="${STAR_RELEASE_TAU_START_SCALE:--1.0}"
STAR_RELEASE_TAU_END_SCALE="${STAR_RELEASE_TAU_END_SCALE:-1.0}"
STAR_RELEASE_MIN_ALPHA="${STAR_RELEASE_MIN_ALPHA:-1e-6}"
STAR_RELEASE_OPACITY="${STAR_RELEASE_OPACITY:-0}"
REPARAM_OUTPUT_SOURCE_IDX_PATH="${REPARAM_OUTPUT_SOURCE_IDX_PATH:-}"
REPARAM_PARENT_MASK_PATH="${REPARAM_PARENT_MASK_PATH:-}"
REPARAM_CHILD_MASK_PATH="${REPARAM_CHILD_MASK_PATH:-}"
GEOMETRY_PARENT_MASK_PATH="${GEOMETRY_PARENT_MASK_PATH:-}"
LAMBDA_REPARAM_MASS_CAP="${LAMBDA_REPARAM_MASS_CAP:-0.0}"
REPARAM_MASS_CAP_EPS="${REPARAM_MASS_CAP_EPS:-0.10}"
LAMBDA_REPARAM_CHILD_TAU_CAP="${LAMBDA_REPARAM_CHILD_TAU_CAP:-0.0}"
REPARAM_CHILD_TAU_CAP_SCALE="${REPARAM_CHILD_TAU_CAP_SCALE:-2.5}"
REPARAM_CHILD_TAU_CAP_ABS="${REPARAM_CHILD_TAU_CAP_ABS:-0.03}"
LAMBDA_GEOMETRY_PARENT_TAU_BRAKE="${LAMBDA_GEOMETRY_PARENT_TAU_BRAKE:-0.0}"
GEOMETRY_PARENT_TAU_SCALE="${GEOMETRY_PARENT_TAU_SCALE:-1.0}"
REPARAM_PRUNE_ITERS="${REPARAM_PRUNE_ITERS:-}"
REPARAM_PRUNE_CHILD_DEAD_TAU="${REPARAM_PRUNE_CHILD_DEAD_TAU:-0.0}"
REPARAM_PRUNE_CHILD_SPIKE_TAU_SCALE="${REPARAM_PRUNE_CHILD_SPIKE_TAU_SCALE:-0.0}"
REPARAM_PRUNE_CHILD_SPIKE_TAU_ABS="${REPARAM_PRUNE_CHILD_SPIKE_TAU_ABS:-0.0}"
REPARAM_PRUNE_CHILD_SPIKE_ANISOTROPY="${REPARAM_PRUNE_CHILD_SPIKE_ANISOTROPY:-0.0}"
REPARAM_PRUNE_CHILD_SPIKE_RISK_MIN="${REPARAM_PRUNE_CHILD_SPIKE_RISK_MIN:-0.0}"

SR_IMAGES_SUBDIR="${SR_IMAGES_SUBDIR:-images_2}"
SR_MASK_THRESHOLD="${SR_MASK_THRESHOLD:-0.12}"
SR_MASK_MODE="${SR_MASK_MODE:-soft}"
SR_PRIOR_ROOT="${SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/sof_surface_v0_${IMAGES_SUBDIR}_to_${SR_IMAGES_SUBDIR}_mask${SR_MASK_THRESHOLD}_${SR_MASK_MODE}}"
SR_PRIOR_SUBDIR="${SR_PRIOR_SUBDIR:-fused_priors}"
SR_PRIOR_MASK_SUBDIR="${SR_PRIOR_MASK_SUBDIR:-usable_masks}"
SR_ANCHOR_SUBDIR="${SR_ANCHOR_SUBDIR:-aligned_references}"
SR_PRIOR_MASK_SUFFIX="${SR_PRIOR_MASK_SUFFIX:-}"
SR_MAX_VIEWS="${SR_MAX_VIEWS:-${MAX_VIEWS}}"
SR_VIEW_MODE="${SR_VIEW_MODE:-selected_lr}"
SR_PRIOR_L1_WEIGHT="${SR_PRIOR_L1_WEIGHT:-${DEFAULT_SR_PRIOR_L1_WEIGHT}}"
SR_PRIOR_HF_WEIGHT="${SR_PRIOR_HF_WEIGHT:-${DEFAULT_SR_PRIOR_HF_WEIGHT}}"
SR_PRIOR_WARMUP_START_ITER="${SR_PRIOR_WARMUP_START_ITER:-0}"
SR_PRIOR_WARMUP_END_ITER="${SR_PRIOR_WARMUP_END_ITER:-0}"
SR_PRIOR_START_SCALE="${SR_PRIOR_START_SCALE:-1.0}"
SR_PRIOR_END_SCALE="${SR_PRIOR_END_SCALE:-1.0}"
SR_PRIOR_UPDATE_SCALE="${SR_PRIOR_UPDATE_SCALE:-1.0}"
SR_PRIOR_SCHEDULE_MODE="${SR_PRIOR_SCHEDULE_MODE:-smoothstep}"
LAMBDA_PREMUL_HF_EXCESS="${LAMBDA_PREMUL_HF_EXCESS:-0.0}"
PREMUL_HF_EXCESS_KERNEL="${PREMUL_HF_EXCESS_KERNEL:-9}"
PREMUL_HF_EXCESS_RATIO="${PREMUL_HF_EXCESS_RATIO:-1.25}"
PREMUL_HF_EXCESS_MARGIN="${PREMUL_HF_EXCESS_MARGIN:-0.01}"
SR_RESIDUAL_ANCHOR="${SR_RESIDUAL_ANCHOR:-${DEFAULT_SR_RESIDUAL_ANCHOR}}"
LAMBDA_MIP_HR_LOWFREQ="${LAMBDA_MIP_HR_LOWFREQ:-${DEFAULT_LAMBDA_MIP_HR_LOWFREQ}}"
MIP_HR_LOWFREQ_KERNEL="${MIP_HR_LOWFREQ_KERNEL:-${DEFAULT_MIP_HR_LOWFREQ_KERNEL}}"
MIP_HR_LOWFREQ_CONSISTENCY_THRESHOLD="${MIP_HR_LOWFREQ_CONSISTENCY_THRESHOLD:-${DEFAULT_MIP_HR_LOWFREQ_CONSISTENCY_THRESHOLD}}"
MIP_HR_LOWFREQ_MASK_FLOOR="${MIP_HR_LOWFREQ_MASK_FLOOR:-${DEFAULT_MIP_HR_LOWFREQ_MASK_FLOOR}}"
MIP_CLOSURE_MODEL_PATH="${MIP_CLOSURE_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input_init_repair_v0}"
MIP_CLOSURE_ITERATION="${MIP_CLOSURE_ITERATION:-30000}"
MIP_CLOSURE_IMAGES_SUBDIR="${MIP_CLOSURE_IMAGES_SUBDIR:-${SR_IMAGES_SUBDIR}}"
MIP_CLOSURE_MAX_VIEWS="${MIP_CLOSURE_MAX_VIEWS:-${SR_MAX_VIEWS}}"
LAMBDA_MIP_CLOSURE_ALPHA="${LAMBDA_MIP_CLOSURE_ALPHA:-0.20}"
LAMBDA_MIP_CLOSURE_PREMUL="${LAMBDA_MIP_CLOSURE_PREMUL:-0.60}"
LAMBDA_MIP_CLOSURE_DEPTH="${LAMBDA_MIP_CLOSURE_DEPTH:-0.03}"
LAMBDA_MIP_CLOSURE_ALPHA_OVER="${LAMBDA_MIP_CLOSURE_ALPHA_OVER:-0.20}"
LAMBDA_MIP_CLOSURE_PREMUL_OVER="${LAMBDA_MIP_CLOSURE_PREMUL_OVER:-1.20}"
MIP_CLOSURE_KERNEL="${MIP_CLOSURE_KERNEL:-25}"
MIP_CLOSURE_ALPHA_THRESHOLD="${MIP_CLOSURE_ALPHA_THRESHOLD:-0.05}"
MIP_CLOSURE_REFERENCE_LOWPASS="${MIP_CLOSURE_REFERENCE_LOWPASS:-1}"
MIP_CLOSURE_MIN_PIXELS="${MIP_CLOSURE_MIN_PIXELS:-256}"
MIP_CLOSURE_DEPTH_RELATIVE_MIN="${MIP_CLOSURE_DEPTH_RELATIVE_MIN:-0.5}"
SR_PRIOR_MASK_FLOOR="${SR_PRIOR_MASK_FLOOR:-0.0}"
SR_PRIOR_CONSISTENCY_THRESHOLD="${SR_PRIOR_CONSISTENCY_THRESHOLD:-0.12}"
SR_PRIOR_MIN_VALID_RATIO="${SR_PRIOR_MIN_VALID_RATIO:-0.50}"
SR_PRIOR_MIN_PIXELS="${SR_PRIOR_MIN_PIXELS:-64}"
SR_PRIOR_DELTA_CLIP="${SR_PRIOR_DELTA_CLIP:-0.15}"
DISABLE_SR_PRIOR_HF_RESIDUAL="${DISABLE_SR_PRIOR_HF_RESIDUAL:-0}"
if [[ "${USE_SR_PRIOR}" != "1" ]]; then
  SR_PRIOR_ROOT=""
  SR_PRIOR_L1_WEIGHT="0.0"
  SR_PRIOR_HF_WEIGHT="0.0"
  LAMBDA_PREMUL_HF_EXCESS="0.0"
  SR_RESIDUAL_ANCHOR="prepared"
  LAMBDA_MIP_HR_LOWFREQ="0.0"
  LAMBDA_SR_RISK_OPACITY="0.0"
  LAMBDA_SR_RISK_SCALE="0.0"
  SR_RISK_BOOST="1.0"
fi
RECOMPUTE_FILTER3D_BEFORE_TRAIN="${RECOMPUTE_FILTER3D_BEFORE_TRAIN:-${USE_SR_PRIOR}}"

RUN_RENDER="${RUN_RENDER:-1}"
RENDER_IMAGES_SUBDIR="${RENDER_IMAGES_SUBDIR:-images_2}"
RENDER_SPLIT="${RENDER_SPLIT:-test}"
RENDER_DIR="${RENDER_DIR:-${RUN_ROOT}/recovered_mip_renders_no_gt_hr_v0}"
RENDER_CHECKPOINTS="${RENDER_CHECKPOINTS:-0}"
RENDER_CHECKPOINT_MAX_VIEWS="${RENDER_CHECKPOINT_MAX_VIEWS:-0}"
RENDER_PREVIEW="${RENDER_PREVIEW:-1}"
RENDER_PREVIEW_MAX_IMAGES="${RENDER_PREVIEW_MAX_IMAGES:-16}"
RENDER_PREVIEW_COLUMNS="${RENDER_PREVIEW_COLUMNS:-4}"
RENDER_PREVIEW_THUMB_WIDTH="${RENDER_PREVIEW_THUMB_WIDTH:-360}"
RENDER_PREVIEW_PATH="${RENDER_PREVIEW_PATH:-${RENDER_DIR}/contact_sheet_${RENDER_IMAGES_SUBDIR}_${RENDER_SPLIT}_${OUTPUT_ITERATION}.png}"

has_point_cloud_iteration() {
  local model_root="$1"
  local iteration="$2"
  if [[ "${iteration}" =~ ^- ]]; then
    if [[ ! -d "${model_root}/point_cloud" ]]; then
      return 1
    fi
    find "${model_root}/point_cloud" -mindepth 2 -maxdepth 2 -path '*/point_cloud.ply' -print -quit | grep -q .
    return $?
  fi
  [[ -e "${model_root}/point_cloud/iteration_${iteration}/point_cloud.ply" ]]
}

resolve_point_cloud_iteration() {
  local model_root="$1"
  local iteration="$2"
  if [[ ! "${iteration}" =~ ^- ]]; then
    printf '%s\n' "${iteration}"
    return 0
  fi
  if [[ ! -d "${model_root}/point_cloud" ]]; then
    return 1
  fi
  local latest_dir
  latest_dir="$(
    find "${model_root}/point_cloud" -mindepth 1 -maxdepth 1 -type d -name 'iteration_*' \
      | sort -V \
      | tail -n 1
  )"
  if [[ -z "${latest_dir}" ]]; then
    return 1
  fi
  basename "${latest_dir}" | sed 's/^iteration_//'
}

if ! has_point_cloud_iteration "${START_MODEL_PATH}" "${START_ITERATION}"; then
  if [[ "${START_ITERATION}" =~ ^- ]]; then
    echo "[recover-cleaned-mip-lr-v0] missing start model: ${START_MODEL_PATH}/point_cloud/iteration_*/point_cloud.ply" >&2
  else
    echo "[recover-cleaned-mip-lr-v0] missing start model: ${START_MODEL_PATH}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply" >&2
  fi
  exit 1
fi

if [[ -n "${ANCHOR_MODEL_PATH}" ]] && ! has_point_cloud_iteration "${ANCHOR_MODEL_PATH}" "${ANCHOR_ITERATION}"; then
  if [[ "${ANCHOR_ITERATION}" =~ ^- ]]; then
    echo "[recover-cleaned-mip-lr-v0] missing anchor model: ${ANCHOR_MODEL_PATH}/point_cloud/iteration_*/point_cloud.ply" >&2
  else
    echo "[recover-cleaned-mip-lr-v0] missing anchor model: ${ANCHOR_MODEL_PATH}/point_cloud/iteration_${ANCHOR_ITERATION}/point_cloud.ply" >&2
  fi
  exit 1
fi

if [[ -n "${MIP_CLOSURE_MODEL_PATH}" ]] && ! has_point_cloud_iteration "${MIP_CLOSURE_MODEL_PATH}" "${MIP_CLOSURE_ITERATION}"; then
  if [[ "${MIP_CLOSURE_ITERATION}" =~ ^- ]]; then
    echo "[recover-cleaned-mip-lr-v0] missing mip closure model: ${MIP_CLOSURE_MODEL_PATH}/point_cloud/iteration_*/point_cloud.ply" >&2
  else
    echo "[recover-cleaned-mip-lr-v0] missing mip closure model: ${MIP_CLOSURE_MODEL_PATH}/point_cloud/iteration_${MIP_CLOSURE_ITERATION}/point_cloud.ply" >&2
  fi
  exit 1
fi

if [[ "${USE_SR_PRIOR}" == "1" ]]; then
  if [[ ! -d "${SR_PRIOR_ROOT}/${SR_PRIOR_SUBDIR}" ]]; then
    echo "[recover-cleaned-mip-lr-v0] missing prepared SR fused prior dir: ${SR_PRIOR_ROOT}/${SR_PRIOR_SUBDIR}" >&2
    echo "[recover-cleaned-mip-lr-v0] run run_mip_to_sof_surface_v0_kitchen.sh once, or pass SR_PRIOR_ROOT to an existing prepared cache." >&2
    exit 1
  fi
  if [[ ! -d "${SR_PRIOR_ROOT}/${SR_ANCHOR_SUBDIR}" ]]; then
    echo "[recover-cleaned-mip-lr-v0] missing prepared SR anchor dir: ${SR_PRIOR_ROOT}/${SR_ANCHOR_SUBDIR}" >&2
    exit 1
  fi
  if [[ ! -d "${SR_PRIOR_ROOT}/${SR_PRIOR_MASK_SUBDIR}" ]]; then
    echo "[recover-cleaned-mip-lr-v0] warning: missing SR usable mask dir, Python will use consistency gate only: ${SR_PRIOR_ROOT}/${SR_PRIOR_MASK_SUBDIR}" >&2
  fi
fi

echo "[recover-cleaned-mip-lr-v0] scene       : ${SCENE_ROOT}"
echo "[recover-cleaned-mip-lr-v0] start model : ${START_MODEL_PATH} iter=${START_ITERATION}"
echo "[recover-cleaned-mip-lr-v0] anchor model: ${ANCHOR_MODEL_PATH} iter=${ANCHOR_ITERATION}"
echo "[recover-cleaned-mip-lr-v0] output      : ${OUTPUT_MODEL}"
echo "[recover-cleaned-mip-lr-v0] profile     : ${REFINE_PROFILE}"
echo "[recover-cleaned-mip-lr-v0] train       : iter=${ITERATIONS} xyz_lr=${XYZ_LR} op=${ENABLE_OPACITY_UPDATE} scale=${ENABLE_SCALE_UPDATE} dc=${ENABLE_DC_UPDATE}/${DC_LR} rest=${ENABLE_REST_UPDATE}/${REST_LR}"
echo "[recover-cleaned-mip-lr-v0] phase       : mode=${PHASE_MODE} block=${PHASE_BLOCK_STEPS} mask=${OPTIMIZE_GAUSSIAN_MASK_PAYLOAD:-disabled} prefilter=${PRIOR_PREFILTER_MODE}"
echo "[recover-cleaned-mip-lr-v0] SR prior    : ${USE_SR_PRIOR} root=${SR_PRIOR_ROOT:-disabled} images=${SR_IMAGES_SUBDIR} view_mode=${SR_VIEW_MODE} weights=${SR_PRIOR_L1_WEIGHT}/${SR_PRIOR_HF_WEIGHT} hf_excess=${LAMBDA_PREMUL_HF_EXCESS} k=${PREMUL_HF_EXCESS_KERNEL} ratio=${PREMUL_HF_EXCESS_RATIO} margin=${PREMUL_HF_EXCESS_MARGIN}"
echo "[recover-cleaned-mip-lr-v0] mip HR sup  : residual_anchor=${SR_RESIDUAL_ANCHOR} lowfreq=${LAMBDA_MIP_HR_LOWFREQ} kernel=${MIP_HR_LOWFREQ_KERNEL}"
echo "[recover-cleaned-mip-lr-v0] mip closure : ${MIP_CLOSURE_MODEL_PATH} images=${MIP_CLOSURE_IMAGES_SUBDIR} max=${MIP_CLOSURE_MAX_VIEWS} weights=${LAMBDA_MIP_CLOSURE_ALPHA}/${LAMBDA_MIP_CLOSURE_PREMUL}/${LAMBDA_MIP_CLOSURE_DEPTH} over=${LAMBDA_MIP_CLOSURE_ALPHA_OVER}/${LAMBDA_MIP_CLOSURE_PREMUL_OVER}"
echo "[recover-cleaned-mip-lr-v0] star release: payload=${STAR_QUARANTINE_PAYLOAD_PATH:-disabled} iter=${STAR_RELEASE_START_ITER}->${STAR_RELEASE_END_ITER} rest=${STAR_RELEASE_REST_START_SCALE}->${STAR_RELEASE_REST_END_SCALE} tau=${STAR_RELEASE_TAU_START_SCALE}->${STAR_RELEASE_TAU_END_SCALE} opacity=${STAR_RELEASE_OPACITY}"
echo "[recover-cleaned-mip-lr-v0] reparam ctrl : src=${REPARAM_OUTPUT_SOURCE_IDX_PATH:-disabled} parent=${REPARAM_PARENT_MASK_PATH:-disabled} child=${REPARAM_CHILD_MASK_PATH:-disabled} geom=${GEOMETRY_PARENT_MASK_PATH:-disabled} weights=${LAMBDA_REPARAM_MASS_CAP}/${LAMBDA_REPARAM_CHILD_TAU_CAP}/${LAMBDA_GEOMETRY_PARENT_TAU_BRAKE}"
echo "[recover-cleaned-mip-lr-v0] risk guard  : base=${LAMBDA_RISK_OPACITY}/${LAMBDA_RISK_SCALE}/${LAMBDA_RISK_REST} sr=${LAMBDA_SR_RISK_OPACITY}/${LAMBDA_SR_RISK_SCALE}/${LAMBDA_SR_RISK_REST} boost=${SR_RISK_BOOST}"
echo "[recover-cleaned-mip-lr-v0] prefilter   : payload=${PRIOR_PREFILTER_MASK_PAYLOAD:-disabled} save=${PRIOR_PREFILTER_SAVE_PATH} touch=${PRIOR_PREFILTER_MIN_TOUCH_VIEWS}/${PRIOR_PREFILTER_MIN_VISIBLE_VIEWS} ratio=${PRIOR_PREFILTER_MIN_TOUCH_RATIO}"
echo "[recover-cleaned-mip-lr-v0] filter3D    : before_train=${RECOMPUTE_FILTER3D_BEFORE_TRAIN} final=${RECOMPUTE_FILTER3D}"
echo "[recover-cleaned-mip-lr-v0] render      : ${RUN_RENDER} images=${RENDER_IMAGES_SUBDIR} split=${RENDER_SPLIT}"
echo "[recover-cleaned-mip-lr-v0] render ckpt : ${RENDER_CHECKPOINTS} save_every=${SAVE_EVERY} max_views=${RENDER_CHECKPOINT_MAX_VIEWS}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/recover_cleaned_mip_lr_v0.py"
  --scene_root "${SCENE_ROOT}"
  --start_model_path "${START_MODEL_PATH}"
  --output_model_path "${OUTPUT_MODEL}"
  --anchor_model_path "${ANCHOR_MODEL_PATH}"
  --start_iteration "${START_ITERATION}"
  --anchor_iteration "${ANCHOR_ITERATION}"
  --output_iteration "${OUTPUT_ITERATION}"
  --images_subdir "${IMAGES_SUBDIR}"
  --max_views "${MAX_VIEWS}"
  --iterations "${ITERATIONS}"
  --phase_mode "${PHASE_MODE}"
  --phase_block_steps "${PHASE_BLOCK_STEPS}"
  --xyz_lr "${XYZ_LR}"
  --opacity_lr "${OPACITY_LR}"
  --scale_lr "${SCALE_LR}"
  --dc_lr "${DC_LR}"
  --rest_lr "${REST_LR}"
  --lambda_lr_rgb "${LAMBDA_LR_RGB}"
  --lambda_anchor_rgb "${LAMBDA_ANCHOR_RGB}"
  --lambda_xyz_anchor "${LAMBDA_XYZ_ANCHOR}"
  --lambda_opacity_anchor "${LAMBDA_OPACITY_ANCHOR}"
  --lambda_scale_anchor "${LAMBDA_SCALE_ANCHOR}"
  --lambda_dc_anchor "${LAMBDA_DC_ANCHOR}"
  --lambda_rest_anchor "${LAMBDA_REST_ANCHOR}"
  --lambda_risk_opacity "${LAMBDA_RISK_OPACITY}"
  --lambda_risk_scale "${LAMBDA_RISK_SCALE}"
  --lambda_risk_rest "${LAMBDA_RISK_REST}"
  --lambda_sr_risk_opacity "${LAMBDA_SR_RISK_OPACITY}"
  --lambda_sr_risk_scale "${LAMBDA_SR_RISK_SCALE}"
  --lambda_sr_risk_rest "${LAMBDA_SR_RISK_REST}"
  --sr_risk_boost "${SR_RISK_BOOST}"
  --sr_prior_root "${SR_PRIOR_ROOT}"
  --sr_prior_subdir "${SR_PRIOR_SUBDIR}"
  --sr_prior_mask_subdir "${SR_PRIOR_MASK_SUBDIR}"
  --sr_anchor_subdir "${SR_ANCHOR_SUBDIR}"
  --sr_prior_mask_suffix "${SR_PRIOR_MASK_SUFFIX}"
  --sr_images_subdir "${SR_IMAGES_SUBDIR}"
  --sr_max_views "${SR_MAX_VIEWS}"
  --sr_view_mode "${SR_VIEW_MODE}"
  --lambda_sr_prior_l1 "${SR_PRIOR_L1_WEIGHT}"
  --lambda_sr_prior_hf "${SR_PRIOR_HF_WEIGHT}"
  --sr_prior_warmup_start_iter "${SR_PRIOR_WARMUP_START_ITER}"
  --sr_prior_warmup_end_iter "${SR_PRIOR_WARMUP_END_ITER}"
  --sr_prior_start_scale "${SR_PRIOR_START_SCALE}"
  --sr_prior_end_scale "${SR_PRIOR_END_SCALE}"
  --sr_prior_update_scale "${SR_PRIOR_UPDATE_SCALE}"
  --sr_prior_schedule_mode "${SR_PRIOR_SCHEDULE_MODE}"
  --lambda_premul_hf_excess "${LAMBDA_PREMUL_HF_EXCESS}"
  --premul_hf_excess_kernel "${PREMUL_HF_EXCESS_KERNEL}"
  --premul_hf_excess_ratio "${PREMUL_HF_EXCESS_RATIO}"
  --premul_hf_excess_margin "${PREMUL_HF_EXCESS_MARGIN}"
  --sr_residual_anchor "${SR_RESIDUAL_ANCHOR}"
  --lambda_mip_hr_lowfreq "${LAMBDA_MIP_HR_LOWFREQ}"
  --mip_hr_lowfreq_kernel "${MIP_HR_LOWFREQ_KERNEL}"
  --mip_hr_lowfreq_consistency_threshold "${MIP_HR_LOWFREQ_CONSISTENCY_THRESHOLD}"
  --mip_hr_lowfreq_mask_floor "${MIP_HR_LOWFREQ_MASK_FLOOR}"
  --mip_closure_model_path "${MIP_CLOSURE_MODEL_PATH}"
  --mip_closure_iteration "${MIP_CLOSURE_ITERATION}"
  --mip_closure_images_subdir "${MIP_CLOSURE_IMAGES_SUBDIR}"
  --mip_closure_max_views "${MIP_CLOSURE_MAX_VIEWS}"
  --lambda_mip_closure_alpha "${LAMBDA_MIP_CLOSURE_ALPHA}"
  --lambda_mip_closure_premul "${LAMBDA_MIP_CLOSURE_PREMUL}"
  --lambda_mip_closure_depth "${LAMBDA_MIP_CLOSURE_DEPTH}"
  --lambda_mip_closure_alpha_over "${LAMBDA_MIP_CLOSURE_ALPHA_OVER}"
  --lambda_mip_closure_premul_over "${LAMBDA_MIP_CLOSURE_PREMUL_OVER}"
  --mip_closure_kernel "${MIP_CLOSURE_KERNEL}"
  --mip_closure_alpha_threshold "${MIP_CLOSURE_ALPHA_THRESHOLD}"
  --mip_closure_reference_lowpass "${MIP_CLOSURE_REFERENCE_LOWPASS}"
  --mip_closure_min_pixels "${MIP_CLOSURE_MIN_PIXELS}"
  --mip_closure_depth_relative_min "${MIP_CLOSURE_DEPTH_RELATIVE_MIN}"
  --sr_prior_mask_floor "${SR_PRIOR_MASK_FLOOR}"
  --sr_prior_consistency_threshold "${SR_PRIOR_CONSISTENCY_THRESHOLD}"
  --sr_prior_min_valid_ratio "${SR_PRIOR_MIN_VALID_RATIO}"
  --sr_prior_min_pixels "${SR_PRIOR_MIN_PIXELS}"
  --sr_prior_delta_clip "${SR_PRIOR_DELTA_CLIP}"
  --risk_radius_threshold "${RISK_RADIUS_THRESHOLD}"
  --risk_lr_residual_threshold "${RISK_LR_RESIDUAL_THRESHOLD}"
  --risk_anisotropy_threshold "${RISK_ANISOTROPY_THRESHOLD}"
  --risk_min_support_views "${RISK_MIN_SUPPORT_VIEWS}"
  --optimize_gaussian_mask_payload "${OPTIMIZE_GAUSSIAN_MASK_PAYLOAD}"
  --optimize_gaussian_mask_key "${OPTIMIZE_GAUSSIAN_MASK_KEY}"
  --gaussian_update_scale "${GAUSSIAN_UPDATE_SCALE}"
  --gaussian_scale_axis_mode "${GAUSSIAN_SCALE_AXIS_MODE}"
  --prior_prefilter_mode "${PRIOR_PREFILTER_MODE}"
  --prior_prefilter_mask_payload "${PRIOR_PREFILTER_MASK_PAYLOAD}"
  --prior_prefilter_mask_key "${PRIOR_PREFILTER_MASK_KEY}"
  --prior_prefilter_view_limit "${PRIOR_PREFILTER_VIEW_LIMIT}"
  --prior_prefilter_min_touch_views "${PRIOR_PREFILTER_MIN_TOUCH_VIEWS}"
  --prior_prefilter_min_visible_views "${PRIOR_PREFILTER_MIN_VISIBLE_VIEWS}"
  --prior_prefilter_min_touch_ratio "${PRIOR_PREFILTER_MIN_TOUCH_RATIO}"
  --prior_prefilter_min_candidate_opacity "${PRIOR_PREFILTER_MIN_CANDIDATE_OPACITY}"
  --prior_prefilter_radius_scale "${PRIOR_PREFILTER_RADIUS_SCALE}"
  --prior_prefilter_min_touch_radius_px "${PRIOR_PREFILTER_MIN_TOUCH_RADIUS_PX}"
  --prior_prefilter_max_touch_radius_px "${PRIOR_PREFILTER_MAX_TOUCH_RADIUS_PX}"
  --prior_prefilter_save_path "${PRIOR_PREFILTER_SAVE_PATH}"
  --max_displacement_ratio "${MAX_DISPLACEMENT_RATIO}"
  --max_displacement_abs "${MAX_DISPLACEMENT_ABS}"
  --save_every "${SAVE_EVERY}"
  --recompute_filter3d_before_train "${RECOMPUTE_FILTER3D_BEFORE_TRAIN}"
  --recompute_filter3d "${RECOMPUTE_FILTER3D}"
  --star_quarantine_payload_path "${STAR_QUARANTINE_PAYLOAD_PATH}"
  --star_release_start_iter "${STAR_RELEASE_START_ITER}"
  --star_release_end_iter "${STAR_RELEASE_END_ITER}"
  --star_release_mode "${STAR_RELEASE_MODE}"
  --star_release_rest_start_scale "${STAR_RELEASE_REST_START_SCALE}"
  --star_release_rest_end_scale "${STAR_RELEASE_REST_END_SCALE}"
  --star_release_tau_start_scale "${STAR_RELEASE_TAU_START_SCALE}"
  --star_release_tau_end_scale "${STAR_RELEASE_TAU_END_SCALE}"
  --star_release_min_alpha "${STAR_RELEASE_MIN_ALPHA}"
  --reparam_output_source_idx_path "${REPARAM_OUTPUT_SOURCE_IDX_PATH}"
  --reparam_parent_mask_path "${REPARAM_PARENT_MASK_PATH}"
  --reparam_child_mask_path "${REPARAM_CHILD_MASK_PATH}"
  --geometry_parent_mask_path "${GEOMETRY_PARENT_MASK_PATH}"
  --lambda_reparam_mass_cap "${LAMBDA_REPARAM_MASS_CAP}"
  --reparam_mass_cap_eps "${REPARAM_MASS_CAP_EPS}"
  --lambda_reparam_child_tau_cap "${LAMBDA_REPARAM_CHILD_TAU_CAP}"
  --reparam_child_tau_cap_scale "${REPARAM_CHILD_TAU_CAP_SCALE}"
  --reparam_child_tau_cap_abs "${REPARAM_CHILD_TAU_CAP_ABS}"
  --lambda_geometry_parent_tau_brake "${LAMBDA_GEOMETRY_PARENT_TAU_BRAKE}"
  --geometry_parent_tau_scale "${GEOMETRY_PARENT_TAU_SCALE}"
  --reparam_prune_iters "${REPARAM_PRUNE_ITERS}"
  --reparam_prune_child_dead_tau "${REPARAM_PRUNE_CHILD_DEAD_TAU}"
  --reparam_prune_child_spike_tau_scale "${REPARAM_PRUNE_CHILD_SPIKE_TAU_SCALE}"
  --reparam_prune_child_spike_tau_abs "${REPARAM_PRUNE_CHILD_SPIKE_TAU_ABS}"
  --reparam_prune_child_spike_anisotropy "${REPARAM_PRUNE_CHILD_SPIKE_ANISOTROPY}"
  --reparam_prune_child_spike_risk_min "${REPARAM_PRUNE_CHILD_SPIKE_RISK_MIN}"
)

if [[ "${ENABLE_OPACITY_UPDATE}" == "1" ]]; then
  CMD+=(--enable_opacity_update)
fi
if [[ "${ENABLE_SCALE_UPDATE}" == "1" ]]; then
  CMD+=(--enable_scale_update)
fi
if [[ "${ENABLE_DC_UPDATE}" == "1" ]]; then
  CMD+=(--enable_dc_update)
fi
if [[ "${ENABLE_REST_UPDATE}" == "1" ]]; then
  CMD+=(--enable_rest_update)
fi
if [[ "${DISABLE_SR_PRIOR_HF_RESIDUAL}" == "1" ]]; then
  CMD+=(--disable_sr_prior_hf_residual)
fi
if [[ "${STAR_RELEASE_OPACITY}" == "1" ]]; then
  CMD+=(--star_release_opacity)
fi

"${CMD[@]}"

EFFECTIVE_OUTPUT_ITERATION="$(resolve_point_cloud_iteration "${OUTPUT_MODEL}" "${OUTPUT_ITERATION}")"
EFFECTIVE_RENDER_PREVIEW_PATH="${RENDER_PREVIEW_PATH}"
DEFAULT_RENDER_PREVIEW_PATH="${RENDER_DIR}/contact_sheet_${RENDER_IMAGES_SUBDIR}_${RENDER_SPLIT}_${OUTPUT_ITERATION}.png"
if [[ "${OUTPUT_ITERATION}" =~ ^- ]] && [[ "${RENDER_PREVIEW_PATH}" == "${DEFAULT_RENDER_PREVIEW_PATH}" ]]; then
  EFFECTIVE_RENDER_PREVIEW_PATH="${RENDER_DIR}/contact_sheet_${RENDER_IMAGES_SUBDIR}_${RENDER_SPLIT}_${EFFECTIVE_OUTPUT_ITERATION}.png"
fi

if [[ "${RUN_RENDER}" == "1" && "${RENDER_CHECKPOINTS}" == "1" && "${SAVE_EVERY}" != "0" ]]; then
  while IFS= read -r checkpoint_dir; do
    checkpoint_name="$(basename "${checkpoint_dir}")"
    checkpoint_iter="${checkpoint_name#iteration_}"
    if [[ -z "${checkpoint_iter}" || "${checkpoint_iter}" == "${EFFECTIVE_OUTPUT_ITERATION}" ]]; then
      continue
    fi
    if [[ ! -f "${checkpoint_dir}/point_cloud.ply" ]]; then
      continue
    fi
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
      --scene_root "${SCENE_ROOT}" \
      --model_path "${OUTPUT_MODEL}" \
      --output_dir "${RENDER_DIR}" \
      --images_subdir "${RENDER_IMAGES_SUBDIR}" \
      --iteration "${checkpoint_iter}" \
      --split "${RENDER_SPLIT}" \
      --max_views "${RENDER_CHECKPOINT_MAX_VIEWS}"

    if [[ "${RENDER_PREVIEW}" == "1" ]]; then
      "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/make_render_contact_sheet.py" \
        --render_dir "${RENDER_DIR}/${RENDER_SPLIT}/ours_${checkpoint_iter}/renders" \
        --output_path "${RENDER_DIR}/contact_sheet_${RENDER_IMAGES_SUBDIR}_${RENDER_SPLIT}_${checkpoint_iter}.png" \
        --max_images "${RENDER_PREVIEW_MAX_IMAGES}" \
        --columns "${RENDER_PREVIEW_COLUMNS}" \
        --thumb_width "${RENDER_PREVIEW_THUMB_WIDTH}"
    fi
  done < <(
    find "${OUTPUT_MODEL}/point_cloud" -mindepth 1 -maxdepth 1 -type d -name 'iteration_*' | sort -V
  )
fi

if [[ "${RUN_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${OUTPUT_MODEL}" \
    --output_dir "${RENDER_DIR}" \
    --images_subdir "${RENDER_IMAGES_SUBDIR}" \
    --iteration "${EFFECTIVE_OUTPUT_ITERATION}" \
    --split "${RENDER_SPLIT}"

  if [[ "${RENDER_PREVIEW}" == "1" ]]; then
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/make_render_contact_sheet.py" \
      --render_dir "${RENDER_DIR}/${RENDER_SPLIT}/ours_${EFFECTIVE_OUTPUT_ITERATION}/renders" \
      --output_path "${EFFECTIVE_RENDER_PREVIEW_PATH}" \
      --max_images "${RENDER_PREVIEW_MAX_IMAGES}" \
      --columns "${RENDER_PREVIEW_COLUMNS}" \
      --thumb_width "${RENDER_PREVIEW_THUMB_WIDTH}"
  fi
fi

echo "[done] output model : ${OUTPUT_MODEL}"
echo "[done] output ply   : ${OUTPUT_MODEL}/point_cloud/iteration_${EFFECTIVE_OUTPUT_ITERATION}/point_cloud.ply"
echo "[done] summary      : ${OUTPUT_MODEL}/recover_cleaned_mip_lr_v0_summary.json"
if [[ "${RUN_RENDER}" == "1" ]]; then
  echo "[done] renders      : ${RENDER_DIR}/${RENDER_SPLIT}/ours_${EFFECTIVE_OUTPUT_ITERATION}/renders"
  if [[ "${RENDER_PREVIEW}" == "1" ]]; then
    echo "[done] preview      : ${EFFECTIVE_RENDER_PREVIEW_PATH}"
  fi
fi

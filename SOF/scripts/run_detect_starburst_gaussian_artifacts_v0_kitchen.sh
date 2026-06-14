#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
MIP_RENDER_RESOLUTION="${MIP_RENDER_RESOLUTION:-1}"
MIP_RENDER_DIR="${MIP_RENDER_DIR:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}/test/ours_${MIP_ITERATION}/test_preds_${MIP_RENDER_RESOLUTION}}"
RECOVER_RUN_NAME="${RECOVER_RUN_NAME:-view_aligned_volume_delete_v1_init_energy_curve_refit_v0_prune_more_v1_mip_hr_anchor_v0_miphr_v1}"
MODEL_NAME="${MODEL_NAME:-recovered_mip_model_lr_miphr_v1}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${RECOVER_RUN_NAME}}"
MODEL_PATH="${MODEL_PATH:-${RUN_ROOT}/${MODEL_NAME}}"
ITERATION="${ITERATION:-31600}"

RUN_NAME="${RUN_NAME:-${RECOVER_RUN_NAME}_starburst_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/starburst_gaussian_scores_v0/${SCENE_NAME}/${RUN_NAME}}"

INTERACTION_IMAGES_SUBDIR="${INTERACTION_IMAGES_SUBDIR:-images_2}"
REFERENCE_SOURCE="${REFERENCE_SOURCE:-lr}"
REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-}"
REFERENCE_IMAGE_DIR="${REFERENCE_IMAGE_DIR:-}"
SR_PRIOR_ROOT="${SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft}"
SR_PRIOR_SUBDIR="${SR_PRIOR_SUBDIR:-fused_priors}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-16}"
VIEW_INDICES="${VIEW_INDICES:-}"
VISIBILITY_DOWNSAMPLE="${VISIBILITY_DOWNSAMPLE:-4}"
VISIBILITY_TOPK="${VISIBILITY_TOPK:-8}"
VISIBILITY_MAX_VISIBLE="${VISIBILITY_MAX_VISIBLE:-80000}"
VISIBILITY_MAX_PATCH_RADIUS="${VISIBILITY_MAX_PATCH_RADIUS:-2}"

LINE_LENGTHS="${LINE_LENGTHS:-9,17,31}"
ANGLES_DEG="${ANGLES_DEG:-0,22.5,45,67.5,90,112.5,135,157.5}"
HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-21}"
RIDGE_NORM_PERCENTILE="${RIDGE_NORM_PERCENTILE:-0.995}"
RESIDUAL_NORM_PERCENTILE="${RESIDUAL_NORM_PERCENTILE:-0.990}"
VIEW_STAR_QUANTILE="${VIEW_STAR_QUANTILE:-0.985}"
VIEW_STAR_MIN="${VIEW_STAR_MIN:-0.08}"
SELECT_QUANTILE="${SELECT_QUANTILE:-0.990}"
MIN_STAR_SCORE="${MIN_STAR_SCORE:-0.08}"
MIN_UNSUPPORTED_SCORE="${MIN_UNSUPPORTED_SCORE:-0.04}"
MIN_STAR_VIEW_COUNT="${MIN_STAR_VIEW_COUNT:-1}"
GLOBAL_LONG_AXIS_QUANTILE="${GLOBAL_LONG_AXIS_QUANTILE:-0.94}"
GLOBAL_ANISOTROPY_QUANTILE="${GLOBAL_ANISOTROPY_QUANTILE:-0.90}"
GLOBAL_RADIUS_MAX_QUANTILE="${GLOBAL_RADIUS_MAX_QUANTILE:-0.94}"
GLOBAL_PREFILTER_MIN_HITS="${GLOBAL_PREFILTER_MIN_HITS:-2}"
GLOBAL_PREFILTER_MAX_FRACTION="${GLOBAL_PREFILTER_MAX_FRACTION:-0.10}"
GLOBAL_PREFILTER_MAX_COUNT="${GLOBAL_PREFILTER_MAX_COUNT:-120000}"
MAX_CANDIDATE_FRACTION="${MAX_CANDIDATE_FRACTION:-0.015}"
MAX_CANDIDATE_COUNT="${MAX_CANDIDATE_COUNT:-30000}"
NUM_DEBUG_VIEWS="${NUM_DEBUG_VIEWS:-4}"
EXPORT_CANDIDATE_MODEL="${EXPORT_CANDIDATE_MODEL:-1}"
RENDER_CANDIDATE_AFTER="${RENDER_CANDIDATE_AFTER:-1}"
CANDIDATE_RENDER_SPLIT="${CANDIDATE_RENDER_SPLIT:-test}"
CANDIDATE_RENDER_MAX_VIEWS="${CANDIDATE_RENDER_MAX_VIEWS:-4}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-srtest}"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_model_path() {
  local candidate_root="$1"
  local iteration="$2"
  if [[ -e "${candidate_root}/point_cloud/iteration_${iteration}/point_cloud.ply" ]]; then
    printf '%s\n' "${candidate_root}"
    return 0
  fi
  if [[ -d "${candidate_root}" ]]; then
    local child
    for child in "${candidate_root}"/*; do
      if [[ -d "${child}" && -e "${child}/point_cloud/iteration_${iteration}/point_cloud.ply" ]]; then
        printf '%s\n' "${child}"
        return 0
      fi
    done
  fi
  return 1
}

case "${REFERENCE_SOURCE}" in
  lr)
    REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_8}"
    ;;
  sr)
    REFERENCE_IMAGE_DIR="${REFERENCE_IMAGE_DIR:-${SR_PRIOR_ROOT}/${SR_PRIOR_SUBDIR}}"
    ;;
  mip_lr_render|mip_render)
    REFERENCE_IMAGE_DIR="${REFERENCE_IMAGE_DIR:-${MIP_RENDER_DIR}}"
    ;;
  none)
    REFERENCE_IMAGES_SUBDIR=""
    REFERENCE_IMAGE_DIR=""
    ;;
  custom)
    ;;
  *)
    echo "[starburst-v0] unsupported REFERENCE_SOURCE=${REFERENCE_SOURCE}; use lr, sr, mip_lr_render, custom, or none" >&2
    exit 1
    ;;
esac

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

if RESOLVED_MODEL_PATH="$(resolve_model_path "${MODEL_PATH}" "${ITERATION}")"; then
  MODEL_PATH="${RESOLVED_MODEL_PATH}"
elif RESOLVED_MODEL_PATH="$(resolve_model_path "${RUN_ROOT}" "${ITERATION}")"; then
  MODEL_PATH="${RESOLVED_MODEL_PATH}"
fi

if [[ ! -e "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
  echo "[starburst-v0] model point cloud not found: ${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" >&2
  echo "[starburst-v0] searched run root      : ${RUN_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/${INTERACTION_IMAGES_SUBDIR}" ]]; then
  echo "[starburst-v0] interaction image dir not found: ${SCENE_ROOT}/${INTERACTION_IMAGES_SUBDIR}" >&2
  exit 1
fi

if [[ -n "${REFERENCE_IMAGES_SUBDIR}" && ! -d "${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}" ]]; then
  echo "[starburst-v0] reference image dir not found: ${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}" >&2
  exit 1
fi

if [[ -n "${REFERENCE_IMAGE_DIR}" && ! -d "${REFERENCE_IMAGE_DIR}" ]]; then
  echo "[starburst-v0] reference image dir not found: ${REFERENCE_IMAGE_DIR}" >&2
  exit 1
fi

echo "[starburst-v0] scene root : ${SCENE_ROOT}"
echo "[starburst-v0] model path : ${MODEL_PATH}"
echo "[starburst-v0] iteration  : ${ITERATION}"
echo "[starburst-v0] output dir : ${OUTPUT_DIR}"
echo "[starburst-v0] split/views: ${SPLIT} max=${MAX_VIEWS} indices=${VIEW_INDICES:-<uniform>}"
echo "[starburst-v0] reference  : source=${REFERENCE_SOURCE} subdir=${REFERENCE_IMAGES_SUBDIR:-<none>} dir=${REFERENCE_IMAGE_DIR:-<none>}"
if [[ "${REFERENCE_SOURCE}" == "mip_lr_render" || "${REFERENCE_SOURCE}" == "mip_render" ]]; then
  echo "[starburst-v0] mip render : ${MIP_RENDER_DIR}"
fi
echo "[starburst-v0] global gate: long_q=${GLOBAL_LONG_AXIS_QUANTILE} aniso_q=${GLOBAL_ANISOTROPY_QUANTILE} radius_q=${GLOBAL_RADIUS_MAX_QUANTILE} hits>=${GLOBAL_PREFILTER_MIN_HITS} cap=${GLOBAL_PREFILTER_MAX_FRACTION} max_count=${GLOBAL_PREFILTER_MAX_COUNT}"
echo "[starburst-v0] select     : q=${SELECT_QUANTILE} cap=${MAX_CANDIDATE_FRACTION} max_count=${MAX_CANDIDATE_COUNT}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/detect_starburst_gaussian_artifacts_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --interaction_images_subdir "${INTERACTION_IMAGES_SUBDIR}"
  --reference_images_subdir "${REFERENCE_IMAGES_SUBDIR}"
  --reference_image_dir "${REFERENCE_IMAGE_DIR}"
  --iteration "${ITERATION}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
  --view_indices "${VIEW_INDICES}"
  --visibility_downsample "${VISIBILITY_DOWNSAMPLE}"
  --visibility_topk "${VISIBILITY_TOPK}"
  --visibility_max_visible "${VISIBILITY_MAX_VISIBLE}"
  --visibility_max_patch_radius "${VISIBILITY_MAX_PATCH_RADIUS}"
  --line_lengths "${LINE_LENGTHS}"
  --angles_deg "${ANGLES_DEG}"
  --highpass_kernel "${HIGHPASS_KERNEL}"
  --ridge_norm_percentile "${RIDGE_NORM_PERCENTILE}"
  --residual_norm_percentile "${RESIDUAL_NORM_PERCENTILE}"
  --view_star_quantile "${VIEW_STAR_QUANTILE}"
  --view_star_min "${VIEW_STAR_MIN}"
  --select_quantile "${SELECT_QUANTILE}"
  --min_star_score "${MIN_STAR_SCORE}"
  --min_unsupported_score "${MIN_UNSUPPORTED_SCORE}"
  --min_star_view_count "${MIN_STAR_VIEW_COUNT}"
  --global_long_axis_quantile "${GLOBAL_LONG_AXIS_QUANTILE}"
  --global_anisotropy_quantile "${GLOBAL_ANISOTROPY_QUANTILE}"
  --global_radius_max_quantile "${GLOBAL_RADIUS_MAX_QUANTILE}"
  --global_prefilter_min_hits "${GLOBAL_PREFILTER_MIN_HITS}"
  --global_prefilter_max_fraction "${GLOBAL_PREFILTER_MAX_FRACTION}"
  --global_prefilter_max_count "${GLOBAL_PREFILTER_MAX_COUNT}"
  --max_candidate_fraction "${MAX_CANDIDATE_FRACTION}"
  --max_candidate_count "${MAX_CANDIDATE_COUNT}"
  --num_debug_views "${NUM_DEBUG_VIEWS}"
)

if [[ "${EXPORT_CANDIDATE_MODEL}" == "1" ]]; then
  CMD+=(--export_candidate_model)
fi

"${CMD[@]}"

echo "[done] score payload : ${OUTPUT_DIR}/starburst_gaussian_scores_v0.pt"
echo "[done] candidate mask: ${OUTPUT_DIR}/starburst_candidate_mask.pt"
echo "[done] summary       : ${OUTPUT_DIR}/summary.json"
echo "[done] debug views   : ${OUTPUT_DIR}/debug_views"
if [[ "${EXPORT_CANDIDATE_MODEL}" == "1" ]]; then
  echo "[done] candidate ply : ${OUTPUT_DIR}/starburst_candidate_ply/point_cloud/iteration_${ITERATION}/point_cloud.ply"
fi

if [[ "${EXPORT_CANDIDATE_MODEL}" == "1" && "${RENDER_CANDIDATE_AFTER}" == "1" ]]; then
  CANDIDATE_MODEL_DIR="${OUTPUT_DIR}/starburst_candidate_ply"
  if [[ ! -e "${CANDIDATE_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
    echo "[starburst-v0] candidate model point cloud not found: ${CANDIDATE_MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" >&2
    exit 1
  fi
  echo "[starburst-v0] rendering candidate-only preview on ${INTERACTION_IMAGES_SUBDIR} (${CANDIDATE_RENDER_SPLIT}, max=${CANDIDATE_RENDER_MAX_VIEWS})"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --model_path "${CANDIDATE_MODEL_DIR}" \
    --scene_root "${SCENE_ROOT}" \
    --images_subdir "${INTERACTION_IMAGES_SUBDIR}" \
    --iteration "${ITERATION}" \
    --split "${CANDIDATE_RENDER_SPLIT}" \
    --max_views "${CANDIDATE_RENDER_MAX_VIEWS}"
  echo "[done] candidate renders: ${CANDIDATE_MODEL_DIR}/${CANDIDATE_RENDER_SPLIT}/ours_${ITERATION}/renders"
fi

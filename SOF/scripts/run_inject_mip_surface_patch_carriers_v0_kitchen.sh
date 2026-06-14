#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"

SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-view_aligned_volume_delete_v1_init_energy_curve_refit_v0_prune_more_v1_mip_hr_anchor_v0_miphr_v1}"
MODEL_NAME="${MODEL_NAME:-recovered_mip_model_lr_miphr_v1}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${SOURCE_RUN_NAME}}"
MODEL_PATH="${MODEL_PATH:-${RUN_ROOT}/${MODEL_NAME}}"
ITERATION="${ITERATION:-31600}"

MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input_init_energy_curve_refit_v0}"
MIP_ITERATION="${MIP_ITERATION:-30000}"

RUN_NAME="${RUN_NAME:-${SOURCE_RUN_NAME}_surface_patch_v0}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/post_cleanup_surface_patch_carriers_v0/${SCENE_NAME}/${RUN_NAME}}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-16}"
VIEW_INDICES="${VIEW_INDICES:-}"
RUN_RENDER="${RUN_RENDER:-1}"

ALPHA_MARGIN="${ALPHA_MARGIN:-0.02}"
MIN_MIP_ALPHA="${MIN_MIP_ALPHA:-0.08}"
SELECT_QUANTILE="${SELECT_QUANTILE:-0.985}"
MIN_DEFICIT_SCORE="${MIN_DEFICIT_SCORE:-0.05}"
MIN_PLANARITY="${MIN_PLANARITY:-5.0}"
MIN_OPACITY="${MIN_OPACITY:-0.03}"
MIN_TANGENT_EXTENT_RATIO="${MIN_TANGENT_EXTENT_RATIO:-0.0008}"
MAX_CANDIDATE_FRACTION="${MAX_CANDIDATE_FRACTION:-0.010}"
MAX_CANDIDATE_COUNT="${MAX_CANDIDATE_COUNT:-12000}"

PATCH_GRID_SIDE="${PATCH_GRID_SIDE:-2}"
PATCH_OFFSET_SCALE="${PATCH_OFFSET_SCALE:-0.75}"
PATCH_TANGENT_SCALE_MULTIPLIER="${PATCH_TANGENT_SCALE_MULTIPLIER:-0.55}"
PATCH_NORMAL_SCALE_MULTIPLIER="${PATCH_NORMAL_SCALE_MULTIPLIER:-0.35}"
PATCH_OPACITY_SCALE="${PATCH_OPACITY_SCALE:-0.65}"
PATCH_MAX_OPACITY="${PATCH_MAX_OPACITY:-0.32}"
PATCH_FILTER_SCALE="${PATCH_FILTER_SCALE:-0.25}"
FILTER_CAP_RATIO="${FILTER_CAP_RATIO:-0.0005}"
ENERGY_CONSERVE_MODE="${ENERGY_CONSERVE_MODE:-area}"
FEATURES_REST_SCALE="${FEATURES_REST_SCALE:-0.0}"

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

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

if RESOLVED_MODEL_PATH="$(resolve_model_path "${MODEL_PATH}" "${ITERATION}")"; then
  MODEL_PATH="${RESOLVED_MODEL_PATH}"
elif RESOLVED_MODEL_PATH="$(resolve_model_path "${RUN_ROOT}" "${ITERATION}")"; then
  MODEL_PATH="${RESOLVED_MODEL_PATH}"
fi

echo "[surface-patch-v0] scene root   : ${SCENE_ROOT}"
echo "[surface-patch-v0] model path   : ${MODEL_PATH}"
echo "[surface-patch-v0] mip model    : ${MIP_MODEL_PATH}"
echo "[surface-patch-v0] output model : ${OUTPUT_MODEL_PATH}"
echo "[surface-patch-v0] split/views  : ${SPLIT} max=${MAX_VIEWS} indices=${VIEW_INDICES:-<uniform>}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/inject_mip_surface_patch_carriers_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --mip_model_path "${MIP_MODEL_PATH}" \
  --output_model_path "${OUTPUT_MODEL_PATH}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --iteration "${ITERATION}" \
  --mip_iteration "${MIP_ITERATION}" \
  --split "${SPLIT}" \
  --max_views "${MAX_VIEWS}" \
  --view_indices "${VIEW_INDICES}" \
  --alpha_margin "${ALPHA_MARGIN}" \
  --min_mip_alpha "${MIN_MIP_ALPHA}" \
  --select_quantile "${SELECT_QUANTILE}" \
  --min_deficit_score "${MIN_DEFICIT_SCORE}" \
  --min_planarity "${MIN_PLANARITY}" \
  --min_opacity "${MIN_OPACITY}" \
  --min_tangent_extent_ratio "${MIN_TANGENT_EXTENT_RATIO}" \
  --max_candidate_fraction "${MAX_CANDIDATE_FRACTION}" \
  --max_candidate_count "${MAX_CANDIDATE_COUNT}" \
  --patch_grid_side "${PATCH_GRID_SIDE}" \
  --patch_offset_scale "${PATCH_OFFSET_SCALE}" \
  --patch_tangent_scale_multiplier "${PATCH_TANGENT_SCALE_MULTIPLIER}" \
  --patch_normal_scale_multiplier "${PATCH_NORMAL_SCALE_MULTIPLIER}" \
  --patch_opacity_scale "${PATCH_OPACITY_SCALE}" \
  --patch_max_opacity "${PATCH_MAX_OPACITY}" \
  --patch_filter_scale "${PATCH_FILTER_SCALE}" \
  --filter_cap_ratio "${FILTER_CAP_RATIO}" \
  --energy_conserve_mode "${ENERGY_CONSERVE_MODE}" \
  --features_rest_scale "${FEATURES_REST_SCALE}"

echo "[done] injected model : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
echo "[done] summary        : ${OUTPUT_MODEL_PATH}/mip_surface_patch_carriers_summary.json"
echo "[done] deficit payload : ${OUTPUT_MODEL_PATH}/mip_surface_coverage_deficit_v0.pt"

if [[ "${RUN_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${OUTPUT_MODEL_PATH}" \
    --images_subdir "${IMAGES_SUBDIR}" \
    --iteration "${ITERATION}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}"
  echo "[done] preview renders: ${OUTPUT_MODEL_PATH}/${SPLIT}/ours_${ITERATION}/renders"
fi

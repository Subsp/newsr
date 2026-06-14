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

DETECT_RUN_NAME="${DETECT_RUN_NAME:-${RECOVER_RUN_NAME}_starburst_v0}"
SCORE_DIR="${SCORE_DIR:-${SOF_ROOT}/output/starburst_gaussian_scores_v0/${SCENE_NAME}/${DETECT_RUN_NAME}}"
SCORE_PAYLOAD_PATH="${SCORE_PAYLOAD_PATH:-${SCORE_DIR}/starburst_gaussian_scores_v0.pt}"

OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${RECOVER_RUN_NAME}_starcurve_v0}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/post_cleanup_starburst_curve_repair_v0/${SCENE_NAME}/${OUTPUT_RUN_NAME}}"

INTERACTION_IMAGES_SUBDIR="${INTERACTION_IMAGES_SUBDIR:-images_2}"
REFERENCE_SOURCE="${REFERENCE_SOURCE:-mip_lr_render}"
RUN_DETECT="${RUN_DETECT:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-16}"
VIEW_INDICES="${VIEW_INDICES:-}"

SELECT_QUANTILE="${SELECT_QUANTILE:-0.990}"
MIN_STAR_SCORE="${MIN_STAR_SCORE:-0.08}"
MIN_UNSUPPORTED_SCORE="${MIN_UNSUPPORTED_SCORE:-0.04}"
MIN_STAR_VIEW_COUNT="${MIN_STAR_VIEW_COUNT:-1}"
MAX_CANDIDATE_FRACTION="${MAX_CANDIDATE_FRACTION:-0.015}"
MAX_CANDIDATE_COUNT="${MAX_CANDIDATE_COUNT:-30000}"

USE_PAYLOAD_CANDIDATE_MASK="${USE_PAYLOAD_CANDIDATE_MASK:-1}"
MIN_REPAIR_STARBURST_SCORE="${MIN_REPAIR_STARBURST_SCORE:-0.18}"
MIN_REPAIR_UNSUPPORTED_SCORE="${MIN_REPAIR_UNSUPPORTED_SCORE:-0.08}"
MIN_REPAIR_GEOMETRY_RISK="${MIN_REPAIR_GEOMETRY_RISK:-0.16}"
MIN_REPAIR_VISIBLE_COUNT="${MIN_REPAIR_VISIBLE_COUNT:-1}"
MAX_REPAIR_FRACTION="${MAX_REPAIR_FRACTION:-0.012}"
MAX_REPAIR_COUNT="${MAX_REPAIR_COUNT:-18000}"
SPLIT_COUNT="${SPLIT_COUNT:-6}"
OFFSET_SCALE="${OFFSET_SCALE:-0.90}"
CHILD_MAJOR_SCALE_MULTIPLIER="${CHILD_MAJOR_SCALE_MULTIPLIER:-0.32}"
CHILD_MINOR_SCALE_MULTIPLIER="${CHILD_MINOR_SCALE_MULTIPLIER:-0.78}"
CHILD_NORMAL_SCALE_MULTIPLIER="${CHILD_NORMAL_SCALE_MULTIPLIER:-0.60}"
CHILD_OPACITY_SCALE="${CHILD_OPACITY_SCALE:-0.82}"
CHILD_DC_SCALE="${CHILD_DC_SCALE:-0.92}"
CHILD_REST_SCALE="${CHILD_REST_SCALE:-0.10}"
CHILD_FILTER_SCALE="${CHILD_FILTER_SCALE:-0.35}"
FILTER_CAP_RATIO="${FILTER_CAP_RATIO:-0.0008}"
ENERGY_CONSERVE_MODE="${ENERGY_CONSERVE_MODE:-area}"

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

if [[ "${RUN_DETECT}" == "1" ]]; then
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP}" \
  MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME}" \
  MIP_ITERATION="${MIP_ITERATION}" \
  MIP_RENDER_RESOLUTION="${MIP_RENDER_RESOLUTION}" \
  MIP_RENDER_DIR="${MIP_RENDER_DIR}" \
  RECOVER_RUN_NAME="${RECOVER_RUN_NAME}" \
  MODEL_NAME="${MODEL_NAME}" \
  MODEL_PATH="${MODEL_PATH}" \
  ITERATION="${ITERATION}" \
  RUN_NAME="${DETECT_RUN_NAME}" \
  OUTPUT_DIR="${SCORE_DIR}" \
  INTERACTION_IMAGES_SUBDIR="${INTERACTION_IMAGES_SUBDIR}" \
  REFERENCE_SOURCE="${REFERENCE_SOURCE}" \
  SPLIT="${SPLIT}" \
  MAX_VIEWS="${MAX_VIEWS}" \
  VIEW_INDICES="${VIEW_INDICES}" \
  SELECT_QUANTILE="${SELECT_QUANTILE}" \
  MIN_STAR_SCORE="${MIN_STAR_SCORE}" \
  MIN_UNSUPPORTED_SCORE="${MIN_UNSUPPORTED_SCORE}" \
  MIN_STAR_VIEW_COUNT="${MIN_STAR_VIEW_COUNT}" \
  MAX_CANDIDATE_FRACTION="${MAX_CANDIDATE_FRACTION}" \
  MAX_CANDIDATE_COUNT="${MAX_CANDIDATE_COUNT}" \
  RENDER_CANDIDATE_AFTER=0 \
  bash "${SCRIPT_DIR}/run_detect_starburst_gaussian_artifacts_v0_kitchen.sh"
fi

if [[ ! -f "${SCORE_PAYLOAD_PATH}" ]]; then
  echo "[starcurve-v0] starburst payload not found: ${SCORE_PAYLOAD_PATH}" >&2
  exit 1
fi

echo "[starcurve-v0] scene root  : ${SCENE_ROOT}"
echo "[starcurve-v0] model path  : ${MODEL_PATH}"
echo "[starcurve-v0] score       : ${SCORE_PAYLOAD_PATH}"
echo "[starcurve-v0] output model: ${OUTPUT_MODEL_PATH}"
echo "[starcurve-v0] iteration   : ${ITERATION}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/apply_starburst_curve_repair_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --score_payload_path "${SCORE_PAYLOAD_PATH}"
  --output_model_path "${OUTPUT_MODEL_PATH}"
  --images_subdir "${INTERACTION_IMAGES_SUBDIR}"
  --iteration "${ITERATION}"
  --score_key starburst_score
  --candidate_key starburst_candidate
  --min_starburst_score "${MIN_REPAIR_STARBURST_SCORE}"
  --min_unsupported_score "${MIN_REPAIR_UNSUPPORTED_SCORE}"
  --min_geometry_risk "${MIN_REPAIR_GEOMETRY_RISK}"
  --min_visible_count "${MIN_REPAIR_VISIBLE_COUNT}"
  --max_repair_fraction "${MAX_REPAIR_FRACTION}"
  --max_repair_count "${MAX_REPAIR_COUNT}"
  --split_count "${SPLIT_COUNT}"
  --offset_scale "${OFFSET_SCALE}"
  --child_major_scale_multiplier "${CHILD_MAJOR_SCALE_MULTIPLIER}"
  --child_minor_scale_multiplier "${CHILD_MINOR_SCALE_MULTIPLIER}"
  --child_normal_scale_multiplier "${CHILD_NORMAL_SCALE_MULTIPLIER}"
  --child_opacity_scale "${CHILD_OPACITY_SCALE}"
  --child_dc_scale "${CHILD_DC_SCALE}"
  --child_rest_scale "${CHILD_REST_SCALE}"
  --child_filter_scale "${CHILD_FILTER_SCALE}"
  --filter_cap_ratio "${FILTER_CAP_RATIO}"
  --energy_conserve_mode "${ENERGY_CONSERVE_MODE}"
)

if [[ "${USE_PAYLOAD_CANDIDATE_MASK}" == "1" ]]; then
  CMD+=(--use_payload_candidate_mask)
fi

"${CMD[@]}"

echo "[done] repaired model : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
echo "[done] summary        : ${OUTPUT_MODEL_PATH}/starburst_curve_repair_summary.json"

if [[ "${RUN_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${OUTPUT_MODEL_PATH}" \
    --images_subdir "${INTERACTION_IMAGES_SUBDIR}" \
    --iteration "${ITERATION}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}"
  echo "[done] preview renders: ${OUTPUT_MODEL_PATH}/${SPLIT}/ours_${ITERATION}/renders"
fi

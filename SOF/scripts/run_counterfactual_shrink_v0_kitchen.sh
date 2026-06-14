#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

BASE_RUN_NAME="${BASE_RUN_NAME:-mip_to_soflr_surface_v0}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${BASE_RUN_NAME}}"
MODEL_PATH="${MODEL_PATH:-${BASE_RUN_ROOT}/pulled_mip_model}"
ITERATION="${ITERATION:-34000}"

DETECT_RUN_NAME="${DETECT_RUN_NAME:-${BASE_RUN_NAME}_starburst_mipref_v0}"
CANDIDATE_PAYLOAD_PATH="${CANDIDATE_PAYLOAD_PATH:-${SOF_ROOT}/output/starburst_gaussian_scores_v0/${SCENE_NAME}/${DETECT_RUN_NAME}/starburst_gaussian_scores_v0.pt}"
CANDIDATE_MASK_KEY="${CANDIDATE_MASK_KEY:-global_prefilter_candidate}"
CANDIDATE_SCORE_KEY="${CANDIDATE_SCORE_KEY:-global_prefilter_score}"

RUN_NAME="${RUN_NAME:-${BASE_RUN_NAME}_counterfactual_shrink_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/counterfactual_shrink_scores_v0/${SCENE_NAME}/${RUN_NAME}}"

INTERACTION_IMAGES_SUBDIR="${INTERACTION_IMAGES_SUBDIR:-images_2}"
CAMERA_RESOLUTION="${CAMERA_RESOLUTION:-4}"
REFERENCE_SOURCE="${REFERENCE_SOURCE:-mip_render}"
REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-}"
REFERENCE_IMAGE_DIR="${REFERENCE_IMAGE_DIR:-}"
MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k_render_no_gt_images_2}"
MIP_REFERENCE_ITERATION="${MIP_REFERENCE_ITERATION:-30000}"
MIP_REFERENCE_DIR="${MIP_REFERENCE_DIR:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}/test/ours_${MIP_REFERENCE_ITERATION}/renders}"

SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-4}"
VIEW_INDICES="${VIEW_INDICES:-}"
MAX_TEST_COUNT="${MAX_TEST_COUNT:-2048}"
MAX_TEST_FRACTION="${MAX_TEST_FRACTION:-0.0}"
GROUP_SIZE="${GROUP_SIZE:-1}"
SHRINK_FACTOR="${SHRINK_FACTOR:-0.5}"
SHRINK_AXIS_MODE="${SHRINK_AXIS_MODE:-uniform}"
LINE_LENGTHS="${LINE_LENGTHS:-9,17,31}"
ANGLES_DEG="${ANGLES_DEG:-0,22.5,45,67.5,90,112.5,135,157.5}"
HIGHPASS_KERNEL="${HIGHPASS_KERNEL:-21}"
RIDGE_NORM_PERCENTILE="${RIDGE_NORM_PERCENTILE:-0.995}"
RESIDUAL_NORM_PERCENTILE="${RESIDUAL_NORM_PERCENTILE:-0.990}"
MIN_ARTIFACT_IMPROVE="${MIN_ARTIFACT_IMPROVE:-0.002}"
MAX_PRESERVE_INCREASE="${MAX_PRESERVE_INCREASE:-0.0015}"
PRESERVE_PENALTY_WEIGHT="${PRESERVE_PENALTY_WEIGHT:-1.0}"
NUM_DEBUG_GROUPS="${NUM_DEBUG_GROUPS:-4}"

case "${REFERENCE_SOURCE}" in
  mip_render|mip_lr_render)
    REFERENCE_IMAGE_DIR="${REFERENCE_IMAGE_DIR:-${MIP_REFERENCE_DIR}}"
    ;;
  lr)
    REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_8}"
    ;;
  custom)
    ;;
  *)
    echo "[counterfactual-shrink-v0] unsupported REFERENCE_SOURCE=${REFERENCE_SOURCE}; use mip_render, lr, or custom" >&2
    exit 1
    ;;
esac

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

for path in "${SCENE_ROOT}" "${MODEL_PATH}" "${CANDIDATE_PAYLOAD_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[counterfactual-shrink-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ -n "${REFERENCE_IMAGES_SUBDIR}" && ! -d "${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}" ]]; then
  echo "[counterfactual-shrink-v0] reference images dir not found: ${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}" >&2
  exit 1
fi
if [[ -n "${REFERENCE_IMAGE_DIR}" && ! -d "${REFERENCE_IMAGE_DIR}" ]]; then
  echo "[counterfactual-shrink-v0] reference image dir not found: ${REFERENCE_IMAGE_DIR}" >&2
  exit 1
fi

echo "[counterfactual-shrink-v0] scene root     : ${SCENE_ROOT}"
echo "[counterfactual-shrink-v0] model path     : ${MODEL_PATH}"
echo "[counterfactual-shrink-v0] iteration      : ${ITERATION}"
echo "[counterfactual-shrink-v0] candidate      : ${CANDIDATE_PAYLOAD_PATH} key=${CANDIDATE_MASK_KEY} score=${CANDIDATE_SCORE_KEY}"
echo "[counterfactual-shrink-v0] reference      : source=${REFERENCE_SOURCE} subdir=${REFERENCE_IMAGES_SUBDIR:-<none>} dir=${REFERENCE_IMAGE_DIR:-<none>}"
echo "[counterfactual-shrink-v0] camera res     : ${CAMERA_RESOLUTION}"
echo "[counterfactual-shrink-v0] shrink         : factor=${SHRINK_FACTOR} axis=${SHRINK_AXIS_MODE} group=${GROUP_SIZE}"
echo "[counterfactual-shrink-v0] test/views     : max_test=${MAX_TEST_COUNT} frac=${MAX_TEST_FRACTION} split=${SPLIT} max_views=${MAX_VIEWS} indices=${VIEW_INDICES:-<uniform>}"
echo "[counterfactual-shrink-v0] thresholds     : artifact>=${MIN_ARTIFACT_IMPROVE} preserve<=${MAX_PRESERVE_INCREASE} penalty_w=${PRESERVE_PENALTY_WEIGHT}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/score_counterfactual_shrink_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --candidate_payload_path "${CANDIDATE_PAYLOAD_PATH}" \
  --candidate_mask_key "${CANDIDATE_MASK_KEY}" \
  --candidate_score_key "${CANDIDATE_SCORE_KEY}" \
  --reference_images_subdir "${REFERENCE_IMAGES_SUBDIR}" \
  --reference_image_dir "${REFERENCE_IMAGE_DIR}" \
  --interaction_images_subdir "${INTERACTION_IMAGES_SUBDIR}" \
  --camera_resolution "${CAMERA_RESOLUTION}" \
  --iteration "${ITERATION}" \
  --split "${SPLIT}" \
  --max_views "${MAX_VIEWS}" \
  --view_indices "${VIEW_INDICES}" \
  --max_test_count "${MAX_TEST_COUNT}" \
  --max_test_fraction "${MAX_TEST_FRACTION}" \
  --group_size "${GROUP_SIZE}" \
  --shrink_factor "${SHRINK_FACTOR}" \
  --shrink_axis_mode "${SHRINK_AXIS_MODE}" \
  --line_lengths "${LINE_LENGTHS}" \
  --angles_deg "${ANGLES_DEG}" \
  --highpass_kernel "${HIGHPASS_KERNEL}" \
  --ridge_norm_percentile "${RIDGE_NORM_PERCENTILE}" \
  --residual_norm_percentile "${RESIDUAL_NORM_PERCENTILE}" \
  --min_artifact_improve "${MIN_ARTIFACT_IMPROVE}" \
  --max_preserve_increase "${MAX_PRESERVE_INCREASE}" \
  --preserve_penalty_weight "${PRESERVE_PENALTY_WEIGHT}" \
  --num_debug_groups "${NUM_DEBUG_GROUPS}"

echo "[done] payload    : ${OUTPUT_DIR}/counterfactual_shrink_scores_v0.pt"
echo "[done] summary    : ${OUTPUT_DIR}/summary.json"
echo "[done] debug dirs : ${OUTPUT_DIR}/debug_groups"

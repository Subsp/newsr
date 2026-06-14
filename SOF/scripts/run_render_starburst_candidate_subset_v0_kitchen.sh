#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_RUN_NAME="${BASE_RUN_NAME:-mip_to_soflr_surface_v0}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${BASE_RUN_NAME}}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${BASE_RUN_ROOT}/pulled_mip_model}"
BASE_ITERATION="${BASE_ITERATION:-34000}"

CORRECTION_RUN_NAME="${CORRECTION_RUN_NAME:-${BASE_RUN_NAME}_starcorr_v0}"
CORRECTION_RUN_ROOT="${CORRECTION_RUN_ROOT:-${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${CORRECTION_RUN_NAME}}"
CORRECTION_MODEL_PATH="${CORRECTION_MODEL_PATH:-${CORRECTION_RUN_ROOT}/pulled_mip_model}"
CORRECTION_ITERATION="${CORRECTION_ITERATION:-35000}"

DETECT_RUN_NAME="${DETECT_RUN_NAME:-${BASE_RUN_NAME}_starburst_mipref_v0}"
STARBURST_SCORE_PAYLOAD="${STARBURST_SCORE_PAYLOAD:-${SOF_ROOT}/output/starburst_gaussian_scores_v0/${SCENE_NAME}/${DETECT_RUN_NAME}/starburst_gaussian_scores_v0.pt}"
STARBURST_MASK_KEY="${STARBURST_MASK_KEY:-starburst_candidate}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-8}"
MAKE_CONTACT_SHEETS="${MAKE_CONTACT_SHEETS:-1}"
CONTACT_MAX_IMAGES="${CONTACT_MAX_IMAGES:-8}"
CONTACT_COLUMNS="${CONTACT_COLUMNS:-4}"

RUN_BASE_PREVIEW="${RUN_BASE_PREVIEW:-1}"
RUN_CORRECTED_PREVIEW="${RUN_CORRECTED_PREVIEW:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/starburst_candidate_preview_v0/${SCENE_NAME}/${BASE_RUN_NAME}}"

for path in "${SCENE_ROOT}" "${BASE_MODEL_PATH}" "${STARBURST_SCORE_PAYLOAD}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[starburst-preview-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ "${RUN_CORRECTED_PREVIEW}" == "1" && ! -e "${CORRECTION_MODEL_PATH}/point_cloud/iteration_${CORRECTION_ITERATION}/point_cloud.ply" ]]; then
  echo "[starburst-preview-v0] corrected model point cloud not found: ${CORRECTION_MODEL_PATH}/point_cloud/iteration_${CORRECTION_ITERATION}/point_cloud.ply" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

render_subset() {
  local label="$1"
  local model_path="$2"
  local iteration="$3"
  local out_root="${OUTPUT_ROOT}/${label}"
  echo "[starburst-preview-v0] render ${label}: ${model_path} iter=${iteration}"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/export_gaussian_mask_subset_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${model_path}" \
    --iteration "${iteration}" \
    --mask_payload_path "${STARBURST_SCORE_PAYLOAD}" \
    --mask_key "${STARBURST_MASK_KEY}" \
    --output_root "${out_root}" \
    --images_subdir "${IMAGES_SUBDIR}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}"
  if [[ "${MAKE_CONTACT_SHEETS}" == "1" ]]; then
    local render_dir="${out_root}/${SPLIT}/ours_${iteration}/renders"
    local sheet_path="${out_root}/contact_sheet_${IMAGES_SUBDIR}_${SPLIT}_${iteration}.png"
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/make_render_contact_sheet.py" \
      --render_dir "${render_dir}" \
      --output_path "${sheet_path}" \
      --max_images "${CONTACT_MAX_IMAGES}" \
      --columns "${CONTACT_COLUMNS}"
  fi
}

if [[ "${RUN_BASE_PREVIEW}" == "1" ]]; then
  render_subset "base" "${BASE_MODEL_PATH}" "${BASE_ITERATION}"
fi

if [[ "${RUN_CORRECTED_PREVIEW}" == "1" ]]; then
  render_subset "corrected" "${CORRECTION_MODEL_PATH}" "${CORRECTION_ITERATION}"
fi

echo
if [[ "${RUN_BASE_PREVIEW}" == "1" ]]; then
  echo "[done] base preview root      : ${OUTPUT_ROOT}/base"
  echo "[done] base subset model      : ${OUTPUT_ROOT}/base/subset_model"
  echo "[done] base candidate renders : ${OUTPUT_ROOT}/base/${SPLIT}/ours_${BASE_ITERATION}/renders"
  if [[ "${MAKE_CONTACT_SHEETS}" == "1" ]]; then
    echo "[done] base contact sheet     : ${OUTPUT_ROOT}/base/contact_sheet_${IMAGES_SUBDIR}_${SPLIT}_${BASE_ITERATION}.png"
  fi
fi
if [[ "${RUN_CORRECTED_PREVIEW}" == "1" ]]; then
  echo "[done] corrected preview root      : ${OUTPUT_ROOT}/corrected"
  echo "[done] corrected subset model      : ${OUTPUT_ROOT}/corrected/subset_model"
  echo "[done] corrected candidate renders : ${OUTPUT_ROOT}/corrected/${SPLIT}/ours_${CORRECTION_ITERATION}/renders"
  if [[ "${MAKE_CONTACT_SHEETS}" == "1" ]]; then
    echo "[done] corrected contact sheet     : ${OUTPUT_ROOT}/corrected/contact_sheet_${IMAGES_SUBDIR}_${SPLIT}_${CORRECTION_ITERATION}.png"
  fi
fi

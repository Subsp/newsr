#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PREPARE_DEBUG_MODEL_PATH="${PREPARE_DEBUG_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k_sof_native_input_init_early4ksoft_v1_debug}"
PREPARE_DEBUG_ITERATION="${PREPARE_DEBUG_ITERATION:-34000}"
STAGE_NAME="${STAGE_NAME:-debug_stage_00_after_finite_aabb}"
MODEL_PATH="${MODEL_PATH:-${PREPARE_DEBUG_MODEL_PATH}/debug_prepare_stages/${STAGE_NAME}}"
ITERATION="${ITERATION:-${PREPARE_DEBUG_ITERATION}}"

RUN_NAME="${RUN_NAME:-mesh_delta_star_${STAGE_NAME}_v0}"
PAYLOAD_ROOT="${PAYLOAD_ROOT:-${SOF_ROOT}/output/mesh_delta_star_gaussian_probe_v0/${SCENE_NAME}/${RUN_NAME}}"
MASK_PAYLOAD_PATH="${MASK_PAYLOAD_PATH:-${PAYLOAD_ROOT}/mesh_delta_star_gaussian_candidates_v0.pt}"
MASK_KEY="${MASK_KEY:-candidate_mask}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${RUN_NAME}_${MASK_KEY}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/mesh_delta_star_candidate_preview_v0/${SCENE_NAME}/${OUTPUT_RUN_NAME}}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-8}"
MAKE_CONTACT_SHEETS="${MAKE_CONTACT_SHEETS:-1}"
CONTACT_MAX_IMAGES="${CONTACT_MAX_IMAGES:-8}"
CONTACT_COLUMNS="${CONTACT_COLUMNS:-4}"

RUN_SUBSET_ONLY="${RUN_SUBSET_ONLY:-1}"
RUN_REMOVED_FULL="${RUN_REMOVED_FULL:-0}"
RUN_RESTSH_ZERO_FULL="${RUN_RESTSH_ZERO_FULL:-0}"
RUN_TAU_HALF_FULL="${RUN_TAU_HALF_FULL:-0}"
RUN_MAJOR_SHRINK_FULL="${RUN_MAJOR_SHRINK_FULL:-0}"

echo "[mesh-delta-preview-v0] scene      : ${SCENE_ROOT}"
echo "[mesh-delta-preview-v0] model      : ${MODEL_PATH} iter=${ITERATION}"
echo "[mesh-delta-preview-v0] payload    : ${MASK_PAYLOAD_PATH} key=${MASK_KEY}"
echo "[mesh-delta-preview-v0] output root: ${OUTPUT_ROOT}"

for path in "${SCENE_ROOT}" "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" "${MASK_PAYLOAD_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[mesh-delta-preview-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"

make_sheet() {
  local render_dir="$1"
  local sheet_path="$2"
  if [[ "${MAKE_CONTACT_SHEETS}" == "1" ]]; then
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/make_render_contact_sheet.py" \
      --render_dir "${render_dir}" \
      --output_path "${sheet_path}" \
      --max_images "${CONTACT_MAX_IMAGES}" \
      --columns "${CONTACT_COLUMNS}"
  fi
}

run_subset_only() {
  local out_root="${OUTPUT_ROOT}/candidate_only"
  echo "[mesh-delta-preview-v0] render candidate_only"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/export_gaussian_mask_subset_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --iteration "${ITERATION}" \
    --mask_payload_path "${MASK_PAYLOAD_PATH}" \
    --mask_key "${MASK_KEY}" \
    --output_root "${out_root}" \
    --images_subdir "${IMAGES_SUBDIR}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}"
  make_sheet \
    "${out_root}/${SPLIT}/ours_${ITERATION}/renders" \
    "${out_root}/contact_sheet_${IMAGES_SUBDIR}_${SPLIT}_${ITERATION}.png"
}

run_full_variant() {
  local label="$1"
  local selection_mode="$2"
  local rest_scale="$3"
  local tau_scale="$4"
  local scale_multiplier="$5"
  local scale_axis_mode="$6"
  local out_root="${OUTPUT_ROOT}/${label}"
  echo "[mesh-delta-preview-v0] render ${label}"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/export_gaussian_group_variant_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --iteration "${ITERATION}" \
    --output_root "${out_root}" \
    --images_subdir "${IMAGES_SUBDIR}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}" \
    --selection_source payload \
    --selection_key "${MASK_KEY}" \
    --mask_payload_path "${MASK_PAYLOAD_PATH}" \
    --selection_mode "${selection_mode}" \
    --rest_scale "${rest_scale}" \
    --tau_scale "${tau_scale}" \
    --scale_multiplier "${scale_multiplier}" \
    --scale_axis_mode "${scale_axis_mode}"
  make_sheet \
    "${out_root}/${SPLIT}/ours_${ITERATION}/renders" \
    "${out_root}/contact_sheet_${IMAGES_SUBDIR}_${SPLIT}_${ITERATION}.png"
}

if [[ "${RUN_SUBSET_ONLY}" == "1" ]]; then
  run_subset_only
fi

if [[ "${RUN_REMOVED_FULL}" == "1" ]]; then
  run_full_variant "candidate_removed_full" "selected_removed" "1.0" "1.0" "1.0" "all"
fi

if [[ "${RUN_RESTSH_ZERO_FULL}" == "1" ]]; then
  run_full_variant "candidate_restsh_zero_full" "full" "0.0" "1.0" "1.0" "all"
fi

if [[ "${RUN_TAU_HALF_FULL}" == "1" ]]; then
  run_full_variant "candidate_tau_half_full" "full" "1.0" "0.5" "1.0" "all"
fi

if [[ "${RUN_MAJOR_SHRINK_FULL}" == "1" ]]; then
  run_full_variant "candidate_major_shrink_half_full" "full" "1.0" "1.0" "0.5" "major_only"
fi

echo
echo "[done] output root       : ${OUTPUT_ROOT}"
if [[ "${RUN_SUBSET_ONLY}" == "1" ]]; then
  echo "[done] candidate renders : ${OUTPUT_ROOT}/candidate_only/${SPLIT}/ours_${ITERATION}/renders"
  echo "[done] candidate sheet   : ${OUTPUT_ROOT}/candidate_only/contact_sheet_${IMAGES_SUBDIR}_${SPLIT}_${ITERATION}.png"
fi
if [[ "${RUN_REMOVED_FULL}" == "1" ]]; then
  echo "[done] removed renders   : ${OUTPUT_ROOT}/candidate_removed_full/${SPLIT}/ours_${ITERATION}/renders"
fi
if [[ "${RUN_RESTSH_ZERO_FULL}" == "1" ]]; then
  echo "[done] restsh0 renders   : ${OUTPUT_ROOT}/candidate_restsh_zero_full/${SPLIT}/ours_${ITERATION}/renders"
fi
if [[ "${RUN_TAU_HALF_FULL}" == "1" ]]; then
  echo "[done] tau half renders  : ${OUTPUT_ROOT}/candidate_tau_half_full/${SPLIT}/ours_${ITERATION}/renders"
fi
if [[ "${RUN_MAJOR_SHRINK_FULL}" == "1" ]]; then
  echo "[done] shrink renders    : ${OUTPUT_ROOT}/candidate_major_shrink_half_full/${SPLIT}/ours_${ITERATION}/renders"
fi

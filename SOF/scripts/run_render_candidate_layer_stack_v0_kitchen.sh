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
STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
MODEL_PATH="${MODEL_PATH:-${PREPARE_DEBUG_MODEL_PATH}/debug_prepare_stages/${STAGE_NAME}}"
ITERATION="${ITERATION:-${PREPARE_DEBUG_ITERATION}}"

RUN_NAME="${RUN_NAME:-single_mesh_offsurface_farthest_small_more_brightq55_${STAGE_NAME}_v0}"
PAYLOAD_ROOT="${PAYLOAD_ROOT:-${SOF_ROOT}/output/single_mesh_tangent_gaussian_probe_v0/${SCENE_NAME}/${RUN_NAME}}"
SOURCE_PAYLOAD_PATH="${SOURCE_PAYLOAD_PATH:-${PAYLOAD_ROOT}/mesh_delta_star_gaussian_candidates_v0.pt}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${RUN_NAME}_layer_stack}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/candidate_layer_stack_v0/${SCENE_NAME}/${OUTPUT_RUN_NAME}}"
LAYER_PAYLOAD_PATH="${LAYER_PAYLOAD_PATH:-${OUTPUT_ROOT}/layer_payload/gaussian_candidate_layers_v0.pt}"

GEOMETRY_MASK_KEY="${GEOMETRY_MASK_KEY:-geometry_candidate_mask}"
BRIGHT_MASK_KEY="${BRIGHT_MASK_KEY:-base_candidate_mask}"
RESIDUAL_MASK_KEY="${RESIDUAL_MASK_KEY:-geometry_minus_bright_mask}"
UNION_MASK_KEY="${UNION_MASK_KEY:-geometry_or_bright_mask}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-8}"
MAKE_CONTACT_SHEETS="${MAKE_CONTACT_SHEETS:-1}"
CONTACT_MAX_IMAGES="${CONTACT_MAX_IMAGES:-8}"
CONTACT_COLUMNS="${CONTACT_COLUMNS:-4}"
LAYER_SHEET_THUMB_WIDTH="${LAYER_SHEET_THUMB_WIDTH:-360}"

RUN_GEOMETRY="${RUN_GEOMETRY:-1}"
RUN_BRIGHT="${RUN_BRIGHT:-1}"
RUN_RESIDUAL="${RUN_RESIDUAL:-1}"
RUN_UNION="${RUN_UNION:-0}"

echo "[candidate-layer-stack-v0] scene      : ${SCENE_ROOT}"
echo "[candidate-layer-stack-v0] model      : ${MODEL_PATH} iter=${ITERATION}"
echo "[candidate-layer-stack-v0] source     : ${SOURCE_PAYLOAD_PATH}"
echo "[candidate-layer-stack-v0] output root: ${OUTPUT_ROOT}"
echo "[candidate-layer-stack-v0] masks      : geometry=${GEOMETRY_MASK_KEY} bright=${BRIGHT_MASK_KEY} residual=${RESIDUAL_MASK_KEY} union=${UNION_MASK_KEY}"

for path in "${SCENE_ROOT}" "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" "${SOURCE_PAYLOAD_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[candidate-layer-stack-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LAYER_PAYLOAD_PATH}")"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/build_gaussian_mask_layers_v0.py" \
  --input_payload "${SOURCE_PAYLOAD_PATH}" \
  --output_payload "${LAYER_PAYLOAD_PATH}" \
  --summary_path "${OUTPUT_ROOT}/layer_payload/summary.json" \
  --geometry_key "${GEOMETRY_MASK_KEY}" \
  --bright_key "${BRIGHT_MASK_KEY}" \
  --residual_key "${RESIDUAL_MASK_KEY}" \
  --union_key "${UNION_MASK_KEY}"

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

render_layer() {
  local label="$1"
  local mask_key="$2"
  local out_root="${OUTPUT_ROOT}/${label}"
  echo "[candidate-layer-stack-v0] render ${label} key=${mask_key}"
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/export_gaussian_mask_subset_v0.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --iteration "${ITERATION}" \
    --mask_payload_path "${LAYER_PAYLOAD_PATH}" \
    --mask_key "${mask_key}" \
    --output_root "${out_root}" \
    --images_subdir "${IMAGES_SUBDIR}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}"
  make_sheet \
    "${out_root}/${SPLIT}/ours_${ITERATION}/renders" \
    "${out_root}/contact_sheet_${IMAGES_SUBDIR}_${SPLIT}_${ITERATION}.png"
}

layer_sheet_args=()
if [[ "${RUN_GEOMETRY}" == "1" ]]; then
  render_layer "geometry_layer" "${GEOMETRY_MASK_KEY}"
  layer_sheet_args+=("--layer" "geometry=${OUTPUT_ROOT}/geometry_layer/${SPLIT}/ours_${ITERATION}/renders")
fi
if [[ "${RUN_BRIGHT}" == "1" ]]; then
  render_layer "bright_layer" "${BRIGHT_MASK_KEY}"
  layer_sheet_args+=("--layer" "bright=${OUTPUT_ROOT}/bright_layer/${SPLIT}/ours_${ITERATION}/renders")
fi
if [[ "${RUN_RESIDUAL}" == "1" ]]; then
  render_layer "residual_geometry_minus_bright" "${RESIDUAL_MASK_KEY}"
  layer_sheet_args+=("--layer" "residual=${OUTPUT_ROOT}/residual_geometry_minus_bright/${SPLIT}/ours_${ITERATION}/renders")
fi
if [[ "${RUN_UNION}" == "1" ]]; then
  render_layer "union_geometry_or_bright" "${UNION_MASK_KEY}"
  layer_sheet_args+=("--layer" "union=${OUTPUT_ROOT}/union_geometry_or_bright/${SPLIT}/ours_${ITERATION}/renders")
fi

if [[ "${#layer_sheet_args[@]}" -gt 0 ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/make_render_layer_sheet_v0.py" \
    "${layer_sheet_args[@]}" \
    --output_path "${OUTPUT_ROOT}/contact_sheet_layers_${IMAGES_SUBDIR}_${SPLIT}_${ITERATION}.png" \
    --max_images "${CONTACT_MAX_IMAGES}" \
    --thumb_width "${LAYER_SHEET_THUMB_WIDTH}"
fi

echo
echo "[done] output root    : ${OUTPUT_ROOT}"
echo "[done] layer payload  : ${LAYER_PAYLOAD_PATH}"
echo "[done] layer summary  : ${OUTPUT_ROOT}/layer_payload/summary.json"
echo "[done] layer sheet    : ${OUTPUT_ROOT}/contact_sheet_layers_${IMAGES_SUBDIR}_${SPLIT}_${ITERATION}.png"

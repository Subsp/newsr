#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-${SOF_ROOT}/output/anneal_mip_covariance_to_surface_v0/${SCENE_NAME}/mip30k_volume_stress_iterclean_v0_conservative_v0_puremip_surfaceanneal_v0}"
MODEL_ITERATION="${MODEL_ITERATION:--1}"
MESH_PATH="${MESH_PATH:-/root/autodl-tmp/kitchen_MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_15_cleaned_mesh.ply}"

PREPARE_RUN_NAME="${PREPARE_RUN_NAME:-mesh_depth_tsdf_regulation_v0_puremip_surfaceanneal}"
PREPARE_OUTPUT_DIR="${PREPARE_OUTPUT_DIR:-${SOF_ROOT}/output/mesh_depth_tsdf_regulation_v0/${SCENE_NAME}/${PREPARE_RUN_NAME}}"
PAYLOAD_PATH="${PAYLOAD_PATH:-${PREPARE_OUTPUT_DIR}/mesh_depth_tsdf_regulation_v0.pt}"
RUN_PREPARE="${RUN_PREPARE:-1}"

APPLY_RUN_NAME="${APPLY_RUN_NAME:-mesh_depth_tsdf_apply_v0_puremip_surfaceanneal}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/apply_mesh_depth_tsdf_regulation_v0/${SCENE_NAME}/${APPLY_RUN_NAME}/regulated_mip_model_v0}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:--1}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-16}"
PREVIEW_VIEWS="${PREVIEW_VIEWS:-8}"

MESH_DEPTH_MODE="${MESH_DEPTH_MODE:-raycast}"
RAYCAST_DOWNSAMPLE="${RAYCAST_DOWNSAMPLE:-4}"
RAYCAST_CHUNK="${RAYCAST_CHUNK:-262144}"
MIN_SUPPORT_VIEWS="${MIN_SUPPORT_VIEWS:-2}"
SUPPORT_SAMPLE_SCALE="${SUPPORT_SAMPLE_SCALE:-1.0}"
RENDER_DEBUG_VIEWS="${RENDER_DEBUG_VIEWS:-6}"

SUPPRESS_STRENGTH="${SUPPRESS_STRENGTH:-0.80}"
MIN_SUPPRESS_APPLY_WEIGHT="${MIN_SUPPRESS_APPLY_WEIGHT:-0.35}"
RESIDUAL_KEEP_SUPPRESS_PROTECT="${RESIDUAL_KEEP_SUPPRESS_PROTECT:-0.85}"
COV_FLATTEN_STRENGTH="${COV_FLATTEN_STRENGTH:-0.55}"
CENTER_ATTACH_STRENGTH="${CENTER_ATTACH_STRENGTH:-0.45}"

echo "[mesh-depth-tsdf-apply-v0] scene        : ${SCENE_ROOT}"
echo "[mesh-depth-tsdf-apply-v0] model        : ${MODEL_PATH} iter=${MODEL_ITERATION}"
echo "[mesh-depth-tsdf-apply-v0] mesh         : ${MESH_PATH}"
echo "[mesh-depth-tsdf-apply-v0] payload      : ${PAYLOAD_PATH}"
echo "[mesh-depth-tsdf-apply-v0] output model : ${OUTPUT_MODEL_PATH}"
echo "[mesh-depth-tsdf-apply-v0] images       : ${IMAGES_SUBDIR} split=${SPLIT} max_views=${MAX_VIEWS}"
echo "[mesh-depth-tsdf-apply-v0] strengths    : suppress=${SUPPRESS_STRENGTH} cov=${COV_FLATTEN_STRENGTH} center=${CENTER_ATTACH_STRENGTH}"
echo "[mesh-depth-tsdf-apply-v0] suppress gate: min=${MIN_SUPPRESS_APPLY_WEIGHT} residual_protect=${RESIDUAL_KEEP_SUPPRESS_PROTECT}"

if [[ "${RUN_PREPARE}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/prepare_mesh_depth_tsdf_regulation_v0.py" \
    --model_path "${MODEL_PATH}" \
    --mesh_path "${MESH_PATH}" \
    --scene_root "${SCENE_ROOT}" \
    --output_dir "${PREPARE_OUTPUT_DIR}" \
    --iteration "${MODEL_ITERATION}" \
    --images_subdir "${IMAGES_SUBDIR}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}" \
    --mesh_depth_mode "${MESH_DEPTH_MODE}" \
    --raycast_downsample "${RAYCAST_DOWNSAMPLE}" \
    --raycast_chunk "${RAYCAST_CHUNK}" \
    --support_sample_scale "${SUPPORT_SAMPLE_SCALE}" \
    --min_support_views "${MIN_SUPPORT_VIEWS}" \
    --render_debug_views "${RENDER_DEBUG_VIEWS}"
fi

if [[ ! -f "${PAYLOAD_PATH}" ]]; then
  echo "[mesh-depth-tsdf-apply-v0] missing payload: ${PAYLOAD_PATH}" >&2
  echo "[mesh-depth-tsdf-apply-v0] set RUN_PREPARE=1 or pass PAYLOAD_PATH." >&2
  exit 1
fi

"${PYTHON_BIN}" -u "${SOF_ROOT}/apply_mesh_depth_tsdf_regulation_v0.py" \
  --model_path "${MODEL_PATH}" \
  --iteration "${MODEL_ITERATION}" \
  --payload_path "${PAYLOAD_PATH}" \
  --output_model_path "${OUTPUT_MODEL_PATH}" \
  --output_iteration "${OUTPUT_ITERATION}" \
  --scene_root "${SCENE_ROOT}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --split "${SPLIT}" \
  --preview_views "${PREVIEW_VIEWS}" \
  --suppress_strength "${SUPPRESS_STRENGTH}" \
  --min_suppress_apply_weight "${MIN_SUPPRESS_APPLY_WEIGHT}" \
  --residual_keep_suppress_protect "${RESIDUAL_KEEP_SUPPRESS_PROTECT}" \
  --cov_flatten_strength "${COV_FLATTEN_STRENGTH}" \
  --center_attach_strength "${CENTER_ATTACH_STRENGTH}"

echo "[done] output model : ${OUTPUT_MODEL_PATH}"
echo "[done] output ply   : ${OUTPUT_MODEL_PATH}/point_cloud"
echo "[done] renders      : ${OUTPUT_MODEL_PATH}/mesh_depth_tsdf_apply_previews_v0/render_previews"
echo "[done] overview     : ${OUTPUT_MODEL_PATH}/mesh_depth_tsdf_apply_previews_v0/render_previews/comparison_overview_v0.png"
echo "[done] summary      : ${OUTPUT_MODEL_PATH}/mesh_depth_tsdf_regulation_apply_v0_summary.json"

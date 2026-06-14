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

RUN_NAME="${RUN_NAME:-mesh_depth_tsdf_regulation_v0_puremip_surfaceanneal}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/mesh_depth_tsdf_regulation_v0/${SCENE_NAME}/${RUN_NAME}}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-16}"
RENDER_DEBUG_VIEWS="${RENDER_DEBUG_VIEWS:-6}"
RENDER_DEBUG_GRID_COLUMNS="${RENDER_DEBUG_GRID_COLUMNS:-3}"

MESH_DEPTH_MODE="${MESH_DEPTH_MODE:-raycast}"
RAYCAST_DOWNSAMPLE="${RAYCAST_DOWNSAMPLE:-4}"
RAYCAST_CHUNK="${RAYCAST_CHUNK:-262144}"
MIN_SUPPORT_VIEWS="${MIN_SUPPORT_VIEWS:-2}"
SUPPORT_SAMPLE_SCALE="${SUPPORT_SAMPLE_SCALE:-1.0}"

echo "[mesh-depth-tsdf-reg-v0] scene   : ${SCENE_ROOT}"
echo "[mesh-depth-tsdf-reg-v0] model   : ${MODEL_PATH} iter=${MODEL_ITERATION}"
echo "[mesh-depth-tsdf-reg-v0] mesh    : ${MESH_PATH}"
echo "[mesh-depth-tsdf-reg-v0] output  : ${OUTPUT_DIR}"
echo "[mesh-depth-tsdf-reg-v0] views   : split=${SPLIT} images=${IMAGES_SUBDIR} max=${MAX_VIEWS}"
echo "[mesh-depth-tsdf-reg-v0] render  : debug_views=${RENDER_DEBUG_VIEWS}"
echo "[mesh-depth-tsdf-reg-v0] depth   : mode=${MESH_DEPTH_MODE} downsample=${RAYCAST_DOWNSAMPLE}"
echo "[mesh-depth-tsdf-reg-v0] samples : support_sample_scale=${SUPPORT_SAMPLE_SCALE}"

"${PYTHON_BIN}" -u "${SOF_ROOT}/prepare_mesh_depth_tsdf_regulation_v0.py" \
  --model_path "${MODEL_PATH}" \
  --mesh_path "${MESH_PATH}" \
  --scene_root "${SCENE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --iteration "${MODEL_ITERATION}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --split "${SPLIT}" \
  --max_views "${MAX_VIEWS}" \
  --mesh_depth_mode "${MESH_DEPTH_MODE}" \
  --raycast_downsample "${RAYCAST_DOWNSAMPLE}" \
  --raycast_chunk "${RAYCAST_CHUNK}" \
  --support_sample_scale "${SUPPORT_SAMPLE_SCALE}" \
  --min_support_views "${MIN_SUPPORT_VIEWS}" \
  --render_debug_views "${RENDER_DEBUG_VIEWS}" \
  --render_debug_grid_columns "${RENDER_DEBUG_GRID_COLUMNS}"

echo "[done] payload : ${OUTPUT_DIR}/mesh_depth_tsdf_regulation_v0.pt"
echo "[done] summary : ${OUTPUT_DIR}/mesh_depth_tsdf_regulation_v0_summary.json"
echo "[done] previews: ${OUTPUT_DIR}/point_cloud_previews"
echo "[done] renders : ${OUTPUT_DIR}/render_debug_v0"

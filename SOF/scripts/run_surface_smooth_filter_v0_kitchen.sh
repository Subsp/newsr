#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"

START_RUN_NAME="${START_RUN_NAME:-debug_stage_00b3_after_scale_canonicalize_geometry_only_v0}"
MODEL_PATH="${MODEL_PATH:-${SOF_ROOT}/output/mask_guided_reparameterization_v0/${SCENE_NAME}/${START_RUN_NAME}}"
ITERATION="${ITERATION:-34000}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
MESH_PATH="${MESH_PATH:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"

MODE="${MODE:-delta_sigma}"
RUN_TAG="${RUN_TAG:-surface_smooth_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/surface_smooth_filter_v0/${SCENE_NAME}}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${START_RUN_NAME}_${MODE}_${RUN_TAG}}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
MAX_CAMERA_VIEWS="${MAX_CAMERA_VIEWS:-64}"
FACE_K="${FACE_K:-8}"
BIND_CHUNK_SIZE="${BIND_CHUNK_SIZE:-16384}"

K_PX="${K_PX:-2.5}"
K_EDGE="${K_EDGE:-0.4}"
TAU_FLOOR_SCENE_SCALE="${TAU_FLOOR_SCENE_SCALE:-1e-5}"
MIN_SURFACE_CONF="${MIN_SURFACE_CONF:-0.25}"
MAX_D_NORM="${MAX_D_NORM:-3.0}"

K_NEIGHBORS="${K_NEIGHBORS:-12}"
RADIUS_PX="${RADIUS_PX:-3.0}"
NORMAL_ANGLE_CUT_DEG="${NORMAL_ANGLE_CUT_DEG:-30}"
SIGMA_ANGLE_DEG="${SIGMA_ANGLE_DEG:-15}"
SIGMA_COLOR="${SIGMA_COLOR:-0.15}"
REQUIRE_FACE_ADJACENCY="${REQUIRE_FACE_ADJACENCY:-1}"

SMOOTH_DELTA_LAMBDA="${SMOOTH_DELTA_LAMBDA:-0.4}"
SMOOTH_SIGMA_LAMBDA="${SMOOTH_SIGMA_LAMBDA:-0.4}"
MAX_DELTA_UPDATE_PX="${MAX_DELTA_UPDATE_PX:-1.0}"
SIGMA_NORMAL_CAP_PX="${SIGMA_NORMAL_CAP_PX:-0.75}"
ENABLE_TAU_CAP="${ENABLE_TAU_CAP:-0}"
TAU_CAP_MAD_K="${TAU_CAP_MAD_K:-3.0}"

ENABLE_BODY_PROBE="${ENABLE_BODY_PROBE:-1}"
BODY_PROBE_SAMPLES="${BODY_PROBE_SAMPLES:--1.0,-0.5,0.0,0.5,1.0}"
BODY_MAJOR_SCALE="${BODY_MAJOR_SCALE:-1.0}"
BODY_FACE_K="${BODY_FACE_K:-0}"
BODY_VALID_RATIO_THRESHOLD="${BODY_VALID_RATIO_THRESHOLD:-0.8}"
BODY_D_NORM_THRESHOLD="${BODY_D_NORM_THRESHOLD:-2.0}"
BODY_ENDPOINT_D_NORM_THRESHOLD="${BODY_ENDPOINT_D_NORM_THRESHOLD:-2.0}"
BODY_NORMAL_ANGLE_CUT_DEG="${BODY_NORMAL_ANGLE_CUT_DEG:-30}"
BODY_SIGMA_NORM_THRESHOLD="${BODY_SIGMA_NORM_THRESHOLD:-0.0}"
BODY_ALL_INVALID_RATIO="${BODY_ALL_INVALID_RATIO:-0.2}"
BODY_MAX_NORMAL_SWITCH_COUNT="${BODY_MAX_NORMAL_SWITCH_COUNT:-0}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[surface-smooth-filter-v0] missing model: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[surface-smooth-filter-v0] missing mesh: ${MESH_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_PATH}"

echo "[surface-smooth-filter-v0] scene     : ${SCENE_ROOT}"
echo "[surface-smooth-filter-v0] model     : ${MODEL_PATH}"
echo "[surface-smooth-filter-v0] iteration : ${ITERATION}"
echo "[surface-smooth-filter-v0] mesh      : ${MESH_PATH}"
echo "[surface-smooth-filter-v0] mode      : ${MODE}"
echo "[surface-smooth-filter-v0] body probe: ${ENABLE_BODY_PROBE} samples=${BODY_PROBE_SAMPLES}"
echo "[surface-smooth-filter-v0] output    : ${OUTPUT_PATH}"

python -u "${SOF_ROOT}/scripts/smooth_surface_bound_gaussians_v0.py" \
  --model_path "${MODEL_PATH}" \
  --mesh_path "${MESH_PATH}" \
  --iteration "${ITERATION}" \
  --output_path "${OUTPUT_PATH}" \
  --scene_root "${SCENE_ROOT}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --mode "${MODE}" \
  --face_k "${FACE_K}" \
  --bind_chunk_size "${BIND_CHUNK_SIZE}" \
  --max_camera_views "${MAX_CAMERA_VIEWS}" \
  --k_px "${K_PX}" \
  --k_edge "${K_EDGE}" \
  --tau_floor_scene_scale "${TAU_FLOOR_SCENE_SCALE}" \
  --min_surface_conf "${MIN_SURFACE_CONF}" \
  --max_d_norm "${MAX_D_NORM}" \
  --k_neighbors "${K_NEIGHBORS}" \
  --radius_px "${RADIUS_PX}" \
  --normal_angle_cut_deg "${NORMAL_ANGLE_CUT_DEG}" \
  --sigma_angle_deg "${SIGMA_ANGLE_DEG}" \
  --sigma_color "${SIGMA_COLOR}" \
  --require_face_adjacency "${REQUIRE_FACE_ADJACENCY}" \
  --smooth_delta_lambda "${SMOOTH_DELTA_LAMBDA}" \
  --smooth_sigma_lambda "${SMOOTH_SIGMA_LAMBDA}" \
  --max_delta_update_px "${MAX_DELTA_UPDATE_PX}" \
  --sigma_normal_cap_px "${SIGMA_NORMAL_CAP_PX}" \
  --enable_tau_cap "${ENABLE_TAU_CAP}" \
  --tau_cap_mad_k "${TAU_CAP_MAD_K}" \
  --enable_body_probe "${ENABLE_BODY_PROBE}" \
  --body_probe_samples="${BODY_PROBE_SAMPLES}" \
  --body_major_scale "${BODY_MAJOR_SCALE}" \
  --body_face_k "${BODY_FACE_K}" \
  --body_valid_ratio_threshold "${BODY_VALID_RATIO_THRESHOLD}" \
  --body_d_norm_threshold "${BODY_D_NORM_THRESHOLD}" \
  --body_endpoint_d_norm_threshold "${BODY_ENDPOINT_D_NORM_THRESHOLD}" \
  --body_normal_angle_cut_deg "${BODY_NORMAL_ANGLE_CUT_DEG}" \
  --body_sigma_norm_threshold "${BODY_SIGMA_NORM_THRESHOLD}" \
  --body_all_invalid_ratio "${BODY_ALL_INVALID_RATIO}" \
  --body_max_normal_switch_count "${BODY_MAX_NORMAL_SWITCH_COUNT}"

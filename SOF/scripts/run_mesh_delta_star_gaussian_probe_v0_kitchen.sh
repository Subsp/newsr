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
STAGE_MODEL_PATH="${STAGE_MODEL_PATH:-${PREPARE_DEBUG_MODEL_PATH}/debug_prepare_stages/${STAGE_NAME}}"
STAGE_POINT_CLOUD="${STAGE_POINT_CLOUD:-${STAGE_MODEL_PATH}/point_cloud/iteration_${PREPARE_DEBUG_ITERATION}/point_cloud.ply}"

MESH_COMPARE_ROOT="${MESH_COMPARE_ROOT:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}}"
RAW_MESH_PATH="${RAW_MESH_PATH:-${MESH_COMPARE_ROOT}/raw_mip_raw_mip_sof_export_mesh_v0_7.ply}"
STAGE_MESH_PATH="${STAGE_MESH_PATH:-${MESH_COMPARE_ROOT}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"

RUN_NAME="${RUN_NAME:-mesh_delta_star_${STAGE_NAME}_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/mesh_delta_star_gaussian_probe_v0/${SCENE_NAME}/${RUN_NAME}}"

MAX_RAW_REFERENCE_VERTICES="${MAX_RAW_REFERENCE_VERTICES:-1000000}"
MAX_STAGE_QUERY_VERTICES="${MAX_STAGE_QUERY_VERTICES:-2000000}"
MAX_DELTA_POINTS="${MAX_DELTA_POINTS:-500000}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-262144}"
MESH_DELTA_QUANTILE="${MESH_DELTA_QUANTILE:-99.0}"
MESH_DELTA_DISTANCE_THRESHOLD="${MESH_DELTA_DISTANCE_THRESHOLD:-0.0}"
MESH_DELTA_DISTANCE_FLOOR="${MESH_DELTA_DISTANCE_FLOOR:-0.08}"
DELTA_RADIUS_ABS="${DELTA_RADIUS_ABS:-0.05}"
DELTA_RADIUS_SCALE="${DELTA_RADIUS_SCALE:-1.25}"
DELTA_RADIUS_MODE="${DELTA_RADIUS_MODE:-scale_max}"
GAUSSIAN_DISTANCE_MODE="${GAUSSIAN_DISTANCE_MODE:-center}"
MAJOR_ENDPOINT_SCALE="${MAJOR_ENDPOINT_SCALE:-1.0}"
TANGENT_ANGLE_MODE="${TANGENT_ANGLE_MODE:-none}"
TANGENT_ANGLE_K="${TANGENT_ANGLE_K:-12}"
TANGENT_ANGLE_DISTANCE_FLOOR="${TANGENT_ANGLE_DISTANCE_FLOOR:-0.15}"
MIN_OPACITY="${MIN_OPACITY:-0.02}"
MIN_ANISOTROPY="${MIN_ANISOTROPY:-2.0}"
MIN_CANDIDATE_SCORE="${MIN_CANDIDATE_SCORE:-0.05}"
CANDIDATE_SCORE_QUANTILE="${CANDIDATE_SCORE_QUANTILE:-99.0}"
MAX_CANDIDATES="${MAX_CANDIDATES:-200000}"
PREVIEW_MAX_POINTS="${PREVIEW_MAX_POINTS:-200000}"

echo "[mesh-delta-star-v0] scene       : ${SCENE_ROOT}"
echo "[mesh-delta-star-v0] raw mesh    : ${RAW_MESH_PATH}"
echo "[mesh-delta-star-v0] stage mesh  : ${STAGE_MESH_PATH}"
echo "[mesh-delta-star-v0] stage ply   : ${STAGE_POINT_CLOUD}"
echo "[mesh-delta-star-v0] output dir  : ${OUTPUT_DIR}"

for path in "${RAW_MESH_PATH}" "${STAGE_MESH_PATH}" "${STAGE_POINT_CLOUD}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[mesh-delta-star-v0] required file not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/score_mesh_delta_star_gaussians_v0.py" \
  --raw_mesh_path "${RAW_MESH_PATH}" \
  --stage_mesh_path "${STAGE_MESH_PATH}" \
  --stage_point_cloud_ply "${STAGE_POINT_CLOUD}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_raw_reference_vertices "${MAX_RAW_REFERENCE_VERTICES}" \
  --max_stage_query_vertices "${MAX_STAGE_QUERY_VERTICES}" \
  --max_delta_points "${MAX_DELTA_POINTS}" \
  --query_chunk_size "${QUERY_CHUNK_SIZE}" \
  --mesh_delta_quantile "${MESH_DELTA_QUANTILE}" \
  --mesh_delta_distance_threshold "${MESH_DELTA_DISTANCE_THRESHOLD}" \
  --mesh_delta_distance_floor "${MESH_DELTA_DISTANCE_FLOOR}" \
  --delta_radius_abs "${DELTA_RADIUS_ABS}" \
  --delta_radius_scale "${DELTA_RADIUS_SCALE}" \
  --delta_radius_mode "${DELTA_RADIUS_MODE}" \
  --gaussian_distance_mode "${GAUSSIAN_DISTANCE_MODE}" \
  --major_endpoint_scale "${MAJOR_ENDPOINT_SCALE}" \
  --tangent_angle_mode "${TANGENT_ANGLE_MODE}" \
  --tangent_angle_k "${TANGENT_ANGLE_K}" \
  --tangent_angle_distance_floor "${TANGENT_ANGLE_DISTANCE_FLOOR}" \
  --min_opacity "${MIN_OPACITY}" \
  --min_anisotropy "${MIN_ANISOTROPY}" \
  --min_candidate_score "${MIN_CANDIDATE_SCORE}" \
  --candidate_score_quantile "${CANDIDATE_SCORE_QUANTILE}" \
  --max_candidates "${MAX_CANDIDATES}" \
  --preview_max_points "${PREVIEW_MAX_POINTS}"

echo
echo "[done] output dir       : ${OUTPUT_DIR}"
echo "[done] payload          : ${OUTPUT_DIR}/mesh_delta_star_gaussian_candidates_v0.pt"
echo "[done] candidate preview: ${OUTPUT_DIR}/mesh_delta_star_gaussian_candidates_v0.ply"
echo "[done] mesh delta preview: ${OUTPUT_DIR}/mesh_delta_vertices_v0.ply"
echo "[done] summary          : ${OUTPUT_DIR}/mesh_delta_star_gaussian_candidates_v0_summary.json"

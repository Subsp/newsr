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
STAGE_MODEL_PATH="${STAGE_MODEL_PATH:-${PREPARE_DEBUG_MODEL_PATH}/debug_prepare_stages/${STAGE_NAME}}"
POINT_CLOUD_PLY="${POINT_CLOUD_PLY:-${STAGE_MODEL_PATH}/point_cloud/iteration_${PREPARE_DEBUG_ITERATION}/point_cloud.ply}"

MESH_COMPARE_ROOT="${MESH_COMPARE_ROOT:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}}"
MESH_PATH="${MESH_PATH:-${MESH_COMPARE_ROOT}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"

RUN_NAME="${RUN_NAME:-single_mesh_tangent_${STAGE_NAME}_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/single_mesh_tangent_gaussian_probe_v0/${SCENE_NAME}/${RUN_NAME}}"

MAX_MESH_REFERENCE_VERTICES="${MAX_MESH_REFERENCE_VERTICES:-2000000}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-262144}"
GAUSSIAN_DISTANCE_MODE="${GAUSSIAN_DISTANCE_MODE:-major_endpoints}"
MAJOR_ENDPOINT_SCALE="${MAJOR_ENDPOINT_SCALE:-1.0}"
ENDPOINT_DISTANCE_REDUCE="${ENDPOINT_DISTANCE_REDUCE:-min}"
RADIUS_MODE="${RADIUS_MODE:-absolute}"
RADIUS_ABS="${RADIUS_ABS:-0.08}"
RADIUS_SCALE="${RADIUS_SCALE:-1.0}"
TANGENT_ANGLE_K="${TANGENT_ANGLE_K:-12}"
TANGENT_ANGLE_DISTANCE_FLOOR="${TANGENT_ANGLE_DISTANCE_FLOOR:-0.15}"
SELECTION_MODE="${SELECTION_MODE:-surface_tangent}"
MIN_SURFACE_DISTANCE="${MIN_SURFACE_DISTANCE:-0.04}"
MIN_TANGENT_ANGLE_TO_TANGENT="${MIN_TANGENT_ANGLE_TO_TANGENT:-0.35}"
MIN_OPACITY="${MIN_OPACITY:-0.0001}"
MIN_ANISOTROPY="${MIN_ANISOTROPY:-1.0}"
MAX_SCALE_MAJOR="${MAX_SCALE_MAJOR:-0.0}"
MIN_DC_LUMA="${MIN_DC_LUMA:--999999.0}"
DC_LUMA_QUANTILE="${DC_LUMA_QUANTILE:-0.0}"
MIN_CANDIDATE_SCORE="${MIN_CANDIDATE_SCORE:-0.00005}"
CANDIDATE_SCORE_QUANTILE="${CANDIDATE_SCORE_QUANTILE:-50.0}"
MAX_CANDIDATES="${MAX_CANDIDATES:-800000}"
PREVIEW_MAX_POINTS="${PREVIEW_MAX_POINTS:-200000}"

echo "[single-mesh-tangent-v0] scene      : ${SCENE_ROOT}"
echo "[single-mesh-tangent-v0] mesh       : ${MESH_PATH}"
echo "[single-mesh-tangent-v0] point cloud: ${POINT_CLOUD_PLY}"
echo "[single-mesh-tangent-v0] output dir : ${OUTPUT_DIR}"

for path in "${MESH_PATH}" "${POINT_CLOUD_PLY}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[single-mesh-tangent-v0] required file not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/score_single_mesh_tangent_gaussians_v0.py" \
  --mesh_path "${MESH_PATH}" \
  --point_cloud_ply "${POINT_CLOUD_PLY}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_mesh_reference_vertices "${MAX_MESH_REFERENCE_VERTICES}" \
  --query_chunk_size "${QUERY_CHUNK_SIZE}" \
  --gaussian_distance_mode "${GAUSSIAN_DISTANCE_MODE}" \
  --major_endpoint_scale "${MAJOR_ENDPOINT_SCALE}" \
  --endpoint_distance_reduce "${ENDPOINT_DISTANCE_REDUCE}" \
  --radius_mode "${RADIUS_MODE}" \
  --radius_abs "${RADIUS_ABS}" \
  --radius_scale "${RADIUS_SCALE}" \
  --tangent_angle_k "${TANGENT_ANGLE_K}" \
  --tangent_angle_distance_floor "${TANGENT_ANGLE_DISTANCE_FLOOR}" \
  --selection_mode "${SELECTION_MODE}" \
  --min_surface_distance "${MIN_SURFACE_DISTANCE}" \
  --min_tangent_angle_to_tangent "${MIN_TANGENT_ANGLE_TO_TANGENT}" \
  --min_opacity "${MIN_OPACITY}" \
  --min_anisotropy "${MIN_ANISOTROPY}" \
  --max_scale_major "${MAX_SCALE_MAJOR}" \
  --min_dc_luma "${MIN_DC_LUMA}" \
  --dc_luma_quantile "${DC_LUMA_QUANTILE}" \
  --min_candidate_score "${MIN_CANDIDATE_SCORE}" \
  --candidate_score_quantile "${CANDIDATE_SCORE_QUANTILE}" \
  --max_candidates "${MAX_CANDIDATES}" \
  --preview_max_points "${PREVIEW_MAX_POINTS}"

echo
echo "[done] output dir       : ${OUTPUT_DIR}"
echo "[done] payload          : ${OUTPUT_DIR}/mesh_delta_star_gaussian_candidates_v0.pt"
echo "[done] candidate preview: ${OUTPUT_DIR}/mesh_delta_star_gaussian_candidates_v0.ply"
echo "[done] surface preview  : ${OUTPUT_DIR}/single_mesh_surface_reference_v0.ply"
echo "[done] summary          : ${OUTPUT_DIR}/mesh_delta_star_gaussian_candidates_v0_summary.json"

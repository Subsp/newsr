#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VGGTSR_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${VGGTSR_ROOT}/mip-splatting}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RAW_MIP_MODEL_PATH="${RAW_MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k}"
RAW_MIP_ITERATION="${RAW_MIP_ITERATION:-30000}"
TARGET_RUN_NAME="${TARGET_RUN_NAME:-mip_to_soflr_surface_early4ksoft_v1}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${TARGET_RUN_NAME}/pulled_mip_model}"
TARGET_ITERATION="${TARGET_ITERATION:-34000}"
TARGET_POINT_CLOUD="${TARGET_POINT_CLOUD:-${TARGET_MODEL_PATH}/point_cloud/iteration_${TARGET_ITERATION}/point_cloud.ply}"

RUN_NAME="${RUN_NAME:-rawmip_local_mesh_to_${TARGET_RUN_NAME}_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/local_mesh_alignment_probe_v0/${SCENE_NAME}/${RUN_NAME}}"
MESH_OUTPUT_DIR="${MESH_OUTPUT_DIR:-${OUTPUT_ROOT}/raw_mip_pseudo_mesh}"
ALIGN_OUTPUT_DIR="${ALIGN_OUTPUT_DIR:-${OUTPUT_ROOT}/target_mesh_alignment}"

FOCUS_MODE="${FOCUS_MODE:-main_object}"
FOCUS_CLUSTER_RATIO="${FOCUS_CLUSTER_RATIO:-0.50}"
FOCUS_INLIER_QUANTILE="${FOCUS_INLIER_QUANTILE:-0.90}"
FOCUS_PADDING_RATIO="${FOCUS_PADDING_RATIO:-0.08}"
FOCUS_CENTER="${FOCUS_CENTER:-}"
FOCUS_EXTENT="${FOCUS_EXTENT:-}"

OPACITY_MIN="${OPACITY_MIN:-0.02}"
SHEETNESS_MIN="${SHEETNESS_MIN:-2.0}"
MAX_SURFELS="${MAX_SURFELS:-60000}"
SAMPLES_PER_SURFEL="${SAMPLES_PER_SURFEL:-3}"
NORMAL_AXIS="${NORMAL_AXIS:-min_scale}"
POISSON_DEPTH="${POISSON_DEPTH:-9}"
DENSITY_PRUNE_QUANTILE="${DENSITY_PRUNE_QUANTILE:-0.05}"

SURFACE_QUERY_MODE="${SURFACE_QUERY_MODE:-auto}"
MESH_SURFACE_SAMPLE_COUNT="${MESH_SURFACE_SAMPLE_COUNT:-1000000}"
SURFACE_QUERY_CHUNK_SIZE="${SURFACE_QUERY_CHUNK_SIZE:-131072}"
SURFACE_DISTANCE_THRESHOLD="${SURFACE_DISTANCE_THRESHOLD:-0.03}"
MIN_CANDIDATE_OPACITY="${MIN_CANDIDATE_OPACITY:-0.02}"
MIN_EFFECTIVE_ANISOTROPY="${MIN_EFFECTIVE_ANISOTROPY:-4.0}"
MIN_NORMAL_OVER_MINOR="${MIN_NORMAL_OVER_MINOR:-1.5}"
MIN_DISTANCE_OVER_MAJOR="${MIN_DISTANCE_OVER_MAJOR:-0.75}"
MIN_CANDIDATE_SCORE="${MIN_CANDIDATE_SCORE:-1.0}"
PREVIEW_MAX_POINTS="${PREVIEW_MAX_POINTS:-200000}"

RUN_EXTRACT_MESH="${RUN_EXTRACT_MESH:-1}"
RUN_SCORE_ALIGNMENT="${RUN_SCORE_ALIGNMENT:-1}"

MESH_PATH="${MESH_PATH:-${MESH_OUTPUT_DIR}/surface_mesh_poisson.ply}"

mkdir -p "${OUTPUT_ROOT}"

echo "[local-mesh-align-v0] scene          : ${SCENE_ROOT}"
echo "[local-mesh-align-v0] raw mip model  : ${RAW_MIP_MODEL_PATH} iter=${RAW_MIP_ITERATION}"
echo "[local-mesh-align-v0] target model   : ${TARGET_MODEL_PATH} iter=${TARGET_ITERATION}"
echo "[local-mesh-align-v0] target ply     : ${TARGET_POINT_CLOUD}"
echo "[local-mesh-align-v0] focus mode     : ${FOCUS_MODE}"
echo "[local-mesh-align-v0] output root    : ${OUTPUT_ROOT}"

for path in "${RAW_MIP_MODEL_PATH}" "${TARGET_POINT_CLOUD}" "${MIPSPLATTING_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[local-mesh-align-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ "${RUN_EXTRACT_MESH}" == "1" ]]; then
  echo
  echo "[1/2] extract local pseudo mesh from vanilla raw MIP"
  RAW_MIP_MODEL_PATH="${RAW_MIP_MODEL_PATH}" \
  MODEL_DIR="${RAW_MIP_MODEL_PATH}" \
  ITERATION="${RAW_MIP_ITERATION}" \
  OUTPUT_DIR="${MESH_OUTPUT_DIR}" \
  MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  DEVICE="${DEVICE:-cuda}" \
  OPACITY_MIN="${OPACITY_MIN}" \
  SHEETNESS_MIN="${SHEETNESS_MIN}" \
  MAX_SURFELS="${MAX_SURFELS}" \
  SAMPLES_PER_SURFEL="${SAMPLES_PER_SURFEL}" \
  NORMAL_AXIS="${NORMAL_AXIS}" \
  FOCUS_MODE="${FOCUS_MODE}" \
  FOCUS_CLUSTER_RATIO="${FOCUS_CLUSTER_RATIO}" \
  FOCUS_INLIER_QUANTILE="${FOCUS_INLIER_QUANTILE}" \
  FOCUS_PADDING_RATIO="${FOCUS_PADDING_RATIO}" \
  FOCUS_CENTER="${FOCUS_CENTER}" \
  FOCUS_EXTENT="${FOCUS_EXTENT}" \
  POISSON_DEPTH="${POISSON_DEPTH}" \
  DENSITY_PRUNE_QUANTILE="${DENSITY_PRUNE_QUANTILE}" \
  bash "${SCRIPT_DIR}/extract_mipsplatting_pseudo_mesh.sh"
else
  echo
  echo "[1/2] skip mesh extraction (RUN_EXTRACT_MESH=${RUN_EXTRACT_MESH})"
fi

if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[local-mesh-align-v0] mesh missing: ${MESH_PATH}" >&2
  exit 1
fi

if [[ "${RUN_SCORE_ALIGNMENT}" == "1" ]]; then
  echo
  echo "[2/2] score target Gaussians against local mesh"
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/score_gaussian_mesh_alignment_v0.py" \
    --point_cloud_ply "${TARGET_POINT_CLOUD}" \
    --mesh_path "${MESH_PATH}" \
    --output_dir "${ALIGN_OUTPUT_DIR}" \
    --surface_query_mode "${SURFACE_QUERY_MODE}" \
    --mesh_surface_sample_count "${MESH_SURFACE_SAMPLE_COUNT}" \
    --surface_query_chunk_size "${SURFACE_QUERY_CHUNK_SIZE}" \
    --surface_distance_threshold "${SURFACE_DISTANCE_THRESHOLD}" \
    --min_candidate_opacity "${MIN_CANDIDATE_OPACITY}" \
    --min_effective_anisotropy "${MIN_EFFECTIVE_ANISOTROPY}" \
    --min_normal_over_minor "${MIN_NORMAL_OVER_MINOR}" \
    --min_distance_over_major "${MIN_DISTANCE_OVER_MAJOR}" \
    --min_candidate_score "${MIN_CANDIDATE_SCORE}" \
    --preview_max_points "${PREVIEW_MAX_POINTS}"
else
  echo
  echo "[2/2] skip mesh alignment scoring (RUN_SCORE_ALIGNMENT=${RUN_SCORE_ALIGNMENT})"
fi

echo
echo "[done] local mesh       : ${MESH_PATH}"
echo "[done] surfel points    : ${MESH_OUTPUT_DIR}/surfel_points.ply"
echo "[done] mesh info        : ${MESH_OUTPUT_DIR}/surface_extract_info.json"
echo "[done] alignment payload: ${ALIGN_OUTPUT_DIR}/gaussian_mesh_alignment_v0.pt"
echo "[done] candidate preview: ${ALIGN_OUTPUT_DIR}/mesh_alignment_candidates_v0.ply"
echo "[done] summary          : ${ALIGN_OUTPUT_DIR}/gaussian_mesh_alignment_v0_summary.json"

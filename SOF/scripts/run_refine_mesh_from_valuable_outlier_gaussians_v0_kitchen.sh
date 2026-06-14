#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VGGTSR_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-${SOF_ROOT}/output/mipsplatting_prior_repro/${SCENE_NAME}/qwen244_nearuncertain_rgbbottleneck_migrate_v0}"
ITERATION="${ITERATION:-33000}"
POINT_CLOUD_PLY="${POINT_CLOUD_PLY:-${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply}"

GS2MESH_MESH_NAME="${GS2MESH_MESH_NAME:-kitchen_MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_15_cleaned_mesh.ply}"
MESH_PATH="${MESH_PATH:-${WORK_ROOT}/${GS2MESH_MESH_NAME}}"

SURFACE_STATE_PAYLOAD="${SURFACE_STATE_PAYLOAD:-${SOF_ROOT}/output/gaussian_surface_sort/${SCENE_NAME}/mip30k_rerun_gs2mesh_surface_state_v0_relaxed_carrier_v1/gaussian_surface_state_v0.pt}"
RUN_NAME="${RUN_NAME:-qwen244_nearuncertain_rgbbottleneck_migrate33000_outlier_meshpull_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/mesh_refine_from_outlier_gs_v0/${SCENE_NAME}/${RUN_NAME}}"

CANDIDATE_KEYS="${CANDIDATE_KEYS:-near_surface_uncertain,off_surface_near_mesh}"
EXCLUDE_KEYS="${EXCLUDE_KEYS:-surface_carrier,low_opacity_neutral,no_mesh_neutral}"
SURFACE_QUERY_MODE="${SURFACE_QUERY_MODE:-auto}"
MESH_SURFACE_SAMPLE_COUNT="${MESH_SURFACE_SAMPLE_COUNT:-1000000}"
SURFACE_QUERY_CHUNK_SIZE="${SURFACE_QUERY_CHUNK_SIZE:-131072}"

MIN_OPACITY="${MIN_OPACITY:-0.02}"
OPACITY_HIGH="${OPACITY_HIGH:-0.35}"
THIN_RATIO_GOOD="${THIN_RATIO_GOOD:-0.12}"
THIN_RATIO_BAD="${THIN_RATIO_BAD:-0.45}"
MIN_ANISOTROPY="${MIN_ANISOTROPY:-2.0}"
ANISOTROPY_HIGH="${ANISOTROPY_HIGH:-8.0}"
MIN_NORMAL_FRACTION="${MIN_NORMAL_FRACTION:-0.45}"
MIN_ABS_NORMAL_OFFSET="${MIN_ABS_NORMAL_OFFSET:-0.001}"
MAX_ABS_NORMAL_OFFSET="${MAX_ABS_NORMAL_OFFSET:-0.035}"
MAX_OFFSET_SCALE_RATIO="${MAX_OFFSET_SCALE_RATIO:-1.5}"
MIN_VALUE_SCORE="${MIN_VALUE_SCORE:-0.04}"

HUBER_DELTA="${HUBER_DELTA:-0.012}"
MAX_VERTEX_OFFSET="${MAX_VERTEX_OFFSET:-0.015}"
MAX_VERTEX_OFFSET_EDGE_RATIO="${MAX_VERTEX_OFFSET_EDGE_RATIO:-0.25}"
NORMAL_OFFSET_GAIN="${NORMAL_OFFSET_GAIN:-0.65}"
SMOOTH_ITERATIONS="${SMOOTH_ITERATIONS:-8}"
SMOOTH_LAMBDA="${SMOOTH_LAMBDA:-0.45}"
DATA_ANCHOR="${DATA_ANCHOR:-0.70}"
PREVIEW_MAX_POINTS="${PREVIEW_MAX_POINTS:-250000}"
SEED="${SEED:-0}"

if [[ ! -f "${POINT_CLOUD_PLY}" ]]; then
  echo "[outlier-meshpull-v0] missing point cloud: ${POINT_CLOUD_PLY}" >&2
  exit 1
fi
if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[outlier-meshpull-v0] missing mesh: ${MESH_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SURFACE_STATE_PAYLOAD}" ]]; then
  echo "[outlier-meshpull-v0] missing surface-state payload: ${SURFACE_STATE_PAYLOAD}" >&2
  exit 1
fi

echo "[outlier-meshpull-v0] scene       : ${SCENE_ROOT}"
echo "[outlier-meshpull-v0] point cloud : ${POINT_CLOUD_PLY}"
echo "[outlier-meshpull-v0] mesh        : ${MESH_PATH}"
echo "[outlier-meshpull-v0] state       : ${SURFACE_STATE_PAYLOAD}"
echo "[outlier-meshpull-v0] output      : ${OUTPUT_DIR}"
echo "[outlier-meshpull-v0] candidates  : ${CANDIDATE_KEYS} exclude=${EXCLUDE_KEYS}"
echo "[outlier-meshpull-v0] gates       : opacity>=${MIN_OPACITY} aniso>=${MIN_ANISOTROPY} normal_frac>=${MIN_NORMAL_FRACTION} score>=${MIN_VALUE_SCORE}"
echo "[outlier-meshpull-v0] offsets     : normal=[${MIN_ABS_NORMAL_OFFSET},${MAX_ABS_NORMAL_OFFSET}] huber=${HUBER_DELTA} max_vertex=${MAX_VERTEX_OFFSET} gain=${NORMAL_OFFSET_GAIN}"
echo "[outlier-meshpull-v0] smooth      : iters=${SMOOTH_ITERATIONS} lambda=${SMOOTH_LAMBDA} anchor=${DATA_ANCHOR}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/refine_mesh_from_valuable_outlier_gaussians_v0.py" \
  --point_cloud_ply "${POINT_CLOUD_PLY}" \
  --mesh_path "${MESH_PATH}" \
  --surface_state_payload "${SURFACE_STATE_PAYLOAD}" \
  --output_dir "${OUTPUT_DIR}" \
  --candidate_keys "${CANDIDATE_KEYS}" \
  --exclude_keys "${EXCLUDE_KEYS}" \
  --surface_query_mode "${SURFACE_QUERY_MODE}" \
  --mesh_surface_sample_count "${MESH_SURFACE_SAMPLE_COUNT}" \
  --surface_query_chunk_size "${SURFACE_QUERY_CHUNK_SIZE}" \
  --min_opacity "${MIN_OPACITY}" \
  --opacity_high "${OPACITY_HIGH}" \
  --thin_ratio_good "${THIN_RATIO_GOOD}" \
  --thin_ratio_bad "${THIN_RATIO_BAD}" \
  --min_anisotropy "${MIN_ANISOTROPY}" \
  --anisotropy_high "${ANISOTROPY_HIGH}" \
  --min_normal_fraction "${MIN_NORMAL_FRACTION}" \
  --min_abs_normal_offset "${MIN_ABS_NORMAL_OFFSET}" \
  --max_abs_normal_offset "${MAX_ABS_NORMAL_OFFSET}" \
  --max_offset_scale_ratio "${MAX_OFFSET_SCALE_RATIO}" \
  --min_value_score "${MIN_VALUE_SCORE}" \
  --huber_delta "${HUBER_DELTA}" \
  --max_vertex_offset "${MAX_VERTEX_OFFSET}" \
  --max_vertex_offset_edge_ratio "${MAX_VERTEX_OFFSET_EDGE_RATIO}" \
  --normal_offset_gain "${NORMAL_OFFSET_GAIN}" \
  --smooth_iterations "${SMOOTH_ITERATIONS}" \
  --smooth_lambda "${SMOOTH_LAMBDA}" \
  --data_anchor "${DATA_ANCHOR}" \
  --preview_max_points "${PREVIEW_MAX_POINTS}" \
  --seed "${SEED}"

echo "[done] refined mesh : ${OUTPUT_DIR}/refined_mesh_from_outlier_gs_v0.ply"
echo "[done] offset heat  : ${OUTPUT_DIR}/refined_mesh_offset_heat_v0.ply"
echo "[done] evidence    : ${OUTPUT_DIR}/valuable_outlier_mesh_refine_v0.pt"
echo "[done] summary     : ${OUTPUT_DIR}/valuable_outlier_mesh_refine_v0_summary.json"
echo "[hint] reclassify with MESH_PATH=${OUTPUT_DIR}/refined_mesh_from_outlier_gs_v0.ply"

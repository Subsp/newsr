#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VGGTSR_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k_rerun_v0}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
POINT_CLOUD_PLY="${POINT_CLOUD_PLY:-${MIP_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply}"

GS2MESH_ROOT="${GS2MESH_ROOT:-${VGGTSR_ROOT}/gs2mesh}"
GS2MESH_EXPERIMENT_NAME="${GS2MESH_EXPERIMENT_NAME:-MipNerf360_kitchen_mipsplatting30k}"
GS2MESH_RENDERER_FOLDER="${GS2MESH_RENDERER_FOLDER:-kitchen}"
GS2MESH_MESH_NAME="${GS2MESH_MESH_NAME:-kitchen_MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_15_cleaned_mesh.ply}"
MESH_PATH="${MESH_PATH:-${WORK_ROOT}/${GS2MESH_MESH_NAME}}"
PRIOR_DIR="${PRIOR_DIR:-${WORK_ROOT}/test_preds_1_vosr_same/qwen_steps1_seed42_rcgm}"
PRIOR_SIZE_POLICY="${PRIOR_SIZE_POLICY:-center_crop_or_pad_to_render_resolution}"

SURFACE_STATE_PROFILE="${SURFACE_STATE_PROFILE:-conservative_v0}"
DEFAULT_RUN_NAME="mip30k_rerun_gs2mesh_surface_state_v0"
if [[ "${SURFACE_STATE_PROFILE}" != "conservative_v0" ]]; then
  DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_${SURFACE_STATE_PROFILE}"
fi
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOF_ROOT}/output/gaussian_surface_state_v0/${SCENE_NAME}/${RUN_NAME}}"

SURFACE_QUERY_MODE="${SURFACE_QUERY_MODE:-auto}"
MESH_SURFACE_SAMPLE_COUNT="${MESH_SURFACE_SAMPLE_COUNT:-1000000}"
SURFACE_QUERY_CHUNK_SIZE="${SURFACE_QUERY_CHUNK_SIZE:-131072}"
case "${SURFACE_STATE_PROFILE}" in
  conservative_v0)
    DEFAULT_AXIS_ANISOTROPY_THRESHOLD="4.0"
    DEFAULT_AXIS_SAMPLE_COUNT="3"
    DEFAULT_AXIS_SAMPLE_EXTENT="1.0"
    DEFAULT_COVERAGE_ABS_RADIUS="0.08"
    DEFAULT_COVERAGE_SCALE_RATIO="3.0"
    DEFAULT_COVERAGE_EDGE_RATIO="0.5"
    DEFAULT_COVERAGE_TAU_FLOOR="0.02"
    DEFAULT_ATTACH_TAU_ABS="0.03"
    DEFAULT_ATTACH_TAU_SCALE_RATIO="1.5"
    DEFAULT_ATTACH_TAU_EDGE_RATIO="0.1"
    DEFAULT_ATTACH_TAU_FLOOR="0.01"
    DEFAULT_SURFACE_CONF_MIN="0.35"
    DEFAULT_SURFACE_MAX_D_NORM="1.0"
    DEFAULT_NEAR_SURFACE_MAX_D_NORM="2.5"
    DEFAULT_OFF_SURFACE_D_NORM="3.5"
    DEFAULT_SIGMA_DIST="1.25"
    DEFAULT_COV_BETA="0.5"
    DEFAULT_NORMAL_SCORE_FLOOR="0.35"
    ;;
  relaxed_carrier_v1)
    DEFAULT_AXIS_ANISOTROPY_THRESHOLD="3.0"
    DEFAULT_AXIS_SAMPLE_COUNT="5"
    DEFAULT_AXIS_SAMPLE_EXTENT="1.25"
    DEFAULT_COVERAGE_ABS_RADIUS="0.10"
    DEFAULT_COVERAGE_SCALE_RATIO="4.0"
    DEFAULT_COVERAGE_EDGE_RATIO="0.75"
    DEFAULT_COVERAGE_TAU_FLOOR="0.025"
    DEFAULT_ATTACH_TAU_ABS="0.05"
    DEFAULT_ATTACH_TAU_SCALE_RATIO="2.5"
    DEFAULT_ATTACH_TAU_EDGE_RATIO="0.20"
    DEFAULT_ATTACH_TAU_FLOOR="0.015"
    DEFAULT_SURFACE_CONF_MIN="0.18"
    DEFAULT_SURFACE_MAX_D_NORM="1.60"
    DEFAULT_NEAR_SURFACE_MAX_D_NORM="3.20"
    DEFAULT_OFF_SURFACE_D_NORM="4.20"
    DEFAULT_SIGMA_DIST="1.60"
    DEFAULT_COV_BETA="0.80"
    DEFAULT_NORMAL_SCORE_FLOOR="0.50"
    ;;
  *)
    echo "[surface-state-v0] unknown SURFACE_STATE_PROFILE=${SURFACE_STATE_PROFILE}" >&2
    exit 1
    ;;
esac

AXIS_ANISOTROPY_THRESHOLD="${AXIS_ANISOTROPY_THRESHOLD:-${DEFAULT_AXIS_ANISOTROPY_THRESHOLD}}"
AXIS_SAMPLE_COUNT="${AXIS_SAMPLE_COUNT:-${DEFAULT_AXIS_SAMPLE_COUNT}}"
AXIS_SAMPLE_EXTENT="${AXIS_SAMPLE_EXTENT:-${DEFAULT_AXIS_SAMPLE_EXTENT}}"

COVERAGE_ABS_RADIUS="${COVERAGE_ABS_RADIUS:-${DEFAULT_COVERAGE_ABS_RADIUS}}"
COVERAGE_SCALE_RATIO="${COVERAGE_SCALE_RATIO:-${DEFAULT_COVERAGE_SCALE_RATIO}}"
COVERAGE_EDGE_RATIO="${COVERAGE_EDGE_RATIO:-${DEFAULT_COVERAGE_EDGE_RATIO}}"
COVERAGE_TAU_FLOOR="${COVERAGE_TAU_FLOOR:-${DEFAULT_COVERAGE_TAU_FLOOR}}"
ATTACH_TAU_ABS="${ATTACH_TAU_ABS:-${DEFAULT_ATTACH_TAU_ABS}}"
ATTACH_TAU_SCALE_RATIO="${ATTACH_TAU_SCALE_RATIO:-${DEFAULT_ATTACH_TAU_SCALE_RATIO}}"
ATTACH_TAU_EDGE_RATIO="${ATTACH_TAU_EDGE_RATIO:-${DEFAULT_ATTACH_TAU_EDGE_RATIO}}"
ATTACH_TAU_FLOOR="${ATTACH_TAU_FLOOR:-${DEFAULT_ATTACH_TAU_FLOOR}}"
SURFACE_CONF_MIN="${SURFACE_CONF_MIN:-${DEFAULT_SURFACE_CONF_MIN}}"
SURFACE_MAX_D_NORM="${SURFACE_MAX_D_NORM:-${DEFAULT_SURFACE_MAX_D_NORM}}"
NEAR_SURFACE_MAX_D_NORM="${NEAR_SURFACE_MAX_D_NORM:-${DEFAULT_NEAR_SURFACE_MAX_D_NORM}}"
OFF_SURFACE_D_NORM="${OFF_SURFACE_D_NORM:-${DEFAULT_OFF_SURFACE_D_NORM}}"
SIGMA_DIST="${SIGMA_DIST:-${DEFAULT_SIGMA_DIST}}"
COV_BETA="${COV_BETA:-${DEFAULT_COV_BETA}}"
NORMAL_SCORE_FLOOR="${NORMAL_SCORE_FLOOR:-${DEFAULT_NORMAL_SCORE_FLOOR}}"
MIN_ACTION_OPACITY="${MIN_ACTION_OPACITY:-0.01}"
PREVIEW_MAX_POINTS="${PREVIEW_MAX_POINTS:-250000}"

if [[ ! -f "${POINT_CLOUD_PLY}" ]]; then
  echo "[surface-state-v0] missing point cloud: ${POINT_CLOUD_PLY}" >&2
  exit 1
fi
if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[surface-state-v0] missing mesh: ${MESH_PATH}" >&2
  echo "[surface-state-v0] pass MESH_PATH=/path/to/gs2mesh_cleaned_mesh.ply if your gs2mesh output is elsewhere." >&2
  exit 1
fi

echo "[surface-state-v0] scene     : ${SCENE_ROOT}"
echo "[surface-state-v0] mip ply   : ${POINT_CLOUD_PLY}"
echo "[surface-state-v0] mesh      : ${MESH_PATH}"
echo "[surface-state-v0] prior     : ${PRIOR_DIR} (reserved, size_policy=${PRIOR_SIZE_POLICY})"
echo "[surface-state-v0] output    : ${OUTPUT_DIR}"
echo "[surface-state-v0] profile   : ${SURFACE_STATE_PROFILE}"
echo "[surface-state-v0] coverage  : abs=${COVERAGE_ABS_RADIUS} scale=${COVERAGE_SCALE_RATIO} edge=${COVERAGE_EDGE_RATIO}"
echo "[surface-state-v0] axis      : aniso>=${AXIS_ANISOTROPY_THRESHOLD} samples=${AXIS_SAMPLE_COUNT} extent=${AXIS_SAMPLE_EXTENT}"
echo "[surface-state-v0] carrier   : conf>=${SURFACE_CONF_MIN} d_norm<=${SURFACE_MAX_D_NORM} near<=${NEAR_SURFACE_MAX_D_NORM}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/classify_gaussian_surface_state_v0.py" \
  --point_cloud_ply "${POINT_CLOUD_PLY}" \
  --mesh_path "${MESH_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --surface_query_mode "${SURFACE_QUERY_MODE}" \
  --mesh_surface_sample_count "${MESH_SURFACE_SAMPLE_COUNT}" \
  --surface_query_chunk_size "${SURFACE_QUERY_CHUNK_SIZE}" \
  --axis_anisotropy_threshold "${AXIS_ANISOTROPY_THRESHOLD}" \
  --axis_sample_count "${AXIS_SAMPLE_COUNT}" \
  --axis_sample_extent "${AXIS_SAMPLE_EXTENT}" \
  --coverage_abs_radius "${COVERAGE_ABS_RADIUS}" \
  --coverage_scale_ratio "${COVERAGE_SCALE_RATIO}" \
  --coverage_edge_ratio "${COVERAGE_EDGE_RATIO}" \
  --coverage_tau_floor "${COVERAGE_TAU_FLOOR}" \
  --attach_tau_abs "${ATTACH_TAU_ABS}" \
  --attach_tau_scale_ratio "${ATTACH_TAU_SCALE_RATIO}" \
  --attach_tau_edge_ratio "${ATTACH_TAU_EDGE_RATIO}" \
  --attach_tau_floor "${ATTACH_TAU_FLOOR}" \
  --surface_conf_min "${SURFACE_CONF_MIN}" \
  --surface_max_d_norm "${SURFACE_MAX_D_NORM}" \
  --near_surface_max_d_norm "${NEAR_SURFACE_MAX_D_NORM}" \
  --off_surface_d_norm "${OFF_SURFACE_D_NORM}" \
  --sigma_dist "${SIGMA_DIST}" \
  --cov_beta "${COV_BETA}" \
  --normal_score_floor "${NORMAL_SCORE_FLOOR}" \
  --min_action_opacity "${MIN_ACTION_OPACITY}" \
  --preview_max_points "${PREVIEW_MAX_POINTS}"

echo "[done] payload : ${OUTPUT_DIR}/gaussian_surface_state_v0.pt"
echo "[done] summary : ${OUTPUT_DIR}/gaussian_surface_state_v0_summary.json"
echo "[done] preview : ${OUTPUT_DIR}/gaussian_surface_state_classes_v0.ply"

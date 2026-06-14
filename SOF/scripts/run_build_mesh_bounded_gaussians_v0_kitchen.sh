#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-${STAGE_NAME}_geometry_only_v0}"
MODEL_PATH="${MODEL_PATH:-${SOF_ROOT}/output/mask_guided_reparameterization_v0/${SCENE_NAME}/${SOURCE_RUN_NAME}}"
ITERATION="${ITERATION:-34000}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
SPLIT="${SPLIT:-both}"
MAX_VIEWS="${MAX_VIEWS:-0}"

MESH_COMPARE_ROOT="${MESH_COMPARE_ROOT:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}}"
MESH_PATH="${MESH_PATH:-${MESH_COMPARE_ROOT}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
MIP_ITERATION="${MIP_ITERATION:-30000}"

# Keep this compatible with downstream mesh-evidence scripts, which address this
# output as MESH_BOUNDED_RUN_NAME.
if [[ -n "${MESH_BOUNDED_RUN_NAME:-}" && -z "${OUTPUT_RUN_NAME:-}" ]]; then
  OUTPUT_RUN_NAME="${MESH_BOUNDED_RUN_NAME}"
fi
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${SOURCE_RUN_NAME}_mesh_bounded_v0}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/mesh_bounded_gaussians_v0/${SCENE_NAME}/${OUTPUT_RUN_NAME}}"

SURFACE_QUERY_MODE="${SURFACE_QUERY_MODE:-auto}"
MESH_SURFACE_SAMPLE_COUNT="${MESH_SURFACE_SAMPLE_COUNT:-1000000}"
SURFACE_QUERY_CHUNK_SIZE="${SURFACE_QUERY_CHUNK_SIZE:-131072}"
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-200000}"
K_PX="${K_PX:-2.5}"
K_EDGE="${K_EDGE:-0.4}"
TAU_FLOOR="${TAU_FLOOR:-1e-4}"
SIGMA_DIST="${SIGMA_DIST:-1.25}"
SIGMA_NORMAL="${SIGMA_NORMAL:-1.25}"
VIEW_CONF_VIEWS="${VIEW_CONF_VIEWS:-4.0}"
MIP_ALPHA_REF="${MIP_ALPHA_REF:-0.15}"
MIP_DEPTH_TAU_RATIO="${MIP_DEPTH_TAU_RATIO:-0.03}"
MIP_COLOR_BLEND="${MIP_COLOR_BLEND:-1.0}"
MIN_MIP_COLOR_WEIGHT="${MIN_MIP_COLOR_WEIGHT:-0.03}"
COLOR_SIGMA="${COLOR_SIGMA:-0.18}"
COLOR_CONFIDENCE_STRENGTH="${COLOR_CONFIDENCE_STRENGTH:-0.25}"
CONFIDENCE_MIN="${CONFIDENCE_MIN:-0.30}"
NORMAL_OFFSET_ETA="${NORMAL_OFFSET_ETA:-0.5}"
TAU_BETA="${TAU_BETA:-0.5}"
ALPHA_MAX="${ALPHA_MAX:-0.035}"
NORMAL_SCALE_CAP="${NORMAL_SCALE_CAP:-0.35}"
TANGENT_SCALE_MIN_RATIO="${TANGENT_SCALE_MIN_RATIO:-0.08}"
TANGENT_SCALE_MAX_RATIO="${TANGENT_SCALE_MAX_RATIO:-1.10}"
LONG_ANISOTROPY_THRESHOLD="${LONG_ANISOTROPY_THRESHOLD:-12.0}"
MAJOR_SAMPLE_COUNT="${MAJOR_SAMPLE_COUNT:-3}"
MAJOR_SAMPLE_EXTENT="${MAJOR_SAMPLE_EXTENT:-1.0}"
RENDER_CONFIDENCE_MAPS="${RENDER_CONFIDENCE_MAPS:-1}"
RENDER_MAX_VIEWS="${RENDER_MAX_VIEWS:-16}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[mesh-bounded-gs-v0] missing source model: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[mesh-bounded-gs-v0] missing mesh: ${MESH_PATH}" >&2
  exit 1
fi
if [[ -n "${MIP_MODEL_PATH}" && ! -d "${MIP_MODEL_PATH}" ]]; then
  echo "[mesh-bounded-gs-v0] missing MIP support model: ${MIP_MODEL_PATH}" >&2
  exit 1
fi

echo "[mesh-bounded-gs-v0] scene       : ${SCENE_ROOT}"
echo "[mesh-bounded-gs-v0] source model: ${MODEL_PATH} iter=${ITERATION}"
echo "[mesh-bounded-gs-v0] mesh        : ${MESH_PATH}"
echo "[mesh-bounded-gs-v0] mip support : ${MIP_MODEL_PATH:-disabled} iter=${MIP_ITERATION}"
echo "[mesh-bounded-gs-v0] views       : split=${SPLIT} images=${IMAGES_SUBDIR} max=${MAX_VIEWS}"
echo "[mesh-bounded-gs-v0] output      : ${OUTPUT_MODEL_PATH}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/build_mesh_bounded_gaussians_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --mesh_path "${MESH_PATH}"
  --output_model_path "${OUTPUT_MODEL_PATH}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --iteration "${ITERATION}"
  --max_views "${MAX_VIEWS}"
  --mip_model_path "${MIP_MODEL_PATH}"
  --mip_iteration "${MIP_ITERATION}"
  --surface_query_mode "${SURFACE_QUERY_MODE}"
  --mesh_surface_sample_count "${MESH_SURFACE_SAMPLE_COUNT}"
  --surface_query_chunk_size "${SURFACE_QUERY_CHUNK_SIZE}"
  --projection_chunk_size "${PROJECTION_CHUNK_SIZE}"
  --k_px "${K_PX}"
  --k_edge "${K_EDGE}"
  --tau_floor "${TAU_FLOOR}"
  --sigma_dist "${SIGMA_DIST}"
  --sigma_normal "${SIGMA_NORMAL}"
  --view_conf_views "${VIEW_CONF_VIEWS}"
  --mip_alpha_ref "${MIP_ALPHA_REF}"
  --mip_depth_tau_ratio "${MIP_DEPTH_TAU_RATIO}"
  --mip_color_blend "${MIP_COLOR_BLEND}"
  --min_mip_color_weight "${MIN_MIP_COLOR_WEIGHT}"
  --color_sigma "${COLOR_SIGMA}"
  --color_confidence_strength "${COLOR_CONFIDENCE_STRENGTH}"
  --confidence_min "${CONFIDENCE_MIN}"
  --normal_offset_eta "${NORMAL_OFFSET_ETA}"
  --tau_beta "${TAU_BETA}"
  --alpha_max "${ALPHA_MAX}"
  --normal_scale_cap "${NORMAL_SCALE_CAP}"
  --tangent_scale_min_ratio "${TANGENT_SCALE_MIN_RATIO}"
  --tangent_scale_max_ratio "${TANGENT_SCALE_MAX_RATIO}"
  --long_anisotropy_threshold "${LONG_ANISOTROPY_THRESHOLD}"
  --major_sample_count "${MAJOR_SAMPLE_COUNT}"
  --major_sample_extent "${MAJOR_SAMPLE_EXTENT}"
  --render_max_views "${RENDER_MAX_VIEWS}"
)

if [[ "${RENDER_CONFIDENCE_MAPS}" == "1" ]]; then
  CMD+=(--render_confidence_maps)
fi
if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
  CMD+=(--white_background)
fi

"${CMD[@]}"

echo "[done] proxy model : ${OUTPUT_MODEL_PATH}"
echo "[done] proxy ply   : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
echo "[done] payload     : ${OUTPUT_MODEL_PATH}/mesh_bounded_gaussians_v0.pt"
echo "[done] summary     : ${OUTPUT_MODEL_PATH}/mesh_bounded_gaussians_v0_summary.json"

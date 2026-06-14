#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input}"
MIP_ITERATION="${MIP_ITERATION:-30000}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-${MIP_ITERATION}}"

RUN_NAME="${RUN_NAME:-view_aligned_volume_delete_v1}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/cleanup_mip_view_aligned_volume_artifacts_v0/${SCENE_NAME}/${RUN_NAME}}"
OUTPUT_MODEL="${OUTPUT_MODEL:-${RUN_ROOT}/cleaned_mip_model_view_volume_v1}"
RENDER_DIR="${RENDER_DIR:-${RUN_ROOT}/cleaned_mip_renders_no_gt_view_volume_v1}"

LR_IMAGES_SUBDIR="${LR_IMAGES_SUBDIR:-images_8}"
MAX_VIEWS="${MAX_VIEWS:-16}"
DEPTH_MIN="${DEPTH_MIN:-0.05}"
SCREEN_MARGIN_PX="${SCREEN_MARGIN_PX:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-131072}"
USE_EFFECTIVE_SCALE="${USE_EFFECTIVE_SCALE:-1}"
RECOMPUTE_FILTER3D="${RECOMPUTE_FILTER3D:-1}"

SURFACE_MESH_PATH="${SURFACE_MESH_PATH:-}"
SURFACE_QUERY_MODE="${SURFACE_QUERY_MODE:-auto}"
MESH_SURFACE_SAMPLE_COUNT="${MESH_SURFACE_SAMPLE_COUNT:-500000}"
SURFACE_QUERY_CHUNK_SIZE="${SURFACE_QUERY_CHUNK_SIZE:-131072}"
SURFACE_DISTANCE_THRESHOLD="${SURFACE_DISTANCE_THRESHOLD:-0.035}"
REQUIRE_SURFACE_OUTSIDE="${REQUIRE_SURFACE_OUTSIDE:-0}"

DELETE_QUANTILE="${DELETE_QUANTILE:-0.980}"
MIN_VISIBLE_VIEWS="${MIN_VISIBLE_VIEWS:-1}"
CANDIDATE_MODE="${CANDIDATE_MODE:-volume_stress}"
MAX_VISIBLE_FRACTION="${MAX_VISIBLE_FRACTION:-1.00}"
MAX_OPACITY="${MAX_OPACITY:-1.00}"
MIN_AXIS_ALIGNMENT="${MIN_AXIS_ALIGNMENT:-0.88}"
MIN_AXIS_ANISOTROPY="${MIN_AXIS_ANISOTROPY:-1.70}"
MIN_RAY_THICKNESS_RATIO="${MIN_RAY_THICKNESS_RATIO:-1.80}"
MIN_SIDE_EXPLOSION="${MIN_SIDE_EXPLOSION:-1.70}"
MIN_SIDE_RADIUS_PX="${MIN_SIDE_RADIUS_PX:-24}"
MIN_RADIUS_PX="${MIN_RADIUS_PX:-18}"
MIN_EFFECTIVE_SCALE_RATIO="${MIN_EFFECTIVE_SCALE_RATIO:-0.0025}"
MIN_VOLUME_RADIUS_RATIO="${MIN_VOLUME_RADIUS_RATIO:-0.0015}"
MIN_FILTER_INFLATION="${MIN_FILTER_INFLATION:-1.25}"
MIN_FILTER_SCALE_RATIO="${MIN_FILTER_SCALE_RATIO:-0.60}"
STRESS_AXIS_SOURCE="${STRESS_AXIS_SOURCE:-effective}"
STRESS_SHORT_AXIS_SCALE_FACTOR="${STRESS_SHORT_AXIS_SCALE_FACTOR:-8.0}"
STRESS_MIN_AXIS_TO_MAX_RATIO="${STRESS_MIN_AXIS_TO_MAX_RATIO:-1.00}"
STRESS_VISIBILITY_DOWNSAMPLE="${STRESS_VISIBILITY_DOWNSAMPLE:-8}"
STRESS_VISIBILITY_TOPK="${STRESS_VISIBILITY_TOPK:-4}"
STRESS_VISIBILITY_MAX_VISIBLE="${STRESS_VISIBILITY_MAX_VISIBLE:-60000}"
STRESS_VISIBILITY_MAX_PATCH_RADIUS="${STRESS_VISIBILITY_MAX_PATCH_RADIUS:-4}"
STRESS_MAJOR_IMPACT_THRESHOLD="${STRESS_MAJOR_IMPACT_THRESHOLD:-0.135}"
MIN_STRESS_IMPACT="${MIN_STRESS_IMPACT:-0.030}"
MIN_STRESS_RADIUS_GAIN="${MIN_STRESS_RADIUS_GAIN:-1.20}"
MIN_STRESS_RADIUS_PX="${MIN_STRESS_RADIUS_PX:-22}"
MIN_STRESS_MAJOR_IMPACT_VIEWS="${MIN_STRESS_MAJOR_IMPACT_VIEWS:-2}"
MIN_STRESS_MAJOR_IMPACT_VISIBLE_FRACTION="${MIN_STRESS_MAJOR_IMPACT_VISIBLE_FRACTION:-0.40}"
MAX_PRUNE_FRACTION="${MAX_PRUNE_FRACTION:-0.035}"
MAX_PRUNE_COUNT="${MAX_PRUNE_COUNT:-0}"

RUN_RENDER="${RUN_RENDER:-1}"
RENDER_SPLIT="${RENDER_SPLIT:-test}"
RENDER_IMAGES_SUBDIR="${RENDER_IMAGES_SUBDIR:-images_2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_LR_RECOVER="${RUN_LR_RECOVER:-0}"
LR_RECOVER_PROFILE="${LR_RECOVER_PROFILE:-mip_hr_anchor_v0}"
LR_RECOVER_RUN_NAME="${LR_RECOVER_RUN_NAME:-${RUN_NAME}_mip_hr_anchor_v0_miphr_v1}"
LR_RECOVER_RUN_RENDER="${LR_RECOVER_RUN_RENDER:-${RUN_RENDER}}"

if [[ ! -e "${MIP_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" ]]; then
  echo "[view-volume-cleanup-v0] missing mip model: ${MIP_MODEL_PATH}/point_cloud/iteration_${MIP_ITERATION}/point_cloud.ply" >&2
  exit 1
fi

echo "[view-volume-cleanup-v0] scene       : ${SCENE_ROOT}"
echo "[view-volume-cleanup-v0] mip model   : ${MIP_MODEL_PATH} iter=${MIP_ITERATION}"
echo "[view-volume-cleanup-v0] output      : ${OUTPUT_MODEL}"
echo "[view-volume-cleanup-v0] geometry    : max_views=${MAX_VIEWS} effective_scale=${USE_EFFECTIVE_SCALE} recompute_filter3d=${RECOMPUTE_FILTER3D}"
echo "[view-volume-cleanup-v0] mode        : candidate=${CANDIDATE_MODE} short_axis=${STRESS_AXIS_SOURCE} scale_x=${STRESS_SHORT_AXIS_SCALE_FACTOR} axis_to_max=${STRESS_MIN_AXIS_TO_MAX_RATIO}"
echo "[view-volume-cleanup-v0] prune       : quantile=${DELETE_QUANTILE} cap=${MAX_PRUNE_FRACTION} opacity<=${MAX_OPACITY} visible<=${MAX_VISIBLE_FRACTION}"
echo "[view-volume-cleanup-v0] stress gate : impact>=${MIN_STRESS_IMPACT} major>=${STRESS_MAJOR_IMPACT_THRESHOLD} views>=${MIN_STRESS_MAJOR_IMPACT_VIEWS} frac>=${MIN_STRESS_MAJOR_IMPACT_VISIBLE_FRACTION}"

ARGS=(
  --scene_root "${SCENE_ROOT}"
  --mip_model_path "${MIP_MODEL_PATH}"
  --output_model_path "${OUTPUT_MODEL}"
  --iteration "${MIP_ITERATION}"
  --output_iteration "${OUTPUT_ITERATION}"
  --images_subdir "${LR_IMAGES_SUBDIR}"
  --max_views "${MAX_VIEWS}"
  --depth_min "${DEPTH_MIN}"
  --screen_margin_px "${SCREEN_MARGIN_PX}"
  --chunk_size "${CHUNK_SIZE}"
  --use_effective_scale "${USE_EFFECTIVE_SCALE}"
  --recompute_filter3d "${RECOMPUTE_FILTER3D}"
  --surface_query_mode "${SURFACE_QUERY_MODE}"
  --mesh_surface_sample_count "${MESH_SURFACE_SAMPLE_COUNT}"
  --surface_query_chunk_size "${SURFACE_QUERY_CHUNK_SIZE}"
  --surface_distance_threshold "${SURFACE_DISTANCE_THRESHOLD}"
  --candidate_mode "${CANDIDATE_MODE}"
  --delete_quantile "${DELETE_QUANTILE}"
  --min_visible_views "${MIN_VISIBLE_VIEWS}"
  --max_visible_fraction "${MAX_VISIBLE_FRACTION}"
  --max_opacity "${MAX_OPACITY}"
  --stress_axis_source "${STRESS_AXIS_SOURCE}"
  --stress_short_axis_scale_factor "${STRESS_SHORT_AXIS_SCALE_FACTOR}"
  --stress_min_axis_to_max_ratio "${STRESS_MIN_AXIS_TO_MAX_RATIO}"
  --stress_visibility_downsample "${STRESS_VISIBILITY_DOWNSAMPLE}"
  --stress_visibility_topk "${STRESS_VISIBILITY_TOPK}"
  --stress_visibility_max_visible "${STRESS_VISIBILITY_MAX_VISIBLE}"
  --stress_visibility_max_patch_radius "${STRESS_VISIBILITY_MAX_PATCH_RADIUS}"
  --stress_major_impact_threshold "${STRESS_MAJOR_IMPACT_THRESHOLD}"
  --min_axis_alignment "${MIN_AXIS_ALIGNMENT}"
  --min_axis_anisotropy "${MIN_AXIS_ANISOTROPY}"
  --min_ray_thickness_ratio "${MIN_RAY_THICKNESS_RATIO}"
  --min_side_explosion "${MIN_SIDE_EXPLOSION}"
  --min_side_radius_px "${MIN_SIDE_RADIUS_PX}"
  --min_radius_px "${MIN_RADIUS_PX}"
  --min_effective_scale_ratio "${MIN_EFFECTIVE_SCALE_RATIO}"
  --min_volume_radius_ratio "${MIN_VOLUME_RADIUS_RATIO}"
  --min_filter_inflation "${MIN_FILTER_INFLATION}"
  --min_filter_scale_ratio "${MIN_FILTER_SCALE_RATIO}"
  --min_stress_impact "${MIN_STRESS_IMPACT}"
  --min_stress_radius_gain "${MIN_STRESS_RADIUS_GAIN}"
  --min_stress_radius_px "${MIN_STRESS_RADIUS_PX}"
  --min_stress_major_impact_views "${MIN_STRESS_MAJOR_IMPACT_VIEWS}"
  --min_stress_major_impact_visible_fraction "${MIN_STRESS_MAJOR_IMPACT_VISIBLE_FRACTION}"
  --max_prune_fraction "${MAX_PRUNE_FRACTION}"
  --max_prune_count "${MAX_PRUNE_COUNT}"
  --export_deleted_model
)

if [[ -n "${SURFACE_MESH_PATH}" ]]; then
  ARGS+=(--surface_mesh_path "${SURFACE_MESH_PATH}")
fi

if [[ "${REQUIRE_SURFACE_OUTSIDE}" == "1" ]]; then
  ARGS+=(--require_surface_outside)
fi

"${PYTHON_BIN}" -u "${SOF_ROOT}/cleanup_mip_view_aligned_volume_artifacts_v0.py" "${ARGS[@]}"

if [[ "${RUN_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${OUTPUT_MODEL}" \
    --output_dir "${RENDER_DIR}" \
    --images_subdir "${RENDER_IMAGES_SUBDIR}" \
    --iteration "${OUTPUT_ITERATION}" \
    --split "${RENDER_SPLIT}"
fi

if [[ "${RUN_LR_RECOVER}" == "1" ]]; then
  SCENE_NAME="${SCENE_NAME}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
  START_MODEL_PATH="${OUTPUT_MODEL}" \
  START_ITERATION="${OUTPUT_ITERATION}" \
  ANCHOR_MODEL_PATH="${MIP_MODEL_PATH}" \
  ANCHOR_ITERATION="${MIP_ITERATION}" \
  REFINE_PROFILE="${LR_RECOVER_PROFILE}" \
  RUN_NAME="${LR_RECOVER_RUN_NAME}" \
  RUN_RENDER="${LR_RECOVER_RUN_RENDER}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SOF_ROOT}/scripts/run_recover_cleaned_mip_lr_v0_kitchen.sh"
fi

echo "[done] cleaned mip model : ${OUTPUT_MODEL}"
echo "[done] output ply        : ${OUTPUT_MODEL}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
echo "[done] deleted subset    : ${OUTPUT_MODEL}_deleted_view_aligned_volume_artifacts"
echo "[done] summary           : ${OUTPUT_MODEL}/summary.json"
echo "[done] payload           : ${OUTPUT_MODEL}/view_aligned_volume_cleanup_payload.pt"
if [[ "${RUN_RENDER}" == "1" ]]; then
  echo "[done] renders           : ${RENDER_DIR}/${RENDER_SPLIT}/ours_${OUTPUT_ITERATION}/renders"
fi
if [[ "${RUN_LR_RECOVER}" == "1" ]]; then
  echo "[done] lr recover       : ${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${LR_RECOVER_RUN_NAME}"
fi

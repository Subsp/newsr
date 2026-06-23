#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_RUN_TAG="${BASE_RUN_TAG:-}"
if [[ -z "${BASE_MODEL_DIR:-}" ]]; then
  if [[ -n "${BASE_RUN_TAG}" ]]; then
    BASE_MODEL_DIR="${SOF_ROOT}/output/mipsplatting_nosr_layerfreq_cleanup_v0/${SCENE_NAME}/${BASE_RUN_TAG}"
    BASE_ITERATION="${BASE_ITERATION:-32000}"
  else
    BASE_MODEL_DIR="${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0"
    BASE_ITERATION="${BASE_ITERATION:-30000}"
  fi
else
  BASE_ITERATION="${BASE_ITERATION:-30000}"
fi
BASE_LABEL="${BASE_RUN_TAG:-$(basename -- "${BASE_MODEL_DIR}")}"

EVIDENCE_NAME="${EVIDENCE_NAME:-qwen_vosr_sr_hf_effective_verywide_8view_v0}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${SCENE_ASSET_ROOT}/sr_hf_evidence/${EVIDENCE_NAME}}"
PRIMITIVE_DIR="${PRIMITIVE_DIR:-${EVIDENCE_ROOT}/primitives}"
WEIGHT_DIR="${WEIGHT_DIR:-${EVIDENCE_ROOT}/effective_hf_weight}"
RGB_DIR="${RGB_DIR:-${EVIDENCE_ROOT}/effective_hf_carrier_rgb}"
SR_PRIOR_NAME="${SR_PRIOR_NAME:-qwen_steps1_seed42_rcgm_aligned_images2_train244_v0}"
CURVE_IMAGE_DIR="${CURVE_IMAGE_DIR:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${SR_PRIOR_NAME}/fused_priors}"
CURVE_IMAGE_MODE="${CURVE_IMAGE_MODE:-sr_hf_luma}"
CURVE_HIGHPASS_BLUR_RADIUS="${CURVE_HIGHPASS_BLUR_RADIUS:-4.0}"
CURVE_WEIGHT_POWER="${CURVE_WEIGHT_POWER:-1.0}"

OUTPUT_NAME="${OUTPUT_NAME:-${BASE_LABEL}_${EVIDENCE_NAME}_curve_tracks_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/sr_hf_curve_tracks/${OUTPUT_NAME}}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-8}"
OVERWRITE="${OVERWRITE:-0}"
KEEP_KINDS="${KEEP_KINDS:-1,2}"
CURVE_SOURCE="${CURVE_SOURCE:-skeleton}"
SKELETON_THRESHOLD_PERCENTILE="${SKELETON_THRESHOLD_PERCENTILE:-86.0}"
SKELETON_MIN_WEIGHT="${SKELETON_MIN_WEIGHT:-0.025}"
SKELETON_MIN_PATH_PIXELS="${SKELETON_MIN_PATH_PIXELS:-8}"
SKELETON_SAMPLE_STEP_PX="${SKELETON_SAMPLE_STEP_PX:-3.0}"
SKELETON_SMOOTH_WINDOW="${SKELETON_SMOOTH_WINDOW:-3}"
SKELETON_MAX_THINNING_ITERS="${SKELETON_MAX_THINNING_ITERS:-80}"
DENSE_STROKE_ENABLE="${DENSE_STROKE_ENABLE:-1}"
DENSE_STROKE_THRESHOLD_PERCENTILE="${DENSE_STROKE_THRESHOLD_PERCENTILE:-78.0}"
DENSE_STROKE_MIN_STRENGTH="${DENSE_STROKE_MIN_STRENGTH:-0.012}"
DENSE_STROKE_GRID_PX="${DENSE_STROKE_GRID_PX:-2}"
DENSE_STROKE_MAX_PER_VIEW="${DENSE_STROKE_MAX_PER_VIEW:-32768}"
DENSE_STROKE_LENGTH_PX="${DENSE_STROKE_LENGTH_PX:-4.0}"
DENSE_STROKE_SHORT_PX="${DENSE_STROKE_SHORT_PX:-0.55}"
PROFILE_WIDTH_ENABLE="${PROFILE_WIDTH_ENABLE:-1}"
PROFILE_WIDTH_RADIUS_PX="${PROFILE_WIDTH_RADIUS_PX:-6}"
PROFILE_WIDTH_FALLOFF="${PROFILE_WIDTH_FALLOFF:-0.35}"
PROFILE_WIDTH_MIN_PX="${PROFILE_WIDTH_MIN_PX:-0.4}"
PROFILE_WIDTH_MAX_PX="${PROFILE_WIDTH_MAX_PX:-5.0}"
MAX_PRIMITIVES_PER_VIEW="${MAX_PRIMITIVES_PER_VIEW:-32768}"
MIN_SCORE="${MIN_SCORE:-0.05}"
MIN_WEIGHT="${MIN_WEIGHT:-0.01}"
BASE_OPACITY_MIN="${BASE_OPACITY_MIN:-0.02}"
DEPTH_MIN="${DEPTH_MIN:-0.02}"
SEARCH_RADIUS_PX="${SEARCH_RADIUS_PX:-5}"
ENDPOINT_SEARCH_RADIUS_PX="${ENDPOINT_SEARCH_RADIUS_PX:-3}"
REQUIRE_ENDPOINT_MATCH="${REQUIRE_ENDPOINT_MATCH:-0}"
MAX_ENDPOINT_DEPTH_DELTA_PX="${MAX_ENDPOINT_DEPTH_DELTA_PX:-8.0}"
FRONT_OFFSET_PX="${FRONT_OFFSET_PX:-0.25}"
SEGMENT_LENGTH_SCALE="${SEGMENT_LENGTH_SCALE:-2.5}"
SEGMENT_MIN_LENGTH_PX="${SEGMENT_MIN_LENGTH_PX:-2.0}"
SEGMENT_MAX_LENGTH_PX="${SEGMENT_MAX_LENGTH_PX:-18.0}"

MAX_SEGMENTS_FOR_MERGE="${MAX_SEGMENTS_FOR_MERGE:-50000}"
MERGE_RADIUS_PX="${MERGE_RADIUS_PX:-6.0}"
MERGE_RADIUS_ABS="${MERGE_RADIUS_ABS:-0.006}"
MERGE_ANGLE_DEG="${MERGE_ANGLE_DEG:-18.0}"
MERGE_MIN_OVERLAP="${MERGE_MIN_OVERLAP:-0.05}"
MERGE_SAME_VIEW="${MERGE_SAME_VIEW:-1}"
LAYER_BIN_RADIUS_PX="${LAYER_BIN_RADIUS_PX:-8.0}"
LAYER_BIN_RADIUS_ABS="${LAYER_BIN_RADIUS_ABS:-0.008}"
LAYER_DIR_BINS="${LAYER_DIR_BINS:-8}"
LAYER_INCLUDE_KIND="${LAYER_INCLUDE_KIND:-0}"
CANDIDATE_RADIUS_PX="${CANDIDATE_RADIUS_PX:-6.0}"
CANDIDATE_RADIUS_ABS="${CANDIDATE_RADIUS_ABS:-0.006}"
CANDIDATE_REPROJ_RADIUS_PX="${CANDIDATE_REPROJ_RADIUS_PX:-5.0}"
CANDIDATE_DIR_ANGLE_DEG="${CANDIDATE_DIR_ANGLE_DEG:-25.0}"
CANDIDATE_NORMAL_ANGLE_DEG="${CANDIDATE_NORMAL_ANGLE_DEG:-60.0}"
CANDIDATE_DEPTH_DELTA_PX="${CANDIDATE_DEPTH_DELTA_PX:-12.0}"
CANDIDATE_MIN_SURVIVE_VIEWS="${CANDIDATE_MIN_SURVIVE_VIEWS:-2}"
CANDIDATE_PROBATION_MIN_SOURCE_STRENGTH="${CANDIDATE_PROBATION_MIN_SOURCE_STRENGTH:-0.04}"
CANDIDATE_PROBATION_MAX_LINE_RESIDUAL_PX="${CANDIDATE_PROBATION_MAX_LINE_RESIDUAL_PX:-8.0}"
CANDIDATE_KEEP_PROBATION="${CANDIDATE_KEEP_PROBATION:-1}"
TRACK_BUILD_MODE="${TRACK_BUILD_MODE:-candidate_graph}"
MIN_TRACK_SEGMENTS="${MIN_TRACK_SEGMENTS:-1}"
MIN_TRACK_VIEWS="${MIN_TRACK_VIEWS:-1}"
STRONG_TRACK_MIN_VIEWS="${STRONG_TRACK_MIN_VIEWS:-2}"
TRACK_MIN_DIR_CONSISTENCY="${TRACK_MIN_DIR_CONSISTENCY:-0.15}"
TRACK_MAX_LINE_RESIDUAL_PX="${TRACK_MAX_LINE_RESIDUAL_PX:-10.0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-8}"
MAX_DRAW_SEGMENTS="${MAX_DRAW_SEGMENTS:-32768}"

REQUIRED_PATHS=("${BASE_MODEL_DIR}" "${BASE_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply" "${PRIMITIVE_DIR}")
if [[ "${CURVE_SOURCE}" == "skeleton" ]]; then
  REQUIRED_PATHS+=("${WEIGHT_DIR}")
  if [[ "${CURVE_IMAGE_MODE}" != "weight" ]]; then
    REQUIRED_PATHS+=("${CURVE_IMAGE_DIR}")
  fi
fi
for required in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[sr-hf-curve-tracks-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

ARGS=(
  --base_model_dir "${BASE_MODEL_DIR}"
  --base_iteration "${BASE_ITERATION}"
  --primitive_dir "${PRIMITIVE_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --curve_image_dir "${CURVE_IMAGE_DIR}"
  --curve_image_mode "${CURVE_IMAGE_MODE}"
  --curve_highpass_blur_radius "${CURVE_HIGHPASS_BLUR_RADIUS}"
  --curve_weight_power "${CURVE_WEIGHT_POWER}"
  --match_policy "${MATCH_POLICY}"
  --limit "${LIMIT}"
  --curve_source "${CURVE_SOURCE}"
  --skeleton_threshold_percentile "${SKELETON_THRESHOLD_PERCENTILE}"
  --skeleton_min_weight "${SKELETON_MIN_WEIGHT}"
  --skeleton_min_path_pixels "${SKELETON_MIN_PATH_PIXELS}"
  --skeleton_sample_step_px "${SKELETON_SAMPLE_STEP_PX}"
  --skeleton_smooth_window "${SKELETON_SMOOTH_WINDOW}"
  --skeleton_max_thinning_iters "${SKELETON_MAX_THINNING_ITERS}"
  --dense_stroke_threshold_percentile "${DENSE_STROKE_THRESHOLD_PERCENTILE}"
  --dense_stroke_min_strength "${DENSE_STROKE_MIN_STRENGTH}"
  --dense_stroke_grid_px "${DENSE_STROKE_GRID_PX}"
  --dense_stroke_max_per_view "${DENSE_STROKE_MAX_PER_VIEW}"
  --dense_stroke_length_px "${DENSE_STROKE_LENGTH_PX}"
  --dense_stroke_short_px "${DENSE_STROKE_SHORT_PX}"
  --profile_width_radius_px "${PROFILE_WIDTH_RADIUS_PX}"
  --profile_width_falloff "${PROFILE_WIDTH_FALLOFF}"
  --profile_width_min_px "${PROFILE_WIDTH_MIN_PX}"
  --profile_width_max_px "${PROFILE_WIDTH_MAX_PX}"
  --keep_kinds "${KEEP_KINDS}"
  --max_primitives_per_view "${MAX_PRIMITIVES_PER_VIEW}"
  --min_score "${MIN_SCORE}"
  --min_weight "${MIN_WEIGHT}"
  --base_opacity_min "${BASE_OPACITY_MIN}"
  --depth_min "${DEPTH_MIN}"
  --search_radius_px "${SEARCH_RADIUS_PX}"
  --endpoint_search_radius_px "${ENDPOINT_SEARCH_RADIUS_PX}"
  --max_endpoint_depth_delta_px "${MAX_ENDPOINT_DEPTH_DELTA_PX}"
  --front_offset_px "${FRONT_OFFSET_PX}"
  --segment_length_scale "${SEGMENT_LENGTH_SCALE}"
  --segment_min_length_px "${SEGMENT_MIN_LENGTH_PX}"
  --segment_max_length_px "${SEGMENT_MAX_LENGTH_PX}"
  --max_segments_for_merge "${MAX_SEGMENTS_FOR_MERGE}"
  --merge_radius_px "${MERGE_RADIUS_PX}"
  --merge_radius_abs "${MERGE_RADIUS_ABS}"
  --merge_angle_deg "${MERGE_ANGLE_DEG}"
  --merge_min_overlap "${MERGE_MIN_OVERLAP}"
  --layer_bin_radius_px "${LAYER_BIN_RADIUS_PX}"
  --layer_bin_radius_abs "${LAYER_BIN_RADIUS_ABS}"
  --layer_dir_bins "${LAYER_DIR_BINS}"
  --candidate_radius_px "${CANDIDATE_RADIUS_PX}"
  --candidate_radius_abs "${CANDIDATE_RADIUS_ABS}"
  --candidate_reproj_radius_px "${CANDIDATE_REPROJ_RADIUS_PX}"
  --candidate_dir_angle_deg "${CANDIDATE_DIR_ANGLE_DEG}"
  --candidate_normal_angle_deg "${CANDIDATE_NORMAL_ANGLE_DEG}"
  --candidate_depth_delta_px "${CANDIDATE_DEPTH_DELTA_PX}"
  --candidate_min_survive_views "${CANDIDATE_MIN_SURVIVE_VIEWS}"
  --candidate_probation_min_source_strength "${CANDIDATE_PROBATION_MIN_SOURCE_STRENGTH}"
  --candidate_probation_max_line_residual_px "${CANDIDATE_PROBATION_MAX_LINE_RESIDUAL_PX}"
  --track_build_mode "${TRACK_BUILD_MODE}"
  --min_track_segments "${MIN_TRACK_SEGMENTS}"
  --min_track_views "${MIN_TRACK_VIEWS}"
  --strong_track_min_views "${STRONG_TRACK_MIN_VIEWS}"
  --track_min_dir_consistency "${TRACK_MIN_DIR_CONSISTENCY}"
  --track_max_line_residual_px "${TRACK_MAX_LINE_RESIDUAL_PX}"
  --debug_limit "${DEBUG_LIMIT}"
  --max_draw_segments "${MAX_DRAW_SEGMENTS}"
)

if [[ -d "${WEIGHT_DIR}" ]]; then
  ARGS+=(--weight_dir "${WEIGHT_DIR}")
fi
if [[ -d "${RGB_DIR}" ]]; then
  ARGS+=(--rgb_dir "${RGB_DIR}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${REQUIRE_ENDPOINT_MATCH}" == "1" ]]; then
  ARGS+=(--require_endpoint_match)
fi
if [[ "${DENSE_STROKE_ENABLE}" == "1" ]]; then
  ARGS+=(--dense_stroke_enable)
fi
if [[ "${PROFILE_WIDTH_ENABLE}" == "1" ]]; then
  ARGS+=(--profile_width_enable)
fi
if [[ "${MERGE_SAME_VIEW}" == "1" ]]; then
  ARGS+=(--merge_same_view)
fi
if [[ "${LAYER_INCLUDE_KIND}" == "1" ]]; then
  ARGS+=(--layer_include_kind)
fi
if [[ "${CANDIDATE_KEEP_PROBATION}" == "1" ]]; then
  ARGS+=(--candidate_keep_probation)
fi

echo "[sr-hf-curve-tracks-v0] base      : ${BASE_MODEL_DIR}"
echo "[sr-hf-curve-tracks-v0] primitives: ${PRIMITIVE_DIR}"
echo "[sr-hf-curve-tracks-v0] weight    : ${WEIGHT_DIR}"
echo "[sr-hf-curve-tracks-v0] rgb       : ${RGB_DIR}"
echo "[sr-hf-curve-tracks-v0] curve img : ${CURVE_IMAGE_DIR} mode=${CURVE_IMAGE_MODE}"
echo "[sr-hf-curve-tracks-v0] output    : ${OUTPUT_ROOT}"
echo "[sr-hf-curve-tracks-v0] limit     : ${LIMIT}"
echo "[sr-hf-curve-tracks-v0] source    : ${CURVE_SOURCE} track_mode=${TRACK_BUILD_MODE}"
echo "[sr-hf-curve-tracks-v0] dense     : enable=${DENSE_STROKE_ENABLE} grid=${DENSE_STROKE_GRID_PX}px max=${DENSE_STROKE_MAX_PER_VIEW}"
echo "[sr-hf-curve-tracks-v0] width     : enable=${PROFILE_WIDTH_ENABLE} radius=${PROFILE_WIDTH_RADIUS_PX}px falloff=${PROFILE_WIDTH_FALLOFF}"
echo "[sr-hf-curve-tracks-v0] merge     : radius=${MERGE_RADIUS_PX}px/${MERGE_RADIUS_ABS} angle=${MERGE_ANGLE_DEG} same_view=${MERGE_SAME_VIEW}"
echo "[sr-hf-curve-tracks-v0] layer     : radius=${LAYER_BIN_RADIUS_PX}px/${LAYER_BIN_RADIUS_ABS} dir_bins=${LAYER_DIR_BINS} include_kind=${LAYER_INCLUDE_KIND}"
echo "[sr-hf-curve-tracks-v0] candidate : radius=${CANDIDATE_RADIUS_PX}px/${CANDIDATE_RADIUS_ABS} reproj=${CANDIDATE_REPROJ_RADIUS_PX}px survive_views=${CANDIDATE_MIN_SURVIVE_VIEWS} keep_probation=${CANDIDATE_KEEP_PROBATION}"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_sr_hf_curve_tracks_v0.py" "${ARGS[@]}"

echo "[sr-hf-curve-tracks-v0] shallow outputs:"
echo "  ${OUTPUT_ROOT}/summary.json"
echo "  ${OUTPUT_ROOT}/sr_hf_curve_tracks_v0.npz"
echo "  ${OUTPUT_ROOT}/tracks_keep_v0.obj"
echo "  ${OUTPUT_ROOT}/tracks_strong_v0.obj"
echo "  ${OUTPUT_ROOT}/segment_overlay"
echo "  ${OUTPUT_ROOT}/track_projection"

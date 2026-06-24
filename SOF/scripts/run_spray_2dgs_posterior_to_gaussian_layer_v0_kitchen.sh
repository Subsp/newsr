#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting}"

BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/${BASE_EXPERIMENT_NAME}}"
BASE_ITERATION="${BASE_ITERATION:-30000}"
BASE_PLY="${BASE_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply"

EVIDENCE_NAME="${EVIDENCE_NAME:-qwen_vosr_sr_hf_effective_8view_v0}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${SCENE_ASSET_ROOT}/sr_hf_evidence/${EVIDENCE_NAME}}"
EVIDENCE_RGB_DIR="${EVIDENCE_RGB_DIR:-${EVIDENCE_ROOT}/effective_hf_carrier_rgb}"
EVIDENCE_WEIGHT_DIR="${EVIDENCE_WEIGHT_DIR:-${EVIDENCE_ROOT}/effective_hf_weight}"

CARRIER_NAME="${CARRIER_NAME:-qwen_vosr_effective_hf_2dgs_one_v0}"
CARRIER_ROOT="${CARRIER_ROOT:-${SOF_ROOT}/output/2dgs_sr_hf_evidence_carrier/${CARRIER_NAME}}"
PRIMITIVE_DIR="${PRIMITIVE_DIR:-${CARRIER_ROOT}/primitives}"
CARRIER_RENDER_DIR="${CARRIER_RENDER_DIR:-${CARRIER_ROOT}/evidence_render}"

OUTPUT_NAME="${OUTPUT_NAME:-${BASE_EXPERIMENT_NAME}_spray_2dgs_posterior_v0}"
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:-${SOF_ROOT}/output/mipsplatting_2dgs_posterior_spray_v0/${SCENE_NAME}/${OUTPUT_NAME}}"
NEWBORN_MODEL_DIR="${NEWBORN_MODEL_DIR:-${OUTPUT_MODEL_DIR}_newborn_only}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-${BASE_ITERATION}}"
MERGE_SCRIPT="${MERGE_SCRIPT:-}"

MATCH_POLICY="${MATCH_POLICY:-order_if_needed}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"

MAX_PRIMITIVES_PER_VIEW="${MAX_PRIMITIVES_PER_VIEW:-32768}"
MAX_TOTAL_NEWBORN="${MAX_TOTAL_NEWBORN:-0}"
MIN_WEIGHT="${MIN_WEIGHT:-0.02}"
MIN_Q="${MIN_Q:-0.01}"
MIN_PRIMITIVE_OPACITY="${MIN_PRIMITIVE_OPACITY:-0.0}"
FIT_ERROR_TAU="${FIT_ERROR_TAU:-0.08}"
FIT_ERROR_FLOOR="${FIT_ERROR_FLOOR:-0.15}"

BASE_OPACITY_MIN="${BASE_OPACITY_MIN:-0.02}"
DEPTH_MIN="${DEPTH_MIN:-0.02}"
LAYER_SEARCH_RADIUS_PX="${LAYER_SEARCH_RADIUS_PX:-3}"
FOOTPRINT_SAMPLE_SCALE="${FOOTPRINT_SAMPLE_SCALE:-1.25}"
MODE_DEPTH_REL="${MODE_DEPTH_REL:-0.018}"
MODE_DEPTH_ABS="${MODE_DEPTH_ABS:-0.006}"
MODE_POSITION_RADIUS="${MODE_POSITION_RADIUS:-0.018}"
MIN_MODE_DOMINANCE="${MIN_MODE_DOMINANCE:-0.42}"
MAX_MODE_ENTROPY="${MAX_MODE_ENTROPY:-0.78}"

ASSOCIATION_RADIUS_PX="${ASSOCIATION_RADIUS_PX:-7.0}"
ASSOCIATION_CELL_PX="${ASSOCIATION_CELL_PX:-8.0}"
ASSOCIATION_COLOR_WEIGHT="${ASSOCIATION_COLOR_WEIGHT:-0.35}"
ASSOCIATION_SHAPE_WEIGHT="${ASSOCIATION_SHAPE_WEIGHT:-0.25}"
ASSOCIATION_MAX_COST="${ASSOCIATION_MAX_COST:-3.25}"
MIN_CLUSTER_VIEWS="${MIN_CLUSTER_VIEWS:-2}"
MIN_CAMERA_ANGLE_DEG="${MIN_CAMERA_ANGLE_DEG:-1.5}"

LOCALIZATION_SIGMA_PX="${LOCALIZATION_SIGMA_PX:-1.4}"
LOCALIZATION_FOOTPRINT_BETA="${LOCALIZATION_FOOTPRINT_BETA:-0.08}"
SURFACE_SIGMA="${SURFACE_SIGMA:-0.006}"
TANGENT_PRIOR_WEIGHT="${TANGENT_PRIOR_WEIGHT:-0.002}"
MAP_ITERATIONS="${MAP_ITERATIONS:-6}"
MAP_HUBER_PX="${MAP_HUBER_PX:-4.0}"
MAP_DAMPING="${MAP_DAMPING:-0.0001}"
MAX_REPROJ_RMS_PX="${MAX_REPROJ_RMS_PX:-3.8}"
MAX_CENTER_STD="${MAX_CENTER_STD:-0.045}"
MAX_HESSIAN_COND="${MAX_HESSIAN_COND:-250000}"

SCREEN_FILTER_SIGMA_PX="${SCREEN_FILTER_SIGMA_PX:-0.45}"
EXTRACT_SIGMA_PX="${EXTRACT_SIGMA_PX:-0.35}"
SCALE_MULTIPLIER="${SCALE_MULTIPLIER:-1.0}"
SCALE_MIN="${SCALE_MIN:-0.0004}"
SCALE_MAX="${SCALE_MAX:-0.009}"
NORMAL_SCALE_RATIO="${NORMAL_SCALE_RATIO:-0.20}"
NORMAL_SCALE_MIN="${NORMAL_SCALE_MIN:-0.00025}"
NORMAL_SCALE_MAX="${NORMAL_SCALE_MAX:-0.0016}"

COLOR_MODE="${COLOR_MODE:-base_anchor_additive}"
COLOR_GAIN="${COLOR_GAIN:-0.32}"
OPACITY_FLOOR="${OPACITY_FLOOR:-0.006}"
OPACITY_SCALE="${OPACITY_SCALE:-0.055}"
OPACITY_POWER="${OPACITY_POWER:-0.70}"
OPACITY_MIN="${OPACITY_MIN:-0.004}"
OPACITY_MAX="${OPACITY_MAX:-0.075}"
WRITE_CPU_MERGED_PREVIEW="${WRITE_CPU_MERGED_PREVIEW:-0}"

for required in "${BASE_MODEL_DIR}" "${BASE_PLY}" "${PRIMITIVE_DIR}" "${EVIDENCE_RGB_DIR}" "${EVIDENCE_WEIGHT_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[2dgs-posterior-v0] required path not found: ${required}" >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_MODEL_DIR}" "${NEWBORN_MODEL_DIR}"
fi

echo "[2dgs-posterior-v0] base model : ${BASE_MODEL_DIR}"
echo "[2dgs-posterior-v0] base ply   : ${BASE_PLY}"
echo "[2dgs-posterior-v0] primitives : ${PRIMITIVE_DIR}"
echo "[2dgs-posterior-v0] evidence   : ${EVIDENCE_RGB_DIR}"
echo "[2dgs-posterior-v0] fit render : ${CARRIER_RENDER_DIR}"
echo "[2dgs-posterior-v0] weight     : ${EVIDENCE_WEIGHT_DIR}"
echo "[2dgs-posterior-v0] newborn   : ${NEWBORN_MODEL_DIR}"
echo "[2dgs-posterior-v0] output    : ${OUTPUT_MODEL_DIR}"

ARGS=(
  --base_model_dir "${BASE_MODEL_DIR}"
  --base_iteration "${BASE_ITERATION}"
  --primitive_dir "${PRIMITIVE_DIR}"
  --carrier_rgb_dir "${EVIDENCE_RGB_DIR}"
  --carrier_weight_dir "${EVIDENCE_WEIGHT_DIR}"
  --output_model_dir "${OUTPUT_MODEL_DIR}"
  --newborn_model_dir "${NEWBORN_MODEL_DIR}"
  --match_policy "${MATCH_POLICY}"
  --limit "${LIMIT}"
  --max_primitives_per_view "${MAX_PRIMITIVES_PER_VIEW}"
  --max_total_newborn "${MAX_TOTAL_NEWBORN}"
  --min_weight "${MIN_WEIGHT}"
  --min_q "${MIN_Q}"
  --min_primitive_opacity "${MIN_PRIMITIVE_OPACITY}"
  --fit_error_tau "${FIT_ERROR_TAU}"
  --fit_error_floor "${FIT_ERROR_FLOOR}"
  --base_opacity_min "${BASE_OPACITY_MIN}"
  --depth_min "${DEPTH_MIN}"
  --layer_search_radius_px "${LAYER_SEARCH_RADIUS_PX}"
  --footprint_sample_scale "${FOOTPRINT_SAMPLE_SCALE}"
  --mode_depth_rel "${MODE_DEPTH_REL}"
  --mode_depth_abs "${MODE_DEPTH_ABS}"
  --mode_position_radius "${MODE_POSITION_RADIUS}"
  --min_mode_dominance "${MIN_MODE_DOMINANCE}"
  --max_mode_entropy "${MAX_MODE_ENTROPY}"
  --association_radius_px "${ASSOCIATION_RADIUS_PX}"
  --association_cell_px "${ASSOCIATION_CELL_PX}"
  --association_color_weight "${ASSOCIATION_COLOR_WEIGHT}"
  --association_shape_weight "${ASSOCIATION_SHAPE_WEIGHT}"
  --association_max_cost "${ASSOCIATION_MAX_COST}"
  --min_cluster_views "${MIN_CLUSTER_VIEWS}"
  --min_camera_angle_deg "${MIN_CAMERA_ANGLE_DEG}"
  --localization_sigma_px "${LOCALIZATION_SIGMA_PX}"
  --localization_footprint_beta "${LOCALIZATION_FOOTPRINT_BETA}"
  --surface_sigma "${SURFACE_SIGMA}"
  --tangent_prior_weight "${TANGENT_PRIOR_WEIGHT}"
  --map_iterations "${MAP_ITERATIONS}"
  --map_huber_px "${MAP_HUBER_PX}"
  --map_damping "${MAP_DAMPING}"
  --max_reproj_rms_px "${MAX_REPROJ_RMS_PX}"
  --max_center_std "${MAX_CENTER_STD}"
  --max_hessian_cond "${MAX_HESSIAN_COND}"
  --screen_filter_sigma_px "${SCREEN_FILTER_SIGMA_PX}"
  --extract_sigma_px "${EXTRACT_SIGMA_PX}"
  --scale_multiplier "${SCALE_MULTIPLIER}"
  --scale_min "${SCALE_MIN}"
  --scale_max "${SCALE_MAX}"
  --normal_scale_ratio "${NORMAL_SCALE_RATIO}"
  --normal_scale_min "${NORMAL_SCALE_MIN}"
  --normal_scale_max "${NORMAL_SCALE_MAX}"
  --color_mode "${COLOR_MODE}"
  --color_gain "${COLOR_GAIN}"
  --opacity_floor "${OPACITY_FLOOR}"
  --opacity_scale "${OPACITY_SCALE}"
  --opacity_power "${OPACITY_POWER}"
  --opacity_min "${OPACITY_MIN}"
  --opacity_max "${OPACITY_MAX}"
)
if [[ -d "${CARRIER_RENDER_DIR}" ]]; then
  ARGS+=(--carrier_render_dir "${CARRIER_RENDER_DIR}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${WRITE_CPU_MERGED_PREVIEW}" == "1" ]]; then
  ARGS+=(--write_cpu_merged_preview)
fi

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/spray_2dgs_posterior_to_gaussian_layer_v0.py" "${ARGS[@]}"

NEWBORN_PLY="${NEWBORN_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply"
if [[ ! -f "${NEWBORN_PLY}" ]]; then
  echo "[2dgs-posterior-v0] newborn PLY not found after fusion: ${NEWBORN_PLY}" >&2
  exit 1
fi

cd "${SOF_ROOT}"
export PYTHONPATH="${SOF_ROOT}:${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${MERGE_SCRIPT}" ]]; then
  if [[ -f "${SOF_ROOT}/merge_gaussian_plys_v0.py" ]]; then
    MERGE_SCRIPT="${SOF_ROOT}/merge_gaussian_plys_v0.py"
  else
    MERGE_SCRIPT="${SOF_ROOT}/scripts/merge_gaussian_plys_v0.py"
  fi
fi
if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "[2dgs-posterior-v0] merge script not found: ${MERGE_SCRIPT}" >&2
  exit 1
fi

echo "[2dgs-posterior-v0] merge     : ${MERGE_SCRIPT}"
"${PYTHON_BIN}" "${MERGE_SCRIPT}" \
  --source_path "${SCENE_ROOT}" \
  --model_path "${OUTPUT_MODEL_DIR}" \
  --images images_2 \
  --resolution 1 \
  --eval \
  --base_ply "${BASE_PLY}" \
  --extra_ply "${NEWBORN_PLY}" \
  --copy_config_from "${BASE_MODEL_DIR}" \
  --output_model_path "${OUTPUT_MODEL_DIR}" \
  --output_iteration "${OUTPUT_ITERATION}"

NEWBORN_METADATA="${NEWBORN_MODEL_DIR}/point_cloud/iteration_${BASE_ITERATION}/sprayed_2dgs_posterior_metadata_v0.npz"
MERGED_METADATA="${OUTPUT_MODEL_DIR}/point_cloud/iteration_${OUTPUT_ITERATION}/sprayed_2dgs_posterior_metadata_v0.npz"
if [[ -f "${NEWBORN_METADATA}" ]]; then
  mkdir -p "$(dirname -- "${MERGED_METADATA}")"
  cp "${NEWBORN_METADATA}" "${MERGED_METADATA}"
  echo "[2dgs-posterior-v0] metadata:"
  echo "  ${MERGED_METADATA}"
fi

echo "[2dgs-posterior-v0] done model:"
echo "  ${OUTPUT_MODEL_DIR}"
echo "[2dgs-posterior-v0] merged ply:"
echo "  ${OUTPUT_MODEL_DIR}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
echo "[2dgs-posterior-v0] tags:"
echo "  ${OUTPUT_MODEL_DIR}/point_cloud/iteration_${OUTPUT_ITERATION}/gaussian_tags.pt"
echo "[2dgs-posterior-v0] summary:"
echo "  ${OUTPUT_MODEL_DIR}/spray_2dgs_posterior_to_gaussian_layer_v0_summary.json"

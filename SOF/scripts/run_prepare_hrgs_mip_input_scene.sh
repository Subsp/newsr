#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
SOF_REF_MODEL="${SOF_REF_MODEL:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images2_v1/sof30k}"

OUTPUT_GROUP="${OUTPUT_GROUP:-${MIP_EXPERIMENT_GROUP}}"
OUTPUT_NAME="${OUTPUT_NAME:-${MIP_EXPERIMENT_NAME}_sof_native_input}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SCENE_ASSET_ROOT}/${OUTPUT_GROUP}/${OUTPUT_NAME}}"

MIP_ITERATION="${MIP_ITERATION:-30000}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:--1}"
USE_AABB_FILTER="${USE_AABB_FILTER:-1}"
AABB_MARGIN_RATIO="${AABB_MARGIN_RATIO:-0.25}"
FILTER_MODE="${FILTER_MODE:-keep}"
FILTER_CONSTANT="${FILTER_CONSTANT:-0.0}"
OPACITY_RAW_MIN="${OPACITY_RAW_MIN:--8.0}"
OPACITY_RAW_MAX="${OPACITY_RAW_MAX:-1.0}"
SCALE_MIN_RATIO="${SCALE_MIN_RATIO:-1e-5}"
SCALE_MAX_RATIO="${SCALE_MAX_RATIO:-1e-2}"
SCALE_CLAMP_MODE="${SCALE_CLAMP_MODE:-both}"
OPACITY_COMPENSATE_SCALE_SHRINK="${OPACITY_COMPENSATE_SCALE_SHRINK:-area}"
OPACITY_COMPENSATION_POWER="${OPACITY_COMPENSATION_POWER:-1.0}"
MIN_OPACITY_COMPENSATION_SCALE="${MIN_OPACITY_COMPENSATION_SCALE:-0.05}"
MAX_COMPENSATED_OPACITY="${MAX_COMPENSATED_OPACITY:-0.95}"
FEATURE_CLIP="${FEATURE_CLIP:-10.0}"
INIT_REPAIR_MODE="${INIT_REPAIR_MODE:-none}"
INIT_REPAIR_MAX_FRACTION="${INIT_REPAIR_MAX_FRACTION:-0.04}"
INIT_REPAIR_MAX_COUNT="${INIT_REPAIR_MAX_COUNT:-50000}"
INIT_REPAIR_MIN_OPACITY="${INIT_REPAIR_MIN_OPACITY:-0.04}"
INIT_REPAIR_MIN_EFFECTIVE_SCALE_RATIO="${INIT_REPAIR_MIN_EFFECTIVE_SCALE_RATIO:-0.003}"
INIT_REPAIR_MIN_VOLUME_RADIUS_RATIO="${INIT_REPAIR_MIN_VOLUME_RADIUS_RATIO:-0.0015}"
INIT_REPAIR_MIN_FILTER_SCALE_RATIO="${INIT_REPAIR_MIN_FILTER_SCALE_RATIO:-0.75}"
INIT_REPAIR_MIN_FULL_ANISOTROPY="${INIT_REPAIR_MIN_FULL_ANISOTROPY:-0.0}"
INIT_REPAIR_SPLIT_COUNT="${INIT_REPAIR_SPLIT_COUNT:-4}"
INIT_REPAIR_CHILD_LAYOUT="${INIT_REPAIR_CHILD_LAYOUT:-grid}"
INIT_REPAIR_CHILD_SCALE_MULTIPLIER="${INIT_REPAIR_CHILD_SCALE_MULTIPLIER:-0.55}"
INIT_REPAIR_CHILD_MAJOR_SCALE_MULTIPLIER="${INIT_REPAIR_CHILD_MAJOR_SCALE_MULTIPLIER:--1.0}"
INIT_REPAIR_CHILD_OPACITY_SCALE="${INIT_REPAIR_CHILD_OPACITY_SCALE:-0.75}"
INIT_REPAIR_ENERGY_CONSERVE_MODE="${INIT_REPAIR_ENERGY_CONSERVE_MODE:-none}"
INIT_REPAIR_FILTER_SCALE="${INIT_REPAIR_FILTER_SCALE:-0.25}"
INIT_REPAIR_FILTER_CAP_RATIO="${INIT_REPAIR_FILTER_CAP_RATIO:-0.0015}"
INIT_REPAIR_OFFSET_SCALE="${INIT_REPAIR_OFFSET_SCALE:-0.45}"
INIT_REPAIR_BRIGHT_MAX_FRACTION="${INIT_REPAIR_BRIGHT_MAX_FRACTION:-0.01}"
INIT_REPAIR_BRIGHT_MAX_COUNT="${INIT_REPAIR_BRIGHT_MAX_COUNT:-15000}"
INIT_REPAIR_BRIGHT_MIN_OPACITY="${INIT_REPAIR_BRIGHT_MIN_OPACITY:-0.06}"
INIT_REPAIR_BRIGHT_MAX_EFFECTIVE_SCALE_RATIO="${INIT_REPAIR_BRIGHT_MAX_EFFECTIVE_SCALE_RATIO:-0.0025}"
INIT_REPAIR_BRIGHT_LUMA_QUANTILE="${INIT_REPAIR_BRIGHT_LUMA_QUANTILE:-0.995}"
INIT_REPAIR_BRIGHT_MIN_LOCAL_LUMA_RATIO="${INIT_REPAIR_BRIGHT_MIN_LOCAL_LUMA_RATIO:-1.8}"
INIT_REPAIR_BRIGHT_MIN_COLOR_DELTA="${INIT_REPAIR_BRIGHT_MIN_COLOR_DELTA:-0.18}"
INIT_REPAIR_BRIGHT_NEIGHBOR_K="${INIT_REPAIR_BRIGHT_NEIGHBOR_K:-8}"
INIT_REPAIR_BRIGHT_EXPAND_SCALE_MULTIPLIER="${INIT_REPAIR_BRIGHT_EXPAND_SCALE_MULTIPLIER:-1.35}"
INIT_REPAIR_BRIGHT_SMALLEST_AXIS_SCALE_MULTIPLIER="${INIT_REPAIR_BRIGHT_SMALLEST_AXIS_SCALE_MULTIPLIER:-1.0}"
INIT_REPAIR_BRIGHT_OPACITY_SCALE="${INIT_REPAIR_BRIGHT_OPACITY_SCALE:-0.6}"
INIT_REPAIR_BRIGHT_DC_SCALE="${INIT_REPAIR_BRIGHT_DC_SCALE:-0.82}"
INIT_REPAIR_BRIGHT_REST_SCALE="${INIT_REPAIR_BRIGHT_REST_SCALE:-0.4}"
INIT_REPAIR_BRIGHT_FILTER_SCALE="${INIT_REPAIR_BRIGHT_FILTER_SCALE:-0.5}"
INIT_REPAIR_BRIGHT_FILTER_CAP_RATIO="${INIT_REPAIR_BRIGHT_FILTER_CAP_RATIO:-0.001}"
RUN_RENDER_SANITY="${RUN_RENDER_SANITY:-1}"
RENDER_SANITY_IMAGES_SUBDIR="${RENDER_SANITY_IMAGES_SUBDIR:-images_2}"
RENDER_SANITY_RESOLUTION="${RENDER_SANITY_RESOLUTION:-1}"
DEBUG_DUMP_PREPARE_STAGES="${DEBUG_DUMP_PREPARE_STAGES:-0}"
DEBUG_DUMP_CANONICALIZE_SUBSTAGES="${DEBUG_DUMP_CANONICALIZE_SUBSTAGES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

export PYTHONUNBUFFERED=1

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

for path in "${SCENE_ROOT}" "${SCENE_ASSET_ROOT}" "${MIP_MODEL_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[prepare-mip-input] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[prepare-mip-input] scene root        : ${SCENE_ROOT}"
echo "[prepare-mip-input] asset root        : ${SCENE_ASSET_ROOT}"
echo "[prepare-mip-input] mip model path    : ${MIP_MODEL_PATH}"
echo "[prepare-mip-input] sof ref model     : ${SOF_REF_MODEL}"
echo "[prepare-mip-input] output model path : ${OUTPUT_MODEL_PATH}"
echo "[prepare-mip-input] mip iteration     : ${MIP_ITERATION}"
echo "[prepare-mip-input] filter mode       : ${FILTER_MODE}"
echo "[prepare-mip-input] scale clamp       : ${SCALE_CLAMP_MODE} min=${SCALE_MIN_RATIO} max=${SCALE_MAX_RATIO}"
echo "[prepare-mip-input] opacity comp      : ${OPACITY_COMPENSATE_SCALE_SHRINK} power=${OPACITY_COMPENSATION_POWER}"
echo "[prepare-mip-input] init repair       : ${INIT_REPAIR_MODE} max_frac=${INIT_REPAIR_MAX_FRACTION} split=${INIT_REPAIR_SPLIT_COUNT}"
echo "[prepare-mip-input] init split layout : ${INIT_REPAIR_CHILD_LAYOUT} scale=${INIT_REPAIR_CHILD_SCALE_MULTIPLIER} major=${INIT_REPAIR_CHILD_MAJOR_SCALE_MULTIPLIER} energy=${INIT_REPAIR_ENERGY_CONSERVE_MODE}"
echo "[prepare-mip-input] bright soften     : max_frac=${INIT_REPAIR_BRIGHT_MAX_FRACTION} luma_q=${INIT_REPAIR_BRIGHT_LUMA_QUANTILE} local_ratio=${INIT_REPAIR_BRIGHT_MIN_LOCAL_LUMA_RATIO}"
echo "[prepare-mip-input] debug stage dump  : ${DEBUG_DUMP_PREPARE_STAGES}"
echo "[prepare-mip-input] canonical substg  : ${DEBUG_DUMP_CANONICALIZE_SUBSTAGES}"
echo "[prepare-mip-input] conda env         : ${CONDA_ENV_NAME:-<none>}"
echo "[prepare-mip-input] render sanity     : ${RUN_RENDER_SANITY}"

CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/prepare_mipsplatting_sof_input_field.py"
  --mip_model_path "${MIP_MODEL_PATH}" \
  --scene_root "${SCENE_ROOT}" \
  --sof_ref_model "${SOF_REF_MODEL}" \
  --output_model_path "${OUTPUT_MODEL_PATH}" \
  --iteration "${MIP_ITERATION}" \
  --output_iteration "${OUTPUT_ITERATION}" \
  --filter_mode "${FILTER_MODE}" \
  --filter_constant "${FILTER_CONSTANT}" \
  --opacity_raw_min "${OPACITY_RAW_MIN}" \
  --opacity_raw_max "${OPACITY_RAW_MAX}" \
  --scale_min_ratio "${SCALE_MIN_RATIO}" \
  --scale_max_ratio "${SCALE_MAX_RATIO}" \
  --scale_clamp_mode "${SCALE_CLAMP_MODE}" \
  --opacity_compensate_scale_shrink "${OPACITY_COMPENSATE_SCALE_SHRINK}" \
  --opacity_compensation_power "${OPACITY_COMPENSATION_POWER}" \
  --min_opacity_compensation_scale "${MIN_OPACITY_COMPENSATION_SCALE}" \
  --max_compensated_opacity "${MAX_COMPENSATED_OPACITY}" \
  --feature_clip "${FEATURE_CLIP}" \
  --aabb_margin_ratio "${AABB_MARGIN_RATIO}" \
  --init_repair_mode "${INIT_REPAIR_MODE}" \
  --init_repair_max_fraction "${INIT_REPAIR_MAX_FRACTION}" \
  --init_repair_max_count "${INIT_REPAIR_MAX_COUNT}" \
  --init_repair_min_opacity "${INIT_REPAIR_MIN_OPACITY}" \
  --init_repair_min_effective_scale_ratio "${INIT_REPAIR_MIN_EFFECTIVE_SCALE_RATIO}" \
  --init_repair_min_volume_radius_ratio "${INIT_REPAIR_MIN_VOLUME_RADIUS_RATIO}" \
  --init_repair_min_filter_scale_ratio "${INIT_REPAIR_MIN_FILTER_SCALE_RATIO}" \
  --init_repair_min_full_anisotropy "${INIT_REPAIR_MIN_FULL_ANISOTROPY}" \
  --init_repair_split_count "${INIT_REPAIR_SPLIT_COUNT}" \
  --init_repair_child_layout "${INIT_REPAIR_CHILD_LAYOUT}" \
  --init_repair_child_scale_multiplier "${INIT_REPAIR_CHILD_SCALE_MULTIPLIER}" \
  --init_repair_child_major_scale_multiplier "${INIT_REPAIR_CHILD_MAJOR_SCALE_MULTIPLIER}" \
  --init_repair_child_opacity_scale "${INIT_REPAIR_CHILD_OPACITY_SCALE}" \
  --init_repair_energy_conserve_mode "${INIT_REPAIR_ENERGY_CONSERVE_MODE}" \
  --init_repair_filter_scale "${INIT_REPAIR_FILTER_SCALE}" \
  --init_repair_filter_cap_ratio "${INIT_REPAIR_FILTER_CAP_RATIO}" \
  --init_repair_offset_scale "${INIT_REPAIR_OFFSET_SCALE}" \
  --init_repair_bright_max_fraction "${INIT_REPAIR_BRIGHT_MAX_FRACTION}" \
  --init_repair_bright_max_count "${INIT_REPAIR_BRIGHT_MAX_COUNT}" \
  --init_repair_bright_target "${INIT_REPAIR_BRIGHT_TARGET}" \
  --init_repair_bright_min_opacity "${INIT_REPAIR_BRIGHT_MIN_OPACITY}" \
  --init_repair_bright_max_effective_scale_ratio "${INIT_REPAIR_BRIGHT_MAX_EFFECTIVE_SCALE_RATIO}" \
  --init_repair_bright_luma_quantile "${INIT_REPAIR_BRIGHT_LUMA_QUANTILE}" \
  --init_repair_bright_min_local_luma_ratio "${INIT_REPAIR_BRIGHT_MIN_LOCAL_LUMA_RATIO}" \
  --init_repair_bright_min_color_delta "${INIT_REPAIR_BRIGHT_MIN_COLOR_DELTA}" \
  --init_repair_bright_neighbor_k "${INIT_REPAIR_BRIGHT_NEIGHBOR_K}" \
  --init_repair_bright_expand_scale_multiplier "${INIT_REPAIR_BRIGHT_EXPAND_SCALE_MULTIPLIER}" \
  --init_repair_bright_smallest_axis_scale_multiplier "${INIT_REPAIR_BRIGHT_SMALLEST_AXIS_SCALE_MULTIPLIER}" \
  --init_repair_bright_opacity_scale "${INIT_REPAIR_BRIGHT_OPACITY_SCALE}" \
  --init_repair_bright_dc_scale "${INIT_REPAIR_BRIGHT_DC_SCALE}" \
  --init_repair_bright_rest_scale "${INIT_REPAIR_BRIGHT_REST_SCALE}" \
  --init_repair_bright_filter_scale "${INIT_REPAIR_BRIGHT_FILTER_SCALE}" \
  --init_repair_bright_filter_cap_ratio "${INIT_REPAIR_BRIGHT_FILTER_CAP_RATIO}" \
  --python_bin "${PYTHON_BIN}"
)

if [[ "${USE_AABB_FILTER}" == "1" ]]; then
  CMD+=(--use_aabb_filter)
fi
if [[ "${RUN_RENDER_SANITY}" == "1" ]]; then
  CMD+=(
    --render_sanity
    --render_sanity_images_subdir "${RENDER_SANITY_IMAGES_SUBDIR}"
    --render_sanity_resolution "${RENDER_SANITY_RESOLUTION}"
  )
fi
if [[ "${DEBUG_DUMP_PREPARE_STAGES}" == "1" ]]; then
  CMD+=(--debug_dump_prepare_stages)
fi
if [[ "${DEBUG_DUMP_CANONICALIZE_SUBSTAGES}" == "1" ]]; then
  CMD+=(--debug_dump_canonicalize_substages)
fi

"${CMD[@]}"

echo
echo "[done] input field root : ${OUTPUT_MODEL_PATH}"
echo "[done] trainer can use  : GS_MODEL_PATH=${OUTPUT_MODEL_PATH}"

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

PREPARE_DEBUG_MODEL_PATH="${PREPARE_DEBUG_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k_sof_native_input_init_early4ksoft_v1_debug}"
PREPARE_DEBUG_ITERATION="${PREPARE_DEBUG_ITERATION:-34000}"
STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PREPARE_DEBUG_MODEL_PATH}/debug_prepare_stages/${STAGE_NAME}}"
BASE_ITERATION="${BASE_ITERATION:-${PREPARE_DEBUG_ITERATION}}"

MASK_REPARAM_RUN_NAME="${MASK_REPARAM_RUN_NAME:-${STAGE_NAME}_geometry_then_highlight_v1}"
MASK_REPARAM_MODEL_PATH="${MASK_REPARAM_MODEL_PATH:-${SOF_ROOT}/output/mask_guided_reparameterization_v0/${SCENE_NAME}/${MASK_REPARAM_RUN_NAME}}"
MASK_REPARAM_ITERATION="${MASK_REPARAM_ITERATION:-${BASE_ITERATION}}"
MASK_SOURCE_ROOT="${MASK_SOURCE_ROOT:-${MASK_REPARAM_MODEL_PATH}/masks}"

SETTLE_RUN_NAME="${SETTLE_RUN_NAME:-${MASK_REPARAM_RUN_NAME}_settle_v1_manual}"
DETAIL_RELEASE_RUN_NAME="${DETAIL_RELEASE_RUN_NAME:-${SETTLE_RUN_NAME}_release_v0}"
SH_STABLE_MODEL_PATH="${SH_STABLE_MODEL_PATH:-${SOF_ROOT}/output/mask_reparam_detail_release_v0/${SCENE_NAME}/${DETAIL_RELEASE_RUN_NAME}/released_model}"
SH_STABLE_ITERATION="${SH_STABLE_ITERATION:--1}"

DEFAULT_SOF_SURFACE_MODEL="${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images8_v1/soflr30k}"
if [[ -z "${SOF_TEACHER_MODEL_PATH+x}" ]]; then
  if [[ -d "${BASE_MODEL_PATH}" ]]; then
    SOF_TEACHER_MODEL_PATH="${BASE_MODEL_PATH}"
    SOF_TEACHER_ITERATION_DEFAULT="${BASE_ITERATION}"
  else
    SOF_TEACHER_MODEL_PATH="${DEFAULT_SOF_SURFACE_MODEL}"
    SOF_TEACHER_ITERATION_DEFAULT="30000"
  fi
fi
SOF_TEACHER_ITERATION="${SOF_TEACHER_ITERATION:-${SOF_TEACHER_ITERATION_DEFAULT:-${BASE_ITERATION}}}"

SOF_RAMP_PROFILE="${SOF_RAMP_PROFILE:-ramp25_v0}"
case "${SOF_RAMP_PROFILE}" in
  ramp25_v0)
    DEFAULT_ITERATIONS="400"
    DEFAULT_MAX_VIEWS="16"
    DEFAULT_XYZ_LR="4e-6"
    DEFAULT_OPACITY_LR="3e-4"
    DEFAULT_SCALE_LR="8e-5"
    DEFAULT_LAMBDA_RGB_PRESERVE="1.0"
    DEFAULT_LAMBDA_DEPTH="0.07"
    DEFAULT_LAMBDA_NORMAL="0.010"
    DEFAULT_LAMBDA_ALPHA="0.0125"
    DEFAULT_LAMBDA_MIP_CLOSURE_ALPHA="0.04"
    DEFAULT_LAMBDA_MIP_CLOSURE_PREMUL="0.10"
    DEFAULT_LAMBDA_MIP_CLOSURE_DEPTH="0.010"
    DEFAULT_LAMBDA_OPACITY_ANCHOR="0.05"
    DEFAULT_LAMBDA_SCALE_ANCHOR="0.18"
    DEFAULT_LAMBDA_ANCHOR="70.0"
    DEFAULT_MAX_DISPLACEMENT_RATIO="0.0008"
    ;;
  ramp50_v0)
    DEFAULT_ITERATIONS="600"
    DEFAULT_MAX_VIEWS="20"
    DEFAULT_XYZ_LR="5e-6"
    DEFAULT_OPACITY_LR="4e-4"
    DEFAULT_SCALE_LR="1.0e-4"
    DEFAULT_LAMBDA_RGB_PRESERVE="1.0"
    DEFAULT_LAMBDA_DEPTH="0.14"
    DEFAULT_LAMBDA_NORMAL="0.020"
    DEFAULT_LAMBDA_ALPHA="0.025"
    DEFAULT_LAMBDA_MIP_CLOSURE_ALPHA="0.06"
    DEFAULT_LAMBDA_MIP_CLOSURE_PREMUL="0.14"
    DEFAULT_LAMBDA_MIP_CLOSURE_DEPTH="0.012"
    DEFAULT_LAMBDA_OPACITY_ANCHOR="0.04"
    DEFAULT_LAMBDA_SCALE_ANCHOR="0.16"
    DEFAULT_LAMBDA_ANCHOR="60.0"
    DEFAULT_MAX_DISPLACEMENT_RATIO="0.0010"
    ;;
  ramp100_v0)
    DEFAULT_ITERATIONS="800"
    DEFAULT_MAX_VIEWS="24"
    DEFAULT_XYZ_LR="6e-6"
    DEFAULT_OPACITY_LR="5e-4"
    DEFAULT_SCALE_LR="1.2e-4"
    DEFAULT_LAMBDA_RGB_PRESERVE="1.0"
    DEFAULT_LAMBDA_DEPTH="0.28"
    DEFAULT_LAMBDA_NORMAL="0.040"
    DEFAULT_LAMBDA_ALPHA="0.050"
    DEFAULT_LAMBDA_MIP_CLOSURE_ALPHA="0.10"
    DEFAULT_LAMBDA_MIP_CLOSURE_PREMUL="0.20"
    DEFAULT_LAMBDA_MIP_CLOSURE_DEPTH="0.015"
    DEFAULT_LAMBDA_OPACITY_ANCHOR="0.035"
    DEFAULT_LAMBDA_SCALE_ANCHOR="0.14"
    DEFAULT_LAMBDA_ANCHOR="50.0"
    DEFAULT_MAX_DISPLACEMENT_RATIO="0.0012"
    ;;
  *)
    echo "[mask-reparam-sof-ramp-v0] unknown SOF_RAMP_PROFILE=${SOF_RAMP_PROFILE}" >&2
    exit 1
    ;;
esac

SURFACE_IMAGES_SUBDIR="${SURFACE_IMAGES_SUBDIR:-images_8}"
RENDER_IMAGES_SUBDIR="${RENDER_IMAGES_SUBDIR:-images_2}"
RENDER_SPLIT="${RENDER_SPLIT:-test}"
ITERATIONS="${ITERATIONS:-${DEFAULT_ITERATIONS}}"
MAX_VIEWS="${MAX_VIEWS:-${DEFAULT_MAX_VIEWS}}"
XYZ_LR="${XYZ_LR:-${DEFAULT_XYZ_LR}}"
OPACITY_LR="${OPACITY_LR:-${DEFAULT_OPACITY_LR}}"
SCALE_LR="${SCALE_LR:-${DEFAULT_SCALE_LR}}"
LAMBDA_RGB_PRESERVE="${LAMBDA_RGB_PRESERVE:-${DEFAULT_LAMBDA_RGB_PRESERVE}}"
LAMBDA_DEPTH="${LAMBDA_DEPTH:-${DEFAULT_LAMBDA_DEPTH}}"
LAMBDA_NORMAL="${LAMBDA_NORMAL:-${DEFAULT_LAMBDA_NORMAL}}"
LAMBDA_ALPHA="${LAMBDA_ALPHA:-${DEFAULT_LAMBDA_ALPHA}}"
LAMBDA_MIP_CLOSURE_ALPHA="${LAMBDA_MIP_CLOSURE_ALPHA:-${DEFAULT_LAMBDA_MIP_CLOSURE_ALPHA}}"
LAMBDA_MIP_CLOSURE_PREMUL="${LAMBDA_MIP_CLOSURE_PREMUL:-${DEFAULT_LAMBDA_MIP_CLOSURE_PREMUL}}"
LAMBDA_MIP_CLOSURE_DEPTH="${LAMBDA_MIP_CLOSURE_DEPTH:-${DEFAULT_LAMBDA_MIP_CLOSURE_DEPTH}}"
LAMBDA_OPACITY_ANCHOR="${LAMBDA_OPACITY_ANCHOR:-${DEFAULT_LAMBDA_OPACITY_ANCHOR}}"
LAMBDA_SCALE_ANCHOR="${LAMBDA_SCALE_ANCHOR:-${DEFAULT_LAMBDA_SCALE_ANCHOR}}"
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-${DEFAULT_LAMBDA_ANCHOR}}"
MAX_DISPLACEMENT_RATIO="${MAX_DISPLACEMENT_RATIO:-${DEFAULT_MAX_DISPLACEMENT_RATIO}}"

MIN_SURFACE_ALPHA="${MIN_SURFACE_ALPHA:-0.08}"
MIN_LOSS_PIXELS="${MIN_LOSS_PIXELS:-256}"
MIP_CLOSURE_KERNEL="${MIP_CLOSURE_KERNEL:-25}"
MIP_CLOSURE_ALPHA_THRESHOLD="${MIP_CLOSURE_ALPHA_THRESHOLD:-0.05}"
MIP_CLOSURE_REFERENCE_LOWPASS="${MIP_CLOSURE_REFERENCE_LOWPASS:-1}"
DEPTH_RELATIVE_MIN="${DEPTH_RELATIVE_MIN:-0.5}"
CHARBONNIER_EPS="${CHARBONNIER_EPS:-1e-3}"
ENABLE_OPACITY_UPDATE="${ENABLE_OPACITY_UPDATE:-1}"
ENABLE_SCALE_UPDATE="${ENABLE_SCALE_UPDATE:-1}"

USE_GAUSSIAN_UPDATE_MASK="${USE_GAUSSIAN_UPDATE_MASK:-1}"
OPTIMIZE_GAUSSIAN_MASK_KEY="${OPTIMIZE_GAUSSIAN_MASK_KEY:-selected_mask}"
GAUSSIAN_UPDATE_SCALE="${GAUSSIAN_UPDATE_SCALE:-1.0}"
GAUSSIAN_SCALE_AXIS_MODE="${GAUSSIAN_SCALE_AXIS_MODE:-all}"
GAUSSIAN_SCALE_MIN_MULTIPLIER="${GAUSSIAN_SCALE_MIN_MULTIPLIER:-0.0}"
GAUSSIAN_SCALE_MAX_MULTIPLIER="${GAUSSIAN_SCALE_MAX_MULTIPLIER:-0.0}"

OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${DETAIL_RELEASE_RUN_NAME}_${SOF_RAMP_PROFILE}}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/mask_reparam_sof_ramp_v0/${SCENE_NAME}/${OUTPUT_RUN_NAME}}"
OUTPUT_MODEL="${OUTPUT_MODEL:-${RUN_ROOT}/sof_ramped_model}"
RENDER_DIR="${RENDER_DIR:-${RUN_ROOT}/preview_renders_no_gt_v0}"
UPDATE_MASK_PAYLOAD_PATH="${UPDATE_MASK_PAYLOAD_PATH:-${RUN_ROOT}/sof_ramp_update_mask_v0.pt}"
UPDATE_MASK_INPUT_GEOMETRY="${UPDATE_MASK_INPUT_GEOMETRY:-${MASK_SOURCE_ROOT}/geometry_selected_output_mask.pt}"
UPDATE_MASK_INPUT_HIGHLIGHT="${UPDATE_MASK_INPUT_HIGHLIGHT:-${MASK_SOURCE_ROOT}/highlight_selected_output_mask.pt}"
UPDATE_MASK_INPUT_CHILDREN="${UPDATE_MASK_INPUT_CHILDREN:-${MASK_SOURCE_ROOT}/is_child_output_mask.pt}"

RUN_RENDER="${RUN_RENDER:-1}"
RENDER_PREVIEW="${RENDER_PREVIEW:-1}"
RENDER_PREVIEW_MAX_IMAGES="${RENDER_PREVIEW_MAX_IMAGES:-16}"
RENDER_PREVIEW_COLUMNS="${RENDER_PREVIEW_COLUMNS:-4}"
RENDER_PREVIEW_THUMB_WIDTH="${RENDER_PREVIEW_THUMB_WIDTH:-360}"
SAVE_EVERY="${SAVE_EVERY:-200}"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_model_iteration() {
  local model_root="$1"
  local requested_iter="$2"
  if [[ ! "${requested_iter}" =~ ^- ]]; then
    printf '%s\n' "${requested_iter}"
    return 0
  fi
  local latest_dir
  latest_dir="$(
    find "${model_root}/point_cloud" -mindepth 1 -maxdepth 1 -type d -name 'iteration_*' \
      | sort -V \
      | tail -n 1
  )"
  if [[ -z "${latest_dir}" ]]; then
    return 1
  fi
  basename "${latest_dir}" | sed 's/^iteration_//'
}

if [[ ! -d "${SH_STABLE_MODEL_PATH}" ]]; then
  echo "[mask-reparam-sof-ramp-v0] SH-stable model path not found: ${SH_STABLE_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${SOF_TEACHER_MODEL_PATH}" ]]; then
  echo "[mask-reparam-sof-ramp-v0] SOF teacher model path not found: ${SOF_TEACHER_MODEL_PATH}" >&2
  exit 1
fi

SH_STABLE_EFFECTIVE_ITERATION="$(resolve_model_iteration "${SH_STABLE_MODEL_PATH}" "${SH_STABLE_ITERATION}")"
SOF_TEACHER_EFFECTIVE_ITERATION="$(resolve_model_iteration "${SOF_TEACHER_MODEL_PATH}" "${SOF_TEACHER_ITERATION}")"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-$((SH_STABLE_EFFECTIVE_ITERATION + ITERATIONS))}"
RENDER_PREVIEW_PATH="${RENDER_PREVIEW_PATH:-${RENDER_DIR}/contact_sheet_${RENDER_IMAGES_SUBDIR}_${RENDER_SPLIT}_${OUTPUT_ITERATION}.png}"

mkdir -p "${RUN_ROOT}"

OPTIMIZE_GAUSSIAN_MASK_PAYLOAD=""
if [[ "${USE_GAUSSIAN_UPDATE_MASK}" == "1" ]]; then
  for required_mask in \
    "${UPDATE_MASK_INPUT_GEOMETRY}" \
    "${UPDATE_MASK_INPUT_HIGHLIGHT}" \
    "${UPDATE_MASK_INPUT_CHILDREN}"; do
    if [[ ! -f "${required_mask}" ]]; then
      echo "[mask-reparam-sof-ramp-v0] update mask input not found: ${required_mask}" >&2
      exit 1
    fi
  done
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/build_union_gaussian_mask_payload_v0.py" \
    --input "${UPDATE_MASK_INPUT_GEOMETRY}" \
    --input "${UPDATE_MASK_INPUT_HIGHLIGHT}" \
    --input "${UPDATE_MASK_INPUT_CHILDREN}" \
    --output_path "${UPDATE_MASK_PAYLOAD_PATH}"
  OPTIMIZE_GAUSSIAN_MASK_PAYLOAD="${UPDATE_MASK_PAYLOAD_PATH}"
fi

echo "[mask-reparam-sof-ramp-v0] scene          : ${SCENE_ROOT}"
echo "[mask-reparam-sof-ramp-v0] sh-stable      : ${SH_STABLE_MODEL_PATH} iter=${SH_STABLE_EFFECTIVE_ITERATION}"
echo "[mask-reparam-sof-ramp-v0] sof teacher    : ${SOF_TEACHER_MODEL_PATH} iter=${SOF_TEACHER_EFFECTIVE_ITERATION}"
echo "[mask-reparam-sof-ramp-v0] output model   : ${OUTPUT_MODEL}"
echo "[mask-reparam-sof-ramp-v0] ramp profile   : ${SOF_RAMP_PROFILE}"
echo "[mask-reparam-sof-ramp-v0] train          : iter=${ITERATIONS} xyz_lr=${XYZ_LR} op_lr=${OPACITY_LR} scale_lr=${SCALE_LR}"
echo "[mask-reparam-sof-ramp-v0] surface losses : rgb=${LAMBDA_RGB_PRESERVE} depth=${LAMBDA_DEPTH} normal=${LAMBDA_NORMAL} alpha=${LAMBDA_ALPHA}"
echo "[mask-reparam-sof-ramp-v0] closure        : ${LAMBDA_MIP_CLOSURE_ALPHA}/${LAMBDA_MIP_CLOSURE_PREMUL}/${LAMBDA_MIP_CLOSURE_DEPTH}"
echo "[mask-reparam-sof-ramp-v0] sr prior       : disabled"
echo "[mask-reparam-sof-ramp-v0] update mask    : ${OPTIMIZE_GAUSSIAN_MASK_PAYLOAD:-disabled}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/train_mip_to_sof_surface_v0.py"
  --scene_root "${SCENE_ROOT}"
  --mip_model_path "${SH_STABLE_MODEL_PATH}"
  --sof_surface_model_path "${SOF_TEACHER_MODEL_PATH}"
  --output_model_path "${OUTPUT_MODEL}"
  --images_subdir "${SURFACE_IMAGES_SUBDIR}"
  --mip_iteration "${SH_STABLE_EFFECTIVE_ITERATION}"
  --sof_iteration "${SOF_TEACHER_EFFECTIVE_ITERATION}"
  --output_iteration "${OUTPUT_ITERATION}"
  --max_views "${MAX_VIEWS}"
  --iterations "${ITERATIONS}"
  --xyz_lr "${XYZ_LR}"
  --opacity_lr "${OPACITY_LR}"
  --scale_lr "${SCALE_LR}"
  --lambda_rgb_preserve "${LAMBDA_RGB_PRESERVE}"
  --lambda_depth "${LAMBDA_DEPTH}"
  --lambda_normal "${LAMBDA_NORMAL}"
  --lambda_alpha "${LAMBDA_ALPHA}"
  --lambda_mip_closure_alpha "${LAMBDA_MIP_CLOSURE_ALPHA}"
  --lambda_mip_closure_premul "${LAMBDA_MIP_CLOSURE_PREMUL}"
  --lambda_mip_closure_depth "${LAMBDA_MIP_CLOSURE_DEPTH}"
  --lambda_mip_obs_closure_alpha "0.0"
  --lambda_mip_obs_closure_premul "0.0"
  --lambda_mip_obs_closure_depth "0.0"
  --lambda_opacity_anchor "${LAMBDA_OPACITY_ANCHOR}"
  --lambda_scale_anchor "${LAMBDA_SCALE_ANCHOR}"
  --lambda_anchor "${LAMBDA_ANCHOR}"
  --min_surface_alpha "${MIN_SURFACE_ALPHA}"
  --min_loss_pixels "${MIN_LOSS_PIXELS}"
  --mip_closure_kernel "${MIP_CLOSURE_KERNEL}"
  --mip_closure_alpha_threshold "${MIP_CLOSURE_ALPHA_THRESHOLD}"
  --mip_closure_reference_lowpass "${MIP_CLOSURE_REFERENCE_LOWPASS}"
  --depth_relative_min "${DEPTH_RELATIVE_MIN}"
  --charbonnier_eps "${CHARBONNIER_EPS}"
  --max_displacement_ratio "${MAX_DISPLACEMENT_RATIO}"
  --enable_opacity_update "${ENABLE_OPACITY_UPDATE}"
  --enable_scale_update "${ENABLE_SCALE_UPDATE}"
  --gaussian_update_scale "${GAUSSIAN_UPDATE_SCALE}"
  --gaussian_scale_axis_mode "${GAUSSIAN_SCALE_AXIS_MODE}"
  --gaussian_scale_min_multiplier "${GAUSSIAN_SCALE_MIN_MULTIPLIER}"
  --gaussian_scale_max_multiplier "${GAUSSIAN_SCALE_MAX_MULTIPLIER}"
  --save_every "${SAVE_EVERY}"
)

if [[ -n "${OPTIMIZE_GAUSSIAN_MASK_PAYLOAD}" ]]; then
  CMD+=(
    --optimize_gaussian_mask_payload "${OPTIMIZE_GAUSSIAN_MASK_PAYLOAD}"
    --optimize_gaussian_mask_key "${OPTIMIZE_GAUSSIAN_MASK_KEY}"
  )
fi

"${CMD[@]}"

if [[ "${RUN_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${OUTPUT_MODEL}" \
    --output_dir "${RENDER_DIR}" \
    --images_subdir "${RENDER_IMAGES_SUBDIR}" \
    --iteration "${OUTPUT_ITERATION}" \
    --split "${RENDER_SPLIT}" \
    --max_views "${MAX_VIEWS}"

  if [[ "${RENDER_PREVIEW}" == "1" ]]; then
    "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/make_render_contact_sheet.py" \
      --render_dir "${RENDER_DIR}/${RENDER_SPLIT}/ours_${OUTPUT_ITERATION}/renders" \
      --output_path "${RENDER_PREVIEW_PATH}" \
      --max_images "${RENDER_PREVIEW_MAX_IMAGES}" \
      --columns "${RENDER_PREVIEW_COLUMNS}" \
      --thumb_width "${RENDER_PREVIEW_THUMB_WIDTH}"
  fi
fi

echo "[done] output model : ${OUTPUT_MODEL}"
echo "[done] output ply   : ${OUTPUT_MODEL}/point_cloud/iteration_${OUTPUT_ITERATION}/point_cloud.ply"
echo "[done] summary      : ${OUTPUT_MODEL}/mip_to_sof_surface_v0_summary.json"
if [[ -n "${OPTIMIZE_GAUSSIAN_MASK_PAYLOAD}" ]]; then
  echo "[done] update mask  : ${OPTIMIZE_GAUSSIAN_MASK_PAYLOAD}"
fi
if [[ "${RUN_RENDER}" == "1" ]]; then
  echo "[done] renders      : ${RENDER_DIR}/${RENDER_SPLIT}/ours_${OUTPUT_ITERATION}/renders"
  if [[ "${RENDER_PREVIEW}" == "1" ]]; then
    echo "[done] preview      : ${RENDER_PREVIEW_PATH}"
  fi
fi

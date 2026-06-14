#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="${SOF_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MIP_MODEL_PATH="${MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k}"
MIP_ITERATION="${MIP_ITERATION:-30000}"

DEPTH_PRIOR_ROOT="${DEPTH_PRIOR_ROOT:-${SCENE_ROOT}/depthpriors}"

RUN_NAME="${RUN_NAME:-mip_depthprior_reinit_sof_v0_images8_30k}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/mip_depthprior_reinit_sof_v0/${SCENE_NAME}/${RUN_NAME}}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-244}"
ITERATIONS="${ITERATIONS:-30000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
SH_DEGREE="${SH_DEGREE:-3}"
MIP_RENDER_ROOT="${MIP_RENDER_ROOT:-${MIP_MODEL_PATH}/${SPLIT}/ours_${MIP_ITERATION}/renders}"
NAMED_MIP_RENDER_DIR="${NAMED_MIP_RENDER_DIR:-${OUTPUT_MODEL_PATH}/named_mip_rgb_supervision_${SPLIT}}"
SUPERVISION_MAX_VIEWS="${SUPERVISION_MAX_VIEWS:-0}"
TRAIN_DATA_DEVICE="${TRAIN_DATA_DEVICE:-cpu}"
SPLATTING_CONFIG_PATH="${SPLATTING_CONFIG_PATH:-${MIP_MODEL_PATH}/config.json}"
AUTO_PREPARE_MIP_TRAIN_RENDERS="${AUTO_PREPARE_MIP_TRAIN_RENDERS:-1}"
MIP_RENDER_DATA_DEVICE="${MIP_RENDER_DATA_DEVICE:-cpu}"
KERNEL_SIZE="${KERNEL_SIZE:-0.1}"
RAY_JITTER="${RAY_JITTER:-0}"
RESAMPLE_GT_IMAGE="${RESAMPLE_GT_IMAGE:-0}"
SAMPLE_MORE_HIGHRES="${SAMPLE_MORE_HIGHRES:-0}"

INIT_MAX_VIEWS="${INIT_MAX_VIEWS:-48}"
INIT_PIXEL_STRIDE="${INIT_PIXEL_STRIDE:-8}"
INIT_MIN_WEIGHT="${INIT_MIN_WEIGHT:-0.02}"
INIT_VOXEL_SIZE="${INIT_VOXEL_SIZE:-0.0}"
INIT_VOXEL_SIZE_FACTOR="${INIT_VOXEL_SIZE_FACTOR:-0.002}"
INIT_MIN_POINTS_PER_VOXEL="${INIT_MIN_POINTS_PER_VOXEL:-1}"
INIT_MAX_POINTS="${INIT_MAX_POINTS:-200000}"
INIT_SFM_FALLBACK="${INIT_SFM_FALLBACK:-1}"
INIT_SFM_MAX_POINTS="${INIT_SFM_MAX_POINTS:-120000}"
INIT_SFM_WEIGHT="${INIT_SFM_WEIGHT:-0.20}"
INIT_SFM_ONLY_MISSING="${INIT_SFM_ONLY_MISSING:-0}"
INIT_SFM_MIN_VISIBLE_VIEWS="${INIT_SFM_MIN_VISIBLE_VIEWS:-1}"

XYZ_LR_INIT="${XYZ_LR_INIT:-0.00016}"
XYZ_LR_FINAL="${XYZ_LR_FINAL:-0.0000016}"
XYZ_LR_DELAY_MULT="${XYZ_LR_DELAY_MULT:-0.01}"
FEATURE_LR="${FEATURE_LR:-0.0025}"
FEATURE_REST_LR="${FEATURE_REST_LR:-0.000125}"
OPACITY_LR="${OPACITY_LR:-0.05}"
SCALING_LR="${SCALING_LR:-0.005}"
ROTATION_LR="${ROTATION_LR:-0.001}"

LAMBDA_RGB="${LAMBDA_RGB:-1.0}"
LAMBDA_TEACHER_DEPTH="${LAMBDA_TEACHER_DEPTH:-0.0}"
LAMBDA_TEACHER_NORMAL="${LAMBDA_TEACHER_NORMAL:-0.0}"
LAMBDA_TEACHER_ALPHA="${LAMBDA_TEACHER_ALPHA:-0.0}"
LAMBDA_OPACITY_REG="${LAMBDA_OPACITY_REG:-1e-4}"
MIN_SURFACE_ALPHA="${MIN_SURFACE_ALPHA:-0.05}"
DEPTH_MIN="${DEPTH_MIN:-0.02}"
DEPTH_RELATIVE_MIN="${DEPTH_RELATIVE_MIN:-1e-3}"
CHARBONNIER_EPS="${CHARBONNIER_EPS:-1e-3}"
MIN_LOSS_PIXELS="${MIN_LOSS_PIXELS:-64}"

DEPTH_PRIOR_SUBDIRS="${DEPTH_PRIOR_SUBDIRS:-depth,}"
DEPTH_PRIOR_CONFIDENCE_SUBDIRS="${DEPTH_PRIOR_CONFIDENCE_SUBDIRS:-auto}"
DEPTH_PRIOR_CONFIDENCE_MIN="${DEPTH_PRIOR_CONFIDENCE_MIN:-0.05}"
DEPTH_PRIOR_AGREEMENT_THRESHOLD="${DEPTH_PRIOR_AGREEMENT_THRESHOLD:-0.15}"
DEPTH_PRIOR_AGREEMENT_FLOOR="${DEPTH_PRIOR_AGREEMENT_FLOOR:-0.0}"
DEPTH_PRIOR_ALIGN_MODE="${DEPTH_PRIOR_ALIGN_MODE:-identity}"
DEPTH_PRIOR_ALIGN_MIN_PIXELS="${DEPTH_PRIOR_ALIGN_MIN_PIXELS:-2048}"
DEPTH_PRIOR_SURFACE_WEIGHT_BOOST="${DEPTH_PRIOR_SURFACE_WEIGHT_BOOST:-0.25}"
DEPTH_PRIOR_WEIGHT_GAIN="${DEPTH_PRIOR_WEIGHT_GAIN:-1.0}"
DEPTH_PRIOR_WEIGHT_POWER="${DEPTH_PRIOR_WEIGHT_POWER:-1.0}"
DEPTH_PRIOR_WEIGHT_MIN="${DEPTH_PRIOR_WEIGHT_MIN:-0.0}"
LAMBDA_DEPTH_PRIOR="${LAMBDA_DEPTH_PRIOR:-0.10}"
LAMBDA_DEPTH_PRIOR_NORMAL="${LAMBDA_DEPTH_PRIOR_NORMAL:-0.03}"
LAMBDA_DEPTH_PRIOR_DISTORTION="${LAMBDA_DEPTH_PRIOR_DISTORTION:-100.0}"
LAMBDA_DEPTH_PRIOR_SELF_NORMAL="${LAMBDA_DEPTH_PRIOR_SELF_NORMAL:-0.05}"
DEPTH_PRIOR_WARMUP_START_ITER="${DEPTH_PRIOR_WARMUP_START_ITER:-0}"
DEPTH_PRIOR_WARMUP_END_ITER="${DEPTH_PRIOR_WARMUP_END_ITER:-4000}"
DEPTH_PRIOR_START_SCALE="${DEPTH_PRIOR_START_SCALE:-1.0}"
DEPTH_PRIOR_END_SCALE="${DEPTH_PRIOR_END_SCALE:-2.0}"
DEPTH_PRIOR_UPDATE_SCALE="${DEPTH_PRIOR_UPDATE_SCALE:-1.0}"
DEPTH_PRIOR_SCHEDULE_MODE="${DEPTH_PRIOR_SCHEDULE_MODE:-smoothstep}"

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[mip-depthprior-reinit] missing scene root: ${SCENE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${MIP_MODEL_PATH}" ]]; then
  echo "[mip-depthprior-reinit] missing mip model path: ${MIP_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${MIP_RENDER_ROOT}" ]]; then
  if [[ "${AUTO_PREPARE_MIP_TRAIN_RENDERS}" == "1" && "${SPLIT}" == "train" ]]; then
    echo "[mip-depthprior-reinit] preparing missing train renders at: ${MIP_RENDER_ROOT}"
    "${PYTHON_BIN}" -u "${SOF_ROOT}/render.py" \
      -m "${MIP_MODEL_PATH}" \
      -s "${SCENE_ROOT}" \
      -i "${IMAGES_SUBDIR}" \
      --iteration "${MIP_ITERATION}" \
      --init_type sfm \
      --eval \
      --data_device "${MIP_RENDER_DATA_DEVICE}" \
      --skip_test
  fi
fi
if [[ ! -d "${MIP_RENDER_ROOT}" ]]; then
  echo "[mip-depthprior-reinit] missing mip render root: ${MIP_RENDER_ROOT}" >&2
  exit 1
fi
if [[ -n "${DEPTH_PRIOR_ROOT}" && ! -d "${DEPTH_PRIOR_ROOT}" ]]; then
  echo "[mip-depthprior-reinit] missing depth prior root: ${DEPTH_PRIOR_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_MODEL_PATH}"

echo "[mip-depthprior-reinit] scene root      : ${SCENE_ROOT}"
echo "[mip-depthprior-reinit] mip model       : ${MIP_MODEL_PATH} iter=${MIP_ITERATION}"
echo "[mip-depthprior-reinit] mip render root : ${MIP_RENDER_ROOT}"
echo "[mip-depthprior-reinit] depth prior     : ${DEPTH_PRIOR_ROOT}"
echo "[mip-depthprior-reinit] output          : ${OUTPUT_MODEL_PATH}"
echo "[mip-depthprior-reinit] train           : steps=${ITERATIONS} views=${MAX_VIEWS} split=${SPLIT}"
echo "[mip-depthprior-reinit] init            : views=${INIT_MAX_VIEWS} stride=${INIT_PIXEL_STRIDE} max_points=${INIT_MAX_POINTS}"
echo "[mip-depthprior-reinit] sfm append      : enabled=${INIT_SFM_FALLBACK} only_missing=${INIT_SFM_ONLY_MISSING} max_points=${INIT_SFM_MAX_POINTS} weight=${INIT_SFM_WEIGHT}"
echo "[mip-depthprior-reinit] named renders   : ${NAMED_MIP_RENDER_DIR}"

INIT_CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/train_mip_depthprior_reinit_sof_v0.py"
  --scene_root "${SCENE_ROOT}"
  --mip_model_path "${MIP_MODEL_PATH}"
  --mip_render_root "${MIP_RENDER_ROOT}"
  --mip_iteration "${MIP_ITERATION}"
  --output_model_path "${OUTPUT_MODEL_PATH}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
  --iterations "${ITERATIONS}"
  --save_every "${SAVE_EVERY}"
  --sh_degree "${SH_DEGREE}"
  --init_max_views "${INIT_MAX_VIEWS}"
  --init_pixel_stride "${INIT_PIXEL_STRIDE}"
  --init_min_weight "${INIT_MIN_WEIGHT}"
  --init_voxel_size "${INIT_VOXEL_SIZE}"
  --init_voxel_size_factor "${INIT_VOXEL_SIZE_FACTOR}"
  --init_min_points_per_voxel "${INIT_MIN_POINTS_PER_VOXEL}"
  --init_max_points "${INIT_MAX_POINTS}"
  --init_sfm_max_points "${INIT_SFM_MAX_POINTS}"
  --init_sfm_weight "${INIT_SFM_WEIGHT}"
  --init_sfm_min_visible_views "${INIT_SFM_MIN_VISIBLE_VIEWS}"
  --xyz_lr_init "${XYZ_LR_INIT}"
  --xyz_lr_final "${XYZ_LR_FINAL}"
  --xyz_lr_delay_mult "${XYZ_LR_DELAY_MULT}"
  --feature_lr "${FEATURE_LR}"
  --feature_rest_lr "${FEATURE_REST_LR}"
  --opacity_lr "${OPACITY_LR}"
  --scaling_lr "${SCALING_LR}"
  --rotation_lr "${ROTATION_LR}"
  --lambda_rgb "${LAMBDA_RGB}"
  --lambda_teacher_depth "${LAMBDA_TEACHER_DEPTH}"
  --lambda_teacher_normal "${LAMBDA_TEACHER_NORMAL}"
  --lambda_teacher_alpha "${LAMBDA_TEACHER_ALPHA}"
  --lambda_opacity_reg "${LAMBDA_OPACITY_REG}"
  --min_surface_alpha "${MIN_SURFACE_ALPHA}"
  --depth_min "${DEPTH_MIN}"
  --depth_relative_min "${DEPTH_RELATIVE_MIN}"
  --charbonnier_eps "${CHARBONNIER_EPS}"
  --min_loss_pixels "${MIN_LOSS_PIXELS}"
  --depth_prior_root "${DEPTH_PRIOR_ROOT}"
  --depth_prior_subdirs "${DEPTH_PRIOR_SUBDIRS}"
  --depth_prior_confidence_subdirs "${DEPTH_PRIOR_CONFIDENCE_SUBDIRS}"
  --depth_prior_confidence_min "${DEPTH_PRIOR_CONFIDENCE_MIN}"
  --depth_prior_agreement_threshold "${DEPTH_PRIOR_AGREEMENT_THRESHOLD}"
  --depth_prior_agreement_floor "${DEPTH_PRIOR_AGREEMENT_FLOOR}"
  --depth_prior_align_mode "${DEPTH_PRIOR_ALIGN_MODE}"
  --depth_prior_align_min_pixels "${DEPTH_PRIOR_ALIGN_MIN_PIXELS}"
  --depth_prior_surface_weight_boost "${DEPTH_PRIOR_SURFACE_WEIGHT_BOOST}"
  --depth_prior_weight_gain "${DEPTH_PRIOR_WEIGHT_GAIN}"
  --depth_prior_weight_power "${DEPTH_PRIOR_WEIGHT_POWER}"
  --depth_prior_weight_min "${DEPTH_PRIOR_WEIGHT_MIN}"
  --lambda_depth_prior "${LAMBDA_DEPTH_PRIOR}"
  --lambda_depth_prior_normal "${LAMBDA_DEPTH_PRIOR_NORMAL}"
  --lambda_depth_prior_distortion "${LAMBDA_DEPTH_PRIOR_DISTORTION}"
  --lambda_depth_prior_self_normal "${LAMBDA_DEPTH_PRIOR_SELF_NORMAL}"
  --depth_prior_warmup_start_iter "${DEPTH_PRIOR_WARMUP_START_ITER}"
  --depth_prior_warmup_end_iter "${DEPTH_PRIOR_WARMUP_END_ITER}"
  --depth_prior_start_scale "${DEPTH_PRIOR_START_SCALE}"
  --depth_prior_end_scale "${DEPTH_PRIOR_END_SCALE}"
  --depth_prior_update_scale "${DEPTH_PRIOR_UPDATE_SCALE}"
  --depth_prior_schedule_mode "${DEPTH_PRIOR_SCHEDULE_MODE}"
  --init_only
)
if [[ "${INIT_SFM_FALLBACK}" == "1" ]]; then
  INIT_CMD+=(--init_sfm_fallback)
fi
if [[ "${INIT_SFM_ONLY_MISSING}" == "1" ]]; then
  INIT_CMD+=(--init_sfm_only_missing)
fi

echo
echo "[1/3] depthprior -> init ply"
"${INIT_CMD[@]}"

echo
echo "[2/3] prepare camera-name mip supervision dir"
"${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/prepare_named_mip_render_supervision_v0.py" \
  --scene_root "${SCENE_ROOT}" \
  --model_path "${OUTPUT_MODEL_PATH}" \
  --images_subdir "${IMAGES_SUBDIR}" \
  --split "${SPLIT}" \
  --max_views "${SUPERVISION_MAX_VIEWS}" \
  --render_root "${MIP_RENDER_ROOT}" \
  --output_dir "${NAMED_MIP_RENDER_DIR}"

echo
echo "[3/3] vanilla mip train from depthprior init"
TRAIN_CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/train_vanilla_mip_from_init_v0.py"
  -s "${SCENE_ROOT}"
  -i "${IMAGES_SUBDIR}"
  -m "${OUTPUT_MODEL_PATH}"
  --eval
  --data_device "${TRAIN_DATA_DEVICE}"
  --kernel_size "${KERNEL_SIZE}"
  --vanilla_mip_mode
  --start_ply "${OUTPUT_MODEL_PATH}/point_cloud/iteration_0/point_cloud.ply"
  --global_image_dir "${NAMED_MIP_RENDER_DIR}"
  --iterations "${ITERATIONS}"
  --test_iterations "${ITERATIONS}"
  --save_iterations "${ITERATIONS}"
  --checkpoint_iterations "${ITERATIONS}"
)
if [[ "${RAY_JITTER}" == "1" ]]; then
  TRAIN_CMD+=(--ray_jitter)
fi
if [[ "${RESAMPLE_GT_IMAGE}" == "1" ]]; then
  TRAIN_CMD+=(--resample_gt_image)
fi
if [[ "${SAMPLE_MORE_HIGHRES}" == "1" ]]; then
  TRAIN_CMD+=(--sample_more_highres)
fi
if [[ -f "${SPLATTING_CONFIG_PATH}" ]]; then
  TRAIN_CMD+=(--splatting_config "${SPLATTING_CONFIG_PATH}")
fi
"${TRAIN_CMD[@]}"

echo "[done] init summary : ${OUTPUT_MODEL_PATH}/mip_depthprior_reinit_sof_v0_summary.json"
echo "[done] init ply     : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_0/point_cloud.ply"
echo "[done] output       : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"

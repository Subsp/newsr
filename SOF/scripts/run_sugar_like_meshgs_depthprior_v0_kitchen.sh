#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="${SOF_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
MESH_COMPARE_ROOT="${MESH_COMPARE_ROOT:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}}"
MESH_PATH="${MESH_PATH:-${MESH_COMPARE_ROOT}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"

CAMERA_MODEL_NAME="${CAMERA_MODEL_NAME:-soflr30k}"
CAMERA_MODEL_PATH="${CAMERA_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images8_v1/${CAMERA_MODEL_NAME}}"
COPY_RENDER_CONFIG_FROM="${COPY_RENDER_CONFIG_FROM:-${CAMERA_MODEL_PATH}}"

PATCH_OBSERVATION_RUN_NAME="${PATCH_OBSERVATION_RUN_NAME:-mesh_patch_observations_smoke_v0}"
PATCH_OBSERVATION_ROOT="${PATCH_OBSERVATION_ROOT:-${SOF_ROOT}/output/mesh_patch_observations_v0/${SCENE_NAME}/${PATCH_OBSERVATION_RUN_NAME}}"

PRIOR_DIR="${PRIOR_DIR:-${SCENE_ROOT}/images_8}"
if [[ -z "${ANCHOR_DIR+x}" ]]; then
  ANCHOR_DIR=""
fi
DEPTH_PRIOR_ROOT="${DEPTH_PRIOR_ROOT:-${SCENE_ROOT}/depthpriors}"

RUN_NAME="${RUN_NAME:-sugar_like_meshgs_depthprior_v0_images8_30k}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/sugar_like_meshgs_depthprior_v0/${SCENE_NAME}/${RUN_NAME}}"

IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
SPLIT="${SPLIT:-train}"
MAX_VIEWS="${MAX_VIEWS:-0}"
ITERATIONS="${ITERATIONS:-30000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
SH_DEGREE="${SH_DEGREE:-3}"

MIN_VIEWS="${MIN_VIEWS:-2}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.02}"
MAX_DISAGREEMENT="${MAX_DISAGREEMENT:-0.20}"
MAX_COUNT="${MAX_COUNT:-0}"
SEED="${SEED:-0}"
DEPTH_MIN="${DEPTH_MIN:-0.02}"
ANCHOR_LOWFREQ_THRESHOLD="${ANCHOR_LOWFREQ_THRESHOLD:-0.0}"
ANCHOR_LOWFREQ_KERNEL="${ANCHOR_LOWFREQ_KERNEL:-15}"

INIT_COLOR_SOURCE="${INIT_COLOR_SOURCE:-fused_rgb}"
INIT_COLOR_GRAY_VALUE="${INIT_COLOR_GRAY_VALUE:-0.5}"
MESHGS_SCALE_MULTIPLIER="${MESHGS_SCALE_MULTIPLIER:-1.0}"
MESHGS_THICKNESS_MULTIPLIER="${MESHGS_THICKNESS_MULTIPLIER:-0.5}"
MESHGS_INIT_OPACITY="${MESHGS_INIT_OPACITY:-0.35}"
MESHGS_FEATURE_LR="${MESHGS_FEATURE_LR:-0.01}"
MESHGS_OPACITY_LR="${MESHGS_OPACITY_LR:-0.02}"
MESHGS_LAMBDA_OPACITY="${MESHGS_LAMBDA_OPACITY:-1e-4}"
MESHGS_MIN_PIXELS="${MESHGS_MIN_PIXELS:-64}"
MESHGS_RENDER_ALPHA_THRESHOLD="${MESHGS_RENDER_ALPHA_THRESHOLD:-1e-4}"
MESH_FUSION_MASK_DIR="${MESH_FUSION_MASK_DIR:-}"

LEARN_SURFACE_VERTICES="${LEARN_SURFACE_VERTICES:-1}"
LEARN_PLANE_SCALES="${LEARN_PLANE_SCALES:-0}"
LEARN_INPLANE_ROTATION="${LEARN_INPLANE_ROTATION:-0}"
SURFACE_VERTEX_LR="${SURFACE_VERTEX_LR:-5e-4}"
PLANE_SCALE_LR="${PLANE_SCALE_LR:-0.0}"
INPLANE_ROTATION_LR="${INPLANE_ROTATION_LR:-0.0}"
LAMBDA_SURFACE_DELTA="${LAMBDA_SURFACE_DELTA:-0.02}"
MAX_SURFACE_VERTEX_DISPLACEMENT="${MAX_SURFACE_VERTEX_DISPLACEMENT:-0.02}"
NORMAL_CONSISTENCY_LAMBDA="${NORMAL_CONSISTENCY_LAMBDA:-0.02}"
DISABLE_NORMAL_CONSISTENCY_PAIRS="${DISABLE_NORMAL_CONSISTENCY_PAIRS:-0}"
MAX_NORMAL_CONSISTENCY_PAIRS="${MAX_NORMAL_CONSISTENCY_PAIRS:-500000}"

DEPTH_RELATIVE_MIN="${DEPTH_RELATIVE_MIN:-1e-3}"
CHARBONNIER_EPS="${CHARBONNIER_EPS:-1e-3}"
MIN_LOSS_PIXELS="${MIN_LOSS_PIXELS:-64}"

DEPTH_PRIOR_SUBDIRS="${DEPTH_PRIOR_SUBDIRS:-depth,}"
DEPTH_PRIOR_CONFIDENCE_SUBDIRS="${DEPTH_PRIOR_CONFIDENCE_SUBDIRS:-auto}"
DEPTH_PRIOR_CONFIDENCE_MIN="${DEPTH_PRIOR_CONFIDENCE_MIN:-0.05}"
DEPTH_PRIOR_AGREEMENT_THRESHOLD="${DEPTH_PRIOR_AGREEMENT_THRESHOLD:-0.15}"
DEPTH_PRIOR_AGREEMENT_FLOOR="${DEPTH_PRIOR_AGREEMENT_FLOOR:-0.0}"
DEPTH_PRIOR_ALIGN_MODE="${DEPTH_PRIOR_ALIGN_MODE:-affine_robust}"
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
  echo "[sugar-like-depthprior] missing scene root: ${SCENE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CAMERA_MODEL_PATH}" ]]; then
  echo "[sugar-like-depthprior] missing camera model path: ${CAMERA_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[sugar-like-depthprior] missing mesh: ${MESH_PATH}" >&2
  exit 1
fi
if [[ ! -d "${PATCH_OBSERVATION_ROOT}" ]]; then
  echo "[sugar-like-depthprior] missing patch observation root: ${PATCH_OBSERVATION_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${PRIOR_DIR}" ]]; then
  echo "[sugar-like-depthprior] missing prior dir: ${PRIOR_DIR}" >&2
  exit 1
fi
if [[ -n "${ANCHOR_DIR}" && ! -d "${ANCHOR_DIR}" ]]; then
  echo "[sugar-like-depthprior] missing anchor dir: ${ANCHOR_DIR}" >&2
  exit 1
fi
if [[ -n "${DEPTH_PRIOR_ROOT}" && ! -d "${DEPTH_PRIOR_ROOT}" ]]; then
  echo "[sugar-like-depthprior] missing depth prior root: ${DEPTH_PRIOR_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_MODEL_PATH}"

echo "[sugar-like-depthprior] scene root      : ${SCENE_ROOT}"
echo "[sugar-like-depthprior] camera model    : ${CAMERA_MODEL_PATH}"
echo "[sugar-like-depthprior] mesh            : ${MESH_PATH}"
echo "[sugar-like-depthprior] patch root      : ${PATCH_OBSERVATION_ROOT}"
echo "[sugar-like-depthprior] prior dir       : ${PRIOR_DIR}"
if [[ -n "${ANCHOR_DIR}" ]]; then
  echo "[sugar-like-depthprior] anchor dir      : ${ANCHOR_DIR}"
else
  echo "[sugar-like-depthprior] anchor dir      : disabled"
fi
echo "[sugar-like-depthprior] depth prior     : ${DEPTH_PRIOR_ROOT}"
echo "[sugar-like-depthprior] output          : ${OUTPUT_MODEL_PATH}"
echo "[sugar-like-depthprior] train           : steps=${ITERATIONS} views=${MAX_VIEWS} split=${SPLIT}"
echo "[sugar-like-depthprior] depth cfg       : l1=${LAMBDA_DEPTH_PRIOR} normal=${LAMBDA_DEPTH_PRIOR_NORMAL} distort=${LAMBDA_DEPTH_PRIOR_DISTORTION} selfn=${LAMBDA_DEPTH_PRIOR_SELF_NORMAL}"

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/train_sugar_like_meshgs_depthprior_v0.py"
  --scene_root "${SCENE_ROOT}"
  --camera_model_path "${CAMERA_MODEL_PATH}"
  --mesh_path "${MESH_PATH}"
  --patch_observation_root "${PATCH_OBSERVATION_ROOT}"
  --prior_dir "${PRIOR_DIR}"
  --output_model_path "${OUTPUT_MODEL_PATH}"
  --copy_render_config_from "${COPY_RENDER_CONFIG_FROM}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --max_views "${MAX_VIEWS}"
  --iterations "${ITERATIONS}"
  --save_every "${SAVE_EVERY}"
  --sh_degree "${SH_DEGREE}"
  --min_views "${MIN_VIEWS}"
  --min_confidence "${MIN_CONFIDENCE}"
  --max_disagreement "${MAX_DISAGREEMENT}"
  --max_count "${MAX_COUNT}"
  --seed "${SEED}"
  --depth_min "${DEPTH_MIN}"
  --anchor_lowfreq_threshold "${ANCHOR_LOWFREQ_THRESHOLD}"
  --anchor_lowfreq_kernel "${ANCHOR_LOWFREQ_KERNEL}"
  --init_color_source "${INIT_COLOR_SOURCE}"
  --init_color_gray_value "${INIT_COLOR_GRAY_VALUE}"
  --meshgs_scale_multiplier "${MESHGS_SCALE_MULTIPLIER}"
  --meshgs_thickness_multiplier "${MESHGS_THICKNESS_MULTIPLIER}"
  --meshgs_init_opacity "${MESHGS_INIT_OPACITY}"
  --meshgs_feature_lr "${MESHGS_FEATURE_LR}"
  --meshgs_opacity_lr "${MESHGS_OPACITY_LR}"
  --meshgs_lambda_opacity "${MESHGS_LAMBDA_OPACITY}"
  --meshgs_min_pixels "${MESHGS_MIN_PIXELS}"
  --meshgs_render_alpha_threshold "${MESHGS_RENDER_ALPHA_THRESHOLD}"
  --surface_vertex_lr "${SURFACE_VERTEX_LR}"
  --plane_scale_lr "${PLANE_SCALE_LR}"
  --inplane_rotation_lr "${INPLANE_ROTATION_LR}"
  --lambda_surface_delta "${LAMBDA_SURFACE_DELTA}"
  --max_surface_vertex_displacement "${MAX_SURFACE_VERTEX_DISPLACEMENT}"
  --normal_consistency_lambda "${NORMAL_CONSISTENCY_LAMBDA}"
  --max_normal_consistency_pairs "${MAX_NORMAL_CONSISTENCY_PAIRS}"
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
)

if [[ -n "${ANCHOR_DIR}" ]]; then
  CMD+=(--anchor_dir "${ANCHOR_DIR}")
fi
if [[ -n "${MESH_FUSION_MASK_DIR}" ]]; then
  CMD+=(--mesh_fusion_mask_dir "${MESH_FUSION_MASK_DIR}")
fi
if [[ "${LEARN_SURFACE_VERTICES}" == "1" ]]; then
  CMD+=(--learn_surface_vertices)
fi
if [[ "${LEARN_PLANE_SCALES}" == "1" ]]; then
  CMD+=(--learn_plane_scales)
fi
if [[ "${LEARN_INPLANE_ROTATION}" == "1" ]]; then
  CMD+=(--learn_inplane_rotation)
fi
if [[ "${DISABLE_NORMAL_CONSISTENCY_PAIRS}" == "1" ]]; then
  CMD+=(--disable_normal_consistency_pairs)
fi

"${CMD[@]}"

echo "[done] summary : ${OUTPUT_MODEL_PATH}/sugar_like_meshgs_depthprior_v0_summary.json"
echo "[done] output  : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"

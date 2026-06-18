#!/usr/bin/env bash
set -euo pipefail

# Canonical NoSR cleanup mainline preserved for the ~28.57 PSNR setting.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MIPSPLATTING_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
TRAIN_SCENE_ROOT="${TRAIN_SCENE_ROOT:-${SCENE_ROOT}}"
EVAL_SCENE_ROOT="${EVAL_SCENE_ROOT:-${SCENE_ROOT}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${DEFAULT_MIPSPLATTING_ROOT}}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
INPUT_EXPERIMENT_NAME="${INPUT_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
INPUT_MODEL_DIR="${INPUT_MODEL_DIR:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${INPUT_EXPERIMENT_NAME}}"
INPUT_ITERATION="${INPUT_ITERATION:-30000}"
CLEANUP_ITERS="${CLEANUP_ITERS:-2000}"
FINAL_ITER="${FINAL_ITER:-$(( INPUT_ITERATION + CLEANUP_ITERS ))}"
START_CHECKPOINT="${START_CHECKPOINT:-${INPUT_MODEL_DIR}/chkpnt${INPUT_ITERATION}.pth}"
INPUT_POINT_CLOUD_PLY="${INPUT_POINT_CLOUD_PLY:-${INPUT_MODEL_DIR}/point_cloud/iteration_${INPUT_ITERATION}/point_cloud.ply}"

GS2MESH_MESH_NAME="${GS2MESH_MESH_NAME:-kitchen_MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_15_cleaned_mesh.ply}"
MESH_PATH="${MESH_PATH:-${WORK_ROOT}/${GS2MESH_MESH_NAME}}"

TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR:-images_2}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
# The original 28.57 NoSR cleanup used images_2 with -r4, i.e. an LR-sized
# supervision grid, then rendered/evaluated images_2 at -r1. Training at -r1
# turns this into a much stronger HR-GT finetune and is not the same test.
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-4}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-1}"

INPUT_BASENAME="$(basename "${INPUT_MODEL_DIR}")"
RUN_TAG="${RUN_TAG:-${INPUT_BASENAME}_to${FINAL_ITER}_trainr${TRAIN_RESOLUTION}_nosr28_layerfreq_cleanup_v0}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/mipsplatting_nosr_layerfreq_cleanup_v0/${SCENE_NAME}}"
MODEL_DIR="${MODEL_DIR:-${RUN_ROOT}/${RUN_TAG}}"

SURFACE_STATE_PROFILE="${SURFACE_STATE_PROFILE:-relaxed_carrier_v1}"
SURFACE_STATE_RUN_TAG="${SURFACE_STATE_RUN_TAG:-${RUN_TAG}_surface_state_${SURFACE_STATE_PROFILE}}"
SURFACE_STATE_DIR="${SURFACE_STATE_DIR:-${OUTPUT_ROOT}/gaussian_surface_state_v0/${SCENE_NAME}/${SURFACE_STATE_RUN_TAG}}"
SURFACE_STATE_PAYLOAD="${SURFACE_STATE_PAYLOAD:-${SURFACE_STATE_DIR}/gaussian_surface_state_v0.pt}"
FORCE_REBUILD_SURFACE_STATE="${FORCE_REBUILD_SURFACE_STATE:-0}"

TRAIN_PORT="${TRAIN_PORT:-6009}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_RENDER_AFTER="${RUN_RENDER_AFTER:-1}"
RUN_METRICS_AFTER="${RUN_METRICS_AFTER:-1}"
RUN_COMPARE_AFTER="${RUN_COMPARE_AFTER:-1}"

# Defaults mirror the high-performing 2026-06-06 fixed-topology NoSR cleanup:
# non-surface layers are discouraged from carrying Laplacian HF, while the
# full-render HF residual is routed only to the surface_carrier mask.
LAYER_FREQUENCY_NON_SURFACE_KEY="${LAYER_FREQUENCY_NON_SURFACE_KEY:-non_surface_active}"
LAYER_FREQUENCY_SURFACE_KEY="${LAYER_FREQUENCY_SURFACE_KEY:-surface_carrier}"
LAYER_FREQUENCY_SURFACE_TARGET="${LAYER_FREQUENCY_SURFACE_TARGET:-gt}"
LAMBDA_NON_SURFACE_HF="${LAMBDA_NON_SURFACE_HF:-0.03}"
LAMBDA_NON_SURFACE_RGB_ENERGY="${LAMBDA_NON_SURFACE_RGB_ENERGY:-0.0}"
LAMBDA_NON_SURFACE_ALPHA_HF="${LAMBDA_NON_SURFACE_ALPHA_HF:-0.0}"
LAMBDA_NON_SURFACE_ALPHA_MASS="${LAMBDA_NON_SURFACE_ALPHA_MASS:-0.0}"
LAMBDA_SURFACE_HF_CLOSURE="${LAMBDA_SURFACE_HF_CLOSURE:-0.02}"
LAMBDA_SURFACE_START_HF_PRESERVE="${LAMBDA_SURFACE_START_HF_PRESERVE:-0.0}"
LAYER_FREQUENCY_START_HF_CHECKPOINT="${LAYER_FREQUENCY_START_HF_CHECKPOINT:-${START_CHECKPOINT}}"
LAYER_FREQUENCY_START_HF_LOWFREQ_KERNEL="${LAYER_FREQUENCY_START_HF_LOWFREQ_KERNEL:-15}"
LAYER_FREQUENCY_START_HF_LOWFREQ_THRESHOLD="${LAYER_FREQUENCY_START_HF_LOWFREQ_THRESHOLD:-0.05}"
LAYER_FREQUENCY_START_HF_ENERGY_THRESHOLD="${LAYER_FREQUENCY_START_HF_ENERGY_THRESHOLD:-0.01}"
LAYER_FREQUENCY_START_HF_MASK_POWER="${LAYER_FREQUENCY_START_HF_MASK_POWER:-1.0}"
LAYER_FREQUENCY_START_HF_PROTECT_NON_SURFACE="${LAYER_FREQUENCY_START_HF_PROTECT_NON_SURFACE:-0}"
SURFACE_HF_UPDATE_SCALE="${SURFACE_HF_UPDATE_SCALE:-0.5}"
LAYER_FREQUENCY_FROM_ITER="${LAYER_FREQUENCY_FROM_ITER:-${INPUT_ITERATION}}"
LAYER_FREQUENCY_UNTIL_ITER="${LAYER_FREQUENCY_UNTIL_ITER:-${FINAL_ITER}}"
LAYER_FREQUENCY_LOG_INTERVAL="${LAYER_FREQUENCY_LOG_INTERVAL:-100}"
LAYER_FREQUENCY_DYNAMIC_ROOTS="${LAYER_FREQUENCY_DYNAMIC_ROOTS:-0}"

EXTERNAL_PRIOR_ROOT="${EXTERNAL_PRIOR_ROOT:-}"
EXTERNAL_PRIOR_SUBDIR="${EXTERNAL_PRIOR_SUBDIR:-priors}"
EXTERNAL_PRIOR_MASK_SUBDIR="${EXTERNAL_PRIOR_MASK_SUBDIR:-}"
EXTERNAL_PRIOR_EXTS="${EXTERNAL_PRIOR_EXTS:-png,jpg,jpeg,webp}"
PRIOR_CONSISTENCY_THRESHOLD="${PRIOR_CONSISTENCY_THRESHOLD:-0.20}"
PRIOR_MIN_VALID_RATIO="${PRIOR_MIN_VALID_RATIO:-0.30}"
PRIOR_LOSS_MODE="${PRIOR_LOSS_MODE:-rgb_hf}"
PRIOR_L1_WEIGHT="${PRIOR_L1_WEIGHT:-0.05}"
PRIOR_HF_WEIGHT="${PRIOR_HF_WEIGHT:-0.1}"
PRIOR_DELTA_CLIP="${PRIOR_DELTA_CLIP:-0.15}"

PRIOR_LOCAL_DIR="${PRIOR_LOCAL_DIR:-}"
PRIOR_LOCAL_MASK_DIR="${PRIOR_LOCAL_MASK_DIR:-}"
LAMBDA_PRIOR_LOCAL="${LAMBDA_PRIOR_LOCAL:-0.0}"
PRIOR_LOCAL_FROM_ITER="${PRIOR_LOCAL_FROM_ITER:-${INPUT_ITERATION}}"
PRIOR_LOCAL_MIN_PIXELS="${PRIOR_LOCAL_MIN_PIXELS:-64}"
LAMBDA_PRIOR_LOCAL_SURFACE="${LAMBDA_PRIOR_LOCAL_SURFACE:-0.0}"
PRIOR_LOCAL_SURFACE_PAYLOAD="${PRIOR_LOCAL_SURFACE_PAYLOAD:-${SURFACE_STATE_PAYLOAD}}"
PRIOR_LOCAL_SURFACE_FROM_ITER="${PRIOR_LOCAL_SURFACE_FROM_ITER:-${INPUT_ITERATION}}"
PRIOR_LOCAL_SURFACE_INTERVAL="${PRIOR_LOCAL_SURFACE_INTERVAL:-1}"
PRIOR_LOCAL_SURFACE_MIN_GAUSSIANS="${PRIOR_LOCAL_SURFACE_MIN_GAUSSIANS:-128}"
PRIOR_LOCAL_SURFACE_MAX_SAMPLES="${PRIOR_LOCAL_SURFACE_MAX_SAMPLES:-2048}"
PRIOR_LOCAL_SURFACE_KNN="${PRIOR_LOCAL_SURFACE_KNN:-4}"
PRIOR_LOCAL_SURFACE_NORMAL_MIN_COS="${PRIOR_LOCAL_SURFACE_NORMAL_MIN_COS:-0.85}"
PRIOR_LOCAL_SURFACE_RADIUS_SCALE="${PRIOR_LOCAL_SURFACE_RADIUS_SCALE:-2.5}"
PRIOR_LOCAL_SURFACE_COLOR_WEIGHT="${PRIOR_LOCAL_SURFACE_COLOR_WEIGHT:-1.0}"
PRIOR_LOCAL_SURFACE_OPACITY_WEIGHT="${PRIOR_LOCAL_SURFACE_OPACITY_WEIGHT:-0.05}"
PRIOR_LOCAL_SURFACE_SCALE_WEIGHT="${PRIOR_LOCAL_SURFACE_SCALE_WEIGHT:-0.02}"
PRIOR_LOCAL_SURFACE_THIN_WEIGHT="${PRIOR_LOCAL_SURFACE_THIN_WEIGHT:-0.05}"
PRIOR_LOCAL_SURFACE_THIN_RATIO="${PRIOR_LOCAL_SURFACE_THIN_RATIO:-0.35}"
PRIOR_LOCAL_SURFACE_RESIDUAL_WEIGHT="${PRIOR_LOCAL_SURFACE_RESIDUAL_WEIGHT:-0.0}"
PRIOR_LOCAL_SURFACE_RESIDUAL_CLIP="${PRIOR_LOCAL_SURFACE_RESIDUAL_CLIP:-0.03}"
PRIOR_LOCAL_SURFACE_RESIDUAL_CONFIDENCE_POWER="${PRIOR_LOCAL_SURFACE_RESIDUAL_CONFIDENCE_POWER:-1.0}"
PRIOR_LOCAL_SURFACE_RESIDUAL_MIN_WEIGHT="${PRIOR_LOCAL_SURFACE_RESIDUAL_MIN_WEIGHT:-0.05}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_WEIGHT="${PRIOR_LOCAL_SURFACE_RAY_PATCH_WEIGHT:-0.0}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_TARGET="${PRIOR_LOCAL_SURFACE_RAY_PATCH_TARGET:-gt}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_MIN_SEEDS="${PRIOR_LOCAL_SURFACE_RAY_PATCH_MIN_SEEDS:-64}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_MAX_SEEDS="${PRIOR_LOCAL_SURFACE_RAY_PATCH_MAX_SEEDS:-512}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_MAX_CANDIDATES="${PRIOR_LOCAL_SURFACE_RAY_PATCH_MAX_CANDIDATES:-16384}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_KNN="${PRIOR_LOCAL_SURFACE_RAY_PATCH_KNN:-8}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_TARGET_BLEND="${PRIOR_LOCAL_SURFACE_RAY_PATCH_TARGET_BLEND:-0.25}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_TARGET_DELTA_CLIP="${PRIOR_LOCAL_SURFACE_RAY_PATCH_TARGET_DELTA_CLIP:-0.08}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_CONFIDENCE_POWER="${PRIOR_LOCAL_SURFACE_RAY_PATCH_CONFIDENCE_POWER:-1.0}"
PRIOR_LOCAL_SURFACE_RAY_PATCH_MIN_WEIGHT="${PRIOR_LOCAL_SURFACE_RAY_PATCH_MIN_WEIGHT:-0.05}"
PRIOR_LOCAL_SURFACE_TOUCH_MIN_RADIUS_PX="${PRIOR_LOCAL_SURFACE_TOUCH_MIN_RADIUS_PX:-1.0}"
PRIOR_LOCAL_SURFACE_TOUCH_RADIUS_SCALE="${PRIOR_LOCAL_SURFACE_TOUCH_RADIUS_SCALE:-0.35}"
PRIOR_LOCAL_SURFACE_TOUCH_MAX_RADIUS_PX="${PRIOR_LOCAL_SURFACE_TOUCH_MAX_RADIUS_PX:-8.0}"
PRIOR_LOCAL_SURFACE_MASK_THRESHOLD="${PRIOR_LOCAL_SURFACE_MASK_THRESHOLD:-0.20}"

PRIOR_EDGE_DIR="${PRIOR_EDGE_DIR:-}"
PRIOR_EDGE_MASK_DIR="${PRIOR_EDGE_MASK_DIR:-}"
PRIOR_EDGE_ANCHOR_DIR="${PRIOR_EDGE_ANCHOR_DIR:-}"
LAMBDA_PRIOR_EDGE="${LAMBDA_PRIOR_EDGE:-0.0}"
PRIOR_EDGE_LOSS_MODE="${PRIOR_EDGE_LOSS_MODE:-detail_v1}"
PRIOR_EDGE_FROM_ITER="${PRIOR_EDGE_FROM_ITER:-${INPUT_ITERATION}}"
PRIOR_EDGE_MIN_PIXELS="${PRIOR_EDGE_MIN_PIXELS:-32}"
PRIOR_EDGE_BLEND_ALPHA="${PRIOR_EDGE_BLEND_ALPHA:-1.0}"
PRIOR_EDGE_DETAIL_BLUR_KERNEL="${PRIOR_EDGE_DETAIL_BLUR_KERNEL:-9}"
PRIOR_EDGE_DETAIL_ALPHA="${PRIOR_EDGE_DETAIL_ALPHA:-0.45}"
PRIOR_EDGE_DETAIL_ALPHA_FINAL="${PRIOR_EDGE_DETAIL_ALPHA_FINAL:--1.0}"
PRIOR_EDGE_DETAIL_WARMUP_ITERS="${PRIOR_EDGE_DETAIL_WARMUP_ITERS:-0}"
PRIOR_EDGE_DETAIL_WEIGHT="${PRIOR_EDGE_DETAIL_WEIGHT:-1.0}"
PRIOR_EDGE_LOWFREQ_WEIGHT="${PRIOR_EDGE_LOWFREQ_WEIGHT:-0.0}"
PRIOR_EDGE_GRAD_WEIGHT="${PRIOR_EDGE_GRAD_WEIGHT:-0.0}"
PRIOR_EDGE_LOWFREQ_THRESHOLD="${PRIOR_EDGE_LOWFREQ_THRESHOLD:-0.08}"
PRIOR_EDGE_LOWFREQ_ANCHOR="${PRIOR_EDGE_LOWFREQ_ANCHOR:-render}"
PRIOR_EDGE_DETAIL_MIN_GAIN="${PRIOR_EDGE_DETAIL_MIN_GAIN:-0.0}"
PRIOR_EDGE_CONFIDENCE_POWER="${PRIOR_EDGE_CONFIDENCE_POWER:-1.5}"
PRIOR_EDGE_HF_RESIDUAL_CLIP="${PRIOR_EDGE_HF_RESIDUAL_CLIP:-0.0}"
PRIOR_EDGE_UPDATE_SCALE="${PRIOR_EDGE_UPDATE_SCALE:-0.75}"
PRIOR_EDGE_CONTRAST_WEIGHT="${PRIOR_EDGE_CONTRAST_WEIGHT:-0.0}"
PRIOR_EDGE_CONTRAST_RADIUS="${PRIOR_EDGE_CONTRAST_RADIUS:-1}"
PRIOR_EDGE_CONTRAST_TARGET_GAIN="${PRIOR_EDGE_CONTRAST_TARGET_GAIN:-1.0}"
PRIOR_EDGE_CONTRAST_TARGET_CLIP="${PRIOR_EDGE_CONTRAST_TARGET_CLIP:-0.0}"
LAMBDA_PRIOR_EDGE_SHAPE="${LAMBDA_PRIOR_EDGE_SHAPE:-0.0}"
PRIOR_EDGE_SHAPE_MIN_GAUSSIANS="${PRIOR_EDGE_SHAPE_MIN_GAUSSIANS:-32}"
PRIOR_EDGE_SHAPE_THIN_RATIO="${PRIOR_EDGE_SHAPE_THIN_RATIO:-0.35}"
PRIOR_EDGE_SHAPE_LINE_RATIO="${PRIOR_EDGE_SHAPE_LINE_RATIO:-0.60}"
PRIOR_EDGE_SHAPE_MAX_AXIS="${PRIOR_EDGE_SHAPE_MAX_AXIS:-0.0}"
PRIOR_EDGE_TOUCH_MIN_RADIUS_PX="${PRIOR_EDGE_TOUCH_MIN_RADIUS_PX:-1.0}"
PRIOR_EDGE_TOUCH_RADIUS_SCALE="${PRIOR_EDGE_TOUCH_RADIUS_SCALE:-0.35}"
PRIOR_EDGE_TOUCH_MAX_RADIUS_PX="${PRIOR_EDGE_TOUCH_MAX_RADIUS_PX:-8.0}"
OPTIMIZE_SOURCE_TAG="${OPTIMIZE_SOURCE_TAG:-all}"
PRIOR_ONLY_EDGE_FINETUNE="${PRIOR_ONLY_EDGE_FINETUNE:-0}"

PRIOR_HF_SEED_ENABLE="${PRIOR_HF_SEED_ENABLE:-0}"
PRIOR_HF_SEED_SOURCE="${PRIOR_HF_SEED_SOURCE:-external}"
PRIOR_HF_SEED_FROM_ITER="${PRIOR_HF_SEED_FROM_ITER:-${INPUT_ITERATION}}"
PRIOR_HF_SEED_UNTIL_ITER="${PRIOR_HF_SEED_UNTIL_ITER:-${FINAL_ITER}}"
PRIOR_HF_SEED_INTERVAL="${PRIOR_HF_SEED_INTERVAL:-100}"
PRIOR_HF_SEED_MAX_PER_ITER="${PRIOR_HF_SEED_MAX_PER_ITER:-512}"
PRIOR_HF_SEED_MAX_TOTAL="${PRIOR_HF_SEED_MAX_TOTAL:-8192}"
PRIOR_HF_SEED_SCALE_MULTIPLIER="${PRIOR_HF_SEED_SCALE_MULTIPLIER:-0.35}"
PRIOR_HF_SEED_SCALE_MODE="${PRIOR_HF_SEED_SCALE_MODE:-min_provider_pixel}"
PRIOR_HF_SEED_MIN_PIXEL_RADIUS="${PRIOR_HF_SEED_MIN_PIXEL_RADIUS:-0.25}"
PRIOR_HF_SEED_MAX_PIXEL_RADIUS="${PRIOR_HF_SEED_MAX_PIXEL_RADIUS:-1.50}"
PRIOR_HF_SEED_MAX_PROVIDER_RATIO="${PRIOR_HF_SEED_MAX_PROVIDER_RATIO:-0.75}"
PRIOR_HF_SEED_SHAPE_MODE="${PRIOR_HF_SEED_SHAPE_MODE:-hf_oriented}"
PRIOR_HF_SEED_SHAPE_LONG_RATIO="${PRIOR_HF_SEED_SHAPE_LONG_RATIO:-2.0}"
PRIOR_HF_SEED_SHAPE_SHORT_RATIO="${PRIOR_HF_SEED_SHAPE_SHORT_RATIO:-0.55}"
PRIOR_HF_SEED_SHAPE_NORMAL_RATIO="${PRIOR_HF_SEED_SHAPE_NORMAL_RATIO:-0.30}"
PRIOR_HF_SEED_SHAPE_CONFIDENCE_POWER="${PRIOR_HF_SEED_SHAPE_CONFIDENCE_POWER:-1.0}"
PRIOR_HF_SEED_DIAG_LARGE_SCALE="${PRIOR_HF_SEED_DIAG_LARGE_SCALE:-0.0}"
PRIOR_HF_SEED_SCALE_CLAMP_ENABLE="${PRIOR_HF_SEED_SCALE_CLAMP_ENABLE:-1}"
PRIOR_HF_SEED_SCALE_CLAMP_MAX_AXIS="${PRIOR_HF_SEED_SCALE_CLAMP_MAX_AXIS:-0.003}"
PRIOR_HF_SEED_SCALE_CLAMP_MIN_AXIS="${PRIOR_HF_SEED_SCALE_CLAMP_MIN_AXIS:-0.0}"
PRIOR_HF_SEED_SCALE_CLAMP_MAX_ANISOTROPY="${PRIOR_HF_SEED_SCALE_CLAMP_MAX_ANISOTROPY:-12.0}"
PRIOR_HF_SEED_OPACITY="${PRIOR_HF_SEED_OPACITY:-0.015}"
PRIOR_HF_SEED_JITTER_SCALE="${PRIOR_HF_SEED_JITTER_SCALE:-0.0}"
PRIOR_HF_SEED_COLOR_RESIDUAL_GAIN="${PRIOR_HF_SEED_COLOR_RESIDUAL_GAIN:-0.5}"
PRIOR_HF_SEED_GUIDANCE_THRESHOLD="${PRIOR_HF_SEED_GUIDANCE_THRESHOLD:-0.0}"
PRIOR_HF_SEED_EDGE_HIGHPASS_KERNEL="${PRIOR_HF_SEED_EDGE_HIGHPASS_KERNEL:-0}"
PRIOR_HF_SEED_EDGE_GUIDANCE_POWER="${PRIOR_HF_SEED_EDGE_GUIDANCE_POWER:-1.0}"
PRIOR_HF_SEED_FIRST_CYCLE_ONLY="${PRIOR_HF_SEED_FIRST_CYCLE_ONLY:-1}"
PRIOR_HF_SEED_ORIGINAL_ONLY="${PRIOR_HF_SEED_ORIGINAL_ONLY:-1}"
PRIOR_HF_SEED_PRUNE_PROTECT_ITERS="${PRIOR_HF_SEED_PRUNE_PROTECT_ITERS:-320}"
PRIOR_HF_SEED_RECENT_HF_BOOST="${PRIOR_HF_SEED_RECENT_HF_BOOST:-0.0}"
PRIOR_HF_SEED_RECENT_IMPRINT_WEIGHT="${PRIOR_HF_SEED_RECENT_IMPRINT_WEIGHT:-0.0}"
PRIOR_HF_SEED_RECENT_IMPRINT_GAIN="${PRIOR_HF_SEED_RECENT_IMPRINT_GAIN:-1.0}"
PRIOR_HF_SEED_RECENT_IMPRINT_ITERS="${PRIOR_HF_SEED_RECENT_IMPRINT_ITERS:-0}"
PRIOR_HF_SEED_RAY_FAN_ENABLE="${PRIOR_HF_SEED_RAY_FAN_ENABLE:-0}"
PRIOR_HF_SEED_RAY_FAN_RAYS="${PRIOR_HF_SEED_RAY_FAN_RAYS:-5}"
PRIOR_HF_SEED_RAY_FAN_RADIUS_PX="${PRIOR_HF_SEED_RAY_FAN_RADIUS_PX:-1.25}"
PRIOR_HF_SEED_RAY_FAN_INCLUDE_CENTER="${PRIOR_HF_SEED_RAY_FAN_INCLUDE_CENTER:-1}"
PRIOR_HF_SEED_RAY_FAN_GUIDANCE_MIN_RATIO="${PRIOR_HF_SEED_RAY_FAN_GUIDANCE_MIN_RATIO:-0.35}"
PRIOR_HF_SEED_RAY_FAN_MASK_THRESHOLD="${PRIOR_HF_SEED_RAY_FAN_MASK_THRESHOLD:-0.20}"
PRIOR_HF_LOWFREQ_CLEANUP_ENABLE="${PRIOR_HF_LOWFREQ_CLEANUP_ENABLE:-0}"
PRIOR_HF_LOWFREQ_CLEANUP_FROM_ITER="${PRIOR_HF_LOWFREQ_CLEANUP_FROM_ITER:-$(( INPUT_ITERATION + 360 ))}"
PRIOR_HF_LOWFREQ_CLEANUP_UNTIL_ITER="${PRIOR_HF_LOWFREQ_CLEANUP_UNTIL_ITER:-${FINAL_ITER}}"
PRIOR_HF_LOWFREQ_CLEANUP_INTERVAL="${PRIOR_HF_LOWFREQ_CLEANUP_INTERVAL:-20}"
PRIOR_HF_LOWFREQ_CLEANUP_STALE_ITERS="${PRIOR_HF_LOWFREQ_CLEANUP_STALE_ITERS:-320}"
PRIOR_HF_LOWFREQ_CLEANUP_OPACITY_MIN="${PRIOR_HF_LOWFREQ_CLEANUP_OPACITY_MIN:-0.06}"
PRIOR_HF_LOWFREQ_CLEANUP_SCALE_MAX_MIN="${PRIOR_HF_LOWFREQ_CLEANUP_SCALE_MAX_MIN:-0.0}"
PRIOR_HF_LOWFREQ_CLEANUP_SCALE_MAX_MAX="${PRIOR_HF_LOWFREQ_CLEANUP_SCALE_MAX_MAX:-0.003}"
PRIOR_HF_LOWFREQ_CLEANUP_ANISOTROPY_MAX="${PRIOR_HF_LOWFREQ_CLEANUP_ANISOTROPY_MAX:-12.0}"
PRIOR_HF_LOWFREQ_CLEANUP_OPACITY_DECAY="${PRIOR_HF_LOWFREQ_CLEANUP_OPACITY_DECAY:-0.92}"
PRIOR_HF_LOWFREQ_CLEANUP_PRUNE_OPACITY_BELOW="${PRIOR_HF_LOWFREQ_CLEANUP_PRUNE_OPACITY_BELOW:-0.006}"
PRIOR_HF_LOWFREQ_CLEANUP_MAX_PRUNE_FRACTION="${PRIOR_HF_LOWFREQ_CLEANUP_MAX_PRUNE_FRACTION:-0.02}"
PRIOR_HF_LOWFREQ_CLEANUP_MAX_PRUNE_COUNT="${PRIOR_HF_LOWFREQ_CLEANUP_MAX_PRUNE_COUNT:-8192}"

SURFACE_NORMAL_LOCK="${SURFACE_NORMAL_LOCK:-1}"
SURFACE_NORMAL_LOCK_NORMAL_KEY="${SURFACE_NORMAL_LOCK_NORMAL_KEY:-anchor_normal}"
SURFACE_NORMAL_LOCK_ANCHOR_KEY="${SURFACE_NORMAL_LOCK_ANCHOR_KEY:-anchor_xyz}"
SURFACE_NORMAL_LOCK_DYNAMIC_ROOTS="${SURFACE_NORMAL_LOCK_DYNAMIC_ROOTS:-0}"

DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-999999999}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-0}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-1000000}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
DENSIFY_MIN_OPACITY="${DENSIFY_MIN_OPACITY:-0.005}"
DENSIFY_GLOBAL_PRUNE_ENABLE="${DENSIFY_GLOBAL_PRUNE_ENABLE:-1}"
OPACITY_RESET_INTERVAL="${OPACITY_RESET_INTERVAL:-1000000}"

CHECKPOINT_PATH="${MODEL_DIR}/chkpnt${FINAL_ITER}.pth"
RESULTS_JSON="${MODEL_DIR}/results_psnr_ssim.json"
COMPARE_JSON="${COMPARE_JSON:-${MODEL_DIR}/nosr_cleanup_compare_vs_input.json}"
MIP_PYTHONPATH="${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=1
fi

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

for path in \
  "${SCENE_ROOT}" \
  "${TRAIN_SCENE_ROOT}" \
  "${TRAIN_SCENE_ROOT}/sparse/0" \
  "${TRAIN_SCENE_ROOT}/${TRAIN_IMAGES_SUBDIR}" \
  "${EVAL_SCENE_ROOT}" \
  "${EVAL_SCENE_ROOT}/sparse/0" \
  "${EVAL_SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}" \
  "${MIPSPLATTING_ROOT}" \
  "${START_CHECKPOINT}" \
  "${INPUT_POINT_CLOUD_PLY}" \
  "${MESH_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[nosr-layerfreq-cleanup-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}" "${MODEL_DIR}" "${SURFACE_STATE_DIR}"

echo "[nosr-layerfreq-cleanup-v0] scene             : ${SCENE_ROOT}"
echo "[nosr-layerfreq-cleanup-v0] train scene       : ${TRAIN_SCENE_ROOT}"
echo "[nosr-layerfreq-cleanup-v0] eval scene        : ${EVAL_SCENE_ROOT}"
echo "[nosr-layerfreq-cleanup-v0] input model       : ${INPUT_MODEL_DIR}"
echo "[nosr-layerfreq-cleanup-v0] start checkpoint  : ${START_CHECKPOINT}"
echo "[nosr-layerfreq-cleanup-v0] input ply         : ${INPUT_POINT_CLOUD_PLY}"
echo "[nosr-layerfreq-cleanup-v0] surface mesh      : ${MESH_PATH}"
echo "[nosr-layerfreq-cleanup-v0] surface profile   : ${SURFACE_STATE_PROFILE}"
echo "[nosr-layerfreq-cleanup-v0] surface payload   : ${SURFACE_STATE_PAYLOAD}"
echo "[nosr-layerfreq-cleanup-v0] train target      : ${TRAIN_IMAGES_SUBDIR}"
echo "[nosr-layerfreq-cleanup-v0] output model      : ${MODEL_DIR}"
echo "[nosr-layerfreq-cleanup-v0] iter schedule     : ${INPUT_ITERATION} -> ${FINAL_ITER}"
echo "[nosr-layerfreq-cleanup-v0] layer freq        : ns=${LAMBDA_NON_SURFACE_HF} surf=${LAMBDA_SURFACE_HF_CLOSURE} start_hf=${LAMBDA_SURFACE_START_HF_PRESERVE} scale=${SURFACE_HF_UPDATE_SCALE} target=${LAYER_FREQUENCY_SURFACE_TARGET} dynamic_roots=${LAYER_FREQUENCY_DYNAMIC_ROOTS}"
if [[ -n "${EXTERNAL_PRIOR_ROOT}" ]]; then
  echo "[nosr-layerfreq-cleanup-v0] external prior   : root=${EXTERNAL_PRIOR_ROOT} subdir=${EXTERNAL_PRIOR_SUBDIR} mask=${EXTERNAL_PRIOR_MASK_SUBDIR:-none}"
fi
if [[ -n "${PRIOR_LOCAL_DIR}" || -n "${PRIOR_LOCAL_MASK_DIR}" ]]; then
  if [[ -z "${PRIOR_LOCAL_DIR}" || -z "${PRIOR_LOCAL_MASK_DIR}" ]]; then
    echo "[nosr-layerfreq-cleanup-v0] PRIOR_LOCAL_DIR and PRIOR_LOCAL_MASK_DIR must be set together." >&2
    exit 1
  fi
  for path in "${PRIOR_LOCAL_DIR}" "${PRIOR_LOCAL_MASK_DIR}"; do
    if [[ ! -d "${path}" ]]; then
      echo "[nosr-layerfreq-cleanup-v0] required NPSE local prior dir not found: ${path}" >&2
      exit 1
    fi
  done
  echo "[nosr-layerfreq-cleanup-v0] prior local     : target=${PRIOR_LOCAL_DIR} mask=${PRIOR_LOCAL_MASK_DIR} lambda=${LAMBDA_PRIOR_LOCAL} local3d=${LAMBDA_PRIOR_LOCAL_SURFACE}"
fi
if [[ "${LAMBDA_PRIOR_LOCAL_SURFACE}" != "0" && "${LAMBDA_PRIOR_LOCAL_SURFACE}" != "0.0" ]]; then
  echo "[nosr-layerfreq-cleanup-v0] local surface  : payload=${PRIOR_LOCAL_SURFACE_PAYLOAD} knn=${PRIOR_LOCAL_SURFACE_KNN} max_samples=${PRIOR_LOCAL_SURFACE_MAX_SAMPLES} normal_cos=${PRIOR_LOCAL_SURFACE_NORMAL_MIN_COS} residual_w=${PRIOR_LOCAL_SURFACE_RESIDUAL_WEIGHT} ray_patch_w=${PRIOR_LOCAL_SURFACE_RAY_PATCH_WEIGHT}"
fi
if [[ -n "${PRIOR_EDGE_DIR}" || -n "${PRIOR_EDGE_MASK_DIR}" ]]; then
  if [[ -z "${PRIOR_EDGE_DIR}" || -z "${PRIOR_EDGE_MASK_DIR}" ]]; then
    echo "[nosr-layerfreq-cleanup-v0] PRIOR_EDGE_DIR and PRIOR_EDGE_MASK_DIR must be set together." >&2
    exit 1
  fi
  for path in "${PRIOR_EDGE_DIR}" "${PRIOR_EDGE_MASK_DIR}"; do
    if [[ ! -d "${path}" ]]; then
      echo "[nosr-layerfreq-cleanup-v0] required NPSE edge prior dir not found: ${path}" >&2
      exit 1
    fi
  done
  if [[ -n "${PRIOR_EDGE_ANCHOR_DIR}" && ! -d "${PRIOR_EDGE_ANCHOR_DIR}" ]]; then
    echo "[nosr-layerfreq-cleanup-v0] required NPSE edge anchor dir not found: ${PRIOR_EDGE_ANCHOR_DIR}" >&2
    exit 1
  fi
  echo "[nosr-layerfreq-cleanup-v0] prior edge      : target=${PRIOR_EDGE_DIR} mask=${PRIOR_EDGE_MASK_DIR} lambda=${LAMBDA_PRIOR_EDGE} mode=${PRIOR_EDGE_LOSS_MODE}"
  echo "[nosr-layerfreq-cleanup-v0] edge carrier    : anchor=${PRIOR_EDGE_ANCHOR_DIR:-render-detach} contrast=${PRIOR_EDGE_CONTRAST_WEIGHT} hf_clip=${PRIOR_EDGE_HF_RESIDUAL_CLIP} shape=${LAMBDA_PRIOR_EDGE_SHAPE} thin=${PRIOR_EDGE_SHAPE_THIN_RATIO} line=${PRIOR_EDGE_SHAPE_LINE_RATIO}"
  echo "[nosr-layerfreq-cleanup-v0] edge ownership  : optimize_source=${OPTIMIZE_SOURCE_TAG} prior_only=${PRIOR_ONLY_EDGE_FINETUNE}"
fi
if [[ "${PRIOR_HF_SEED_ENABLE}" == "1" ]]; then
  if [[ "${PRIOR_HF_SEED_SOURCE}" == "external" && -z "${EXTERNAL_PRIOR_ROOT}" ]]; then
    echo "[nosr-layerfreq-cleanup-v0] PRIOR_HF_SEED_ENABLE=1 requires EXTERNAL_PRIOR_ROOT for seed guidance." >&2
    exit 1
  fi
  if [[ "${PRIOR_HF_SEED_SOURCE}" == "edge" && ( -z "${PRIOR_EDGE_DIR}" || -z "${PRIOR_EDGE_MASK_DIR}" ) ]]; then
    echo "[nosr-layerfreq-cleanup-v0] PRIOR_HF_SEED_SOURCE=edge requires PRIOR_EDGE_DIR and PRIOR_EDGE_MASK_DIR." >&2
    exit 1
  fi
  echo "[nosr-layerfreq-cleanup-v0] hf seed          : source=${PRIOR_HF_SEED_SOURCE} max_iter=${PRIOR_HF_SEED_MAX_PER_ITER} max_total=${PRIOR_HF_SEED_MAX_TOTAL} first_cycle=${PRIOR_HF_SEED_FIRST_CYCLE_ONLY} protect=${PRIOR_HF_SEED_PRUNE_PROTECT_ITERS}"
  if [[ "${PRIOR_HF_SEED_RAY_FAN_ENABLE}" == "1" ]]; then
    echo "[nosr-layerfreq-cleanup-v0] hf seed rayfan   : rays=${PRIOR_HF_SEED_RAY_FAN_RAYS} radius_px=${PRIOR_HF_SEED_RAY_FAN_RADIUS_PX} mask_thr=${PRIOR_HF_SEED_RAY_FAN_MASK_THRESHOLD}"
  fi
fi
if [[ "${PRIOR_HF_LOWFREQ_CLEANUP_ENABLE}" == "1" ]]; then
  echo "[nosr-layerfreq-cleanup-v0] hf seed cleanup  : from=${PRIOR_HF_LOWFREQ_CLEANUP_FROM_ITER} stale=${PRIOR_HF_LOWFREQ_CLEANUP_STALE_ITERS} decay=${PRIOR_HF_LOWFREQ_CLEANUP_OPACITY_DECAY} prune_below=${PRIOR_HF_LOWFREQ_CLEANUP_PRUNE_OPACITY_BELOW}"
fi
if [[ "${LAMBDA_SURFACE_START_HF_PRESERVE}" != "0" && "${LAMBDA_SURFACE_START_HF_PRESERVE}" != "0.0" ]]; then
  echo "[nosr-layerfreq-cleanup-v0] start HF preserve: checkpoint=${LAYER_FREQUENCY_START_HF_CHECKPOINT} lf_kernel=${LAYER_FREQUENCY_START_HF_LOWFREQ_KERNEL} lf_thr=${LAYER_FREQUENCY_START_HF_LOWFREQ_THRESHOLD} energy_thr=${LAYER_FREQUENCY_START_HF_ENERGY_THRESHOLD} power=${LAYER_FREQUENCY_START_HF_MASK_POWER} protect_ns=${LAYER_FREQUENCY_START_HF_PROTECT_NON_SURFACE}"
fi
echo "[nosr-layerfreq-cleanup-v0] densify/prune     : from=${DENSIFY_FROM_ITER} until=${DENSIFY_UNTIL_ITER} interval=${DENSIFICATION_INTERVAL} grad=${DENSIFY_GRAD_THRESHOLD} min_opacity=${DENSIFY_MIN_OPACITY} global_prune=${DENSIFY_GLOBAL_PRUNE_ENABLE}"

echo
echo "[1/4] build row-aligned surface-state payload for input checkpoint"
if [[ "${FORCE_REBUILD_SURFACE_STATE}" == "1" || ! -f "${SURFACE_STATE_PAYLOAD}" ]]; then
  (
    cd "${SOF_ROOT}"
    PYTHON_BIN="${PYTHON_BIN}" \
    POINT_CLOUD_PLY="${INPUT_POINT_CLOUD_PLY}" \
    MESH_PATH="${MESH_PATH}" \
    OUTPUT_DIR="${SURFACE_STATE_DIR}" \
    SURFACE_STATE_PROFILE="${SURFACE_STATE_PROFILE}" \
    bash "${SCRIPT_DIR}/run_classify_mip_surface_state_v0_kitchen.sh"
  )
else
  echo "[nosr-layerfreq-cleanup-v0] reuse surface-state payload: ${SURFACE_STATE_PAYLOAD}"
fi

if [[ ! -f "${SURFACE_STATE_PAYLOAD}" ]]; then
  echo "[nosr-layerfreq-cleanup-v0] surface-state payload was not created: ${SURFACE_STATE_PAYLOAD}" >&2
  exit 1
fi

echo
echo "[2/4] run fixed-topology NoSR layer-frequency cleanup"
if [[ "${FORCE_RERUN}" == "1" || ! -f "${CHECKPOINT_PATH}" ]]; then
  TRAIN_ARGS=(
    "${PYTHON_BIN}" -m hybrid_sdfgs.train
    -s "${TRAIN_SCENE_ROOT}"
    -i "${TRAIN_IMAGES_SUBDIR}"
    -m "${MODEL_DIR}"
    -r "${TRAIN_RESOLUTION}"
    --eval
    --disable_gui
    --port "${TRAIN_PORT}"
    --iterations "${FINAL_ITER}"
    --test_iterations "${FINAL_ITER}"
    --save_iterations "${FINAL_ITER}"
    --checkpoint_iterations "${FINAL_ITER}"
    --start_checkpoint "${START_CHECKPOINT}"
    --densify_from_iter "${DENSIFY_FROM_ITER}"
    --densify_until_iter "${DENSIFY_UNTIL_ITER}"
    --densification_interval "${DENSIFICATION_INTERVAL}"
    --densify_grad_threshold "${DENSIFY_GRAD_THRESHOLD}"
    --densify_min_opacity "${DENSIFY_MIN_OPACITY}"
    --densify_global_prune_enable "${DENSIFY_GLOBAL_PRUNE_ENABLE}"
    --opacity_reset_interval "${OPACITY_RESET_INTERVAL}"
    --optimize_source_tag "${OPTIMIZE_SOURCE_TAG}"
    --layer_frequency_mask_payload "${SURFACE_STATE_PAYLOAD}"
    --layer_frequency_non_surface_key "${LAYER_FREQUENCY_NON_SURFACE_KEY}"
    --layer_frequency_surface_key "${LAYER_FREQUENCY_SURFACE_KEY}"
    --layer_frequency_surface_target "${LAYER_FREQUENCY_SURFACE_TARGET}"
    --lambda_non_surface_hf "${LAMBDA_NON_SURFACE_HF}"
    --lambda_non_surface_rgb_energy "${LAMBDA_NON_SURFACE_RGB_ENERGY}"
    --lambda_non_surface_alpha_hf "${LAMBDA_NON_SURFACE_ALPHA_HF}"
    --lambda_non_surface_alpha_mass "${LAMBDA_NON_SURFACE_ALPHA_MASS}"
    --lambda_surface_hf_closure "${LAMBDA_SURFACE_HF_CLOSURE}"
    --lambda_surface_start_hf_preserve "${LAMBDA_SURFACE_START_HF_PRESERVE}"
    --layer_frequency_start_hf_checkpoint "${LAYER_FREQUENCY_START_HF_CHECKPOINT}"
    --layer_frequency_start_hf_lowfreq_kernel "${LAYER_FREQUENCY_START_HF_LOWFREQ_KERNEL}"
    --layer_frequency_start_hf_lowfreq_threshold "${LAYER_FREQUENCY_START_HF_LOWFREQ_THRESHOLD}"
    --layer_frequency_start_hf_energy_threshold "${LAYER_FREQUENCY_START_HF_ENERGY_THRESHOLD}"
    --layer_frequency_start_hf_mask_power "${LAYER_FREQUENCY_START_HF_MASK_POWER}"
    --surface_hf_update_scale "${SURFACE_HF_UPDATE_SCALE}"
    --layer_frequency_from_iter "${LAYER_FREQUENCY_FROM_ITER}"
    --layer_frequency_until_iter "${LAYER_FREQUENCY_UNTIL_ITER}"
    --layer_frequency_log_interval "${LAYER_FREQUENCY_LOG_INTERVAL}"
    --prior_loss_mode "${PRIOR_LOSS_MODE}"
    --prior_l1_weight "${PRIOR_L1_WEIGHT}"
    --prior_hf_weight "${PRIOR_HF_WEIGHT}"
    --prior_delta_clip "${PRIOR_DELTA_CLIP}"
  )
  if [[ -n "${PRIOR_EDGE_DIR}" || -n "${PRIOR_EDGE_MASK_DIR}" ]]; then
    TRAIN_ARGS+=(
      --prior_edge_dir "${PRIOR_EDGE_DIR}"
      --prior_edge_mask_dir "${PRIOR_EDGE_MASK_DIR}"
      --lambda_prior_edge "${LAMBDA_PRIOR_EDGE}"
      --prior_edge_loss_mode "${PRIOR_EDGE_LOSS_MODE}"
      --prior_edge_anchor_dir "${PRIOR_EDGE_ANCHOR_DIR}"
      --prior_edge_blend_alpha "${PRIOR_EDGE_BLEND_ALPHA}"
      --prior_edge_min_pixels "${PRIOR_EDGE_MIN_PIXELS}"
      --prior_edge_from_iter "${PRIOR_EDGE_FROM_ITER}"
      --prior_edge_touch_min_radius_px "${PRIOR_EDGE_TOUCH_MIN_RADIUS_PX}"
      --prior_edge_touch_radius_scale "${PRIOR_EDGE_TOUCH_RADIUS_SCALE}"
      --prior_edge_touch_max_radius_px "${PRIOR_EDGE_TOUCH_MAX_RADIUS_PX}"
      --prior_edge_detail_blur_kernel "${PRIOR_EDGE_DETAIL_BLUR_KERNEL}"
      --prior_edge_detail_alpha "${PRIOR_EDGE_DETAIL_ALPHA}"
      --prior_edge_detail_alpha_final "${PRIOR_EDGE_DETAIL_ALPHA_FINAL}"
      --prior_edge_detail_warmup_iters "${PRIOR_EDGE_DETAIL_WARMUP_ITERS}"
      --prior_edge_detail_weight "${PRIOR_EDGE_DETAIL_WEIGHT}"
      --prior_edge_lowfreq_weight "${PRIOR_EDGE_LOWFREQ_WEIGHT}"
      --prior_edge_grad_weight "${PRIOR_EDGE_GRAD_WEIGHT}"
      --prior_edge_lowfreq_threshold "${PRIOR_EDGE_LOWFREQ_THRESHOLD}"
      --prior_edge_lowfreq_anchor "${PRIOR_EDGE_LOWFREQ_ANCHOR}"
      --prior_edge_detail_min_gain "${PRIOR_EDGE_DETAIL_MIN_GAIN}"
      --prior_edge_confidence_power "${PRIOR_EDGE_CONFIDENCE_POWER}"
      --prior_edge_hf_residual_clip "${PRIOR_EDGE_HF_RESIDUAL_CLIP}"
      --prior_edge_update_scale "${PRIOR_EDGE_UPDATE_SCALE}"
      --prior_edge_contrast_weight "${PRIOR_EDGE_CONTRAST_WEIGHT}"
      --prior_edge_contrast_radius "${PRIOR_EDGE_CONTRAST_RADIUS}"
      --prior_edge_contrast_target_gain "${PRIOR_EDGE_CONTRAST_TARGET_GAIN}"
      --prior_edge_contrast_target_clip "${PRIOR_EDGE_CONTRAST_TARGET_CLIP}"
      --lambda_prior_edge_shape "${LAMBDA_PRIOR_EDGE_SHAPE}"
      --prior_edge_shape_min_gaussians "${PRIOR_EDGE_SHAPE_MIN_GAUSSIANS}"
      --prior_edge_shape_thin_ratio "${PRIOR_EDGE_SHAPE_THIN_RATIO}"
      --prior_edge_shape_line_ratio "${PRIOR_EDGE_SHAPE_LINE_RATIO}"
      --prior_edge_shape_max_axis "${PRIOR_EDGE_SHAPE_MAX_AXIS}"
    )
    if [[ "${PRIOR_ONLY_EDGE_FINETUNE}" == "1" ]]; then
      TRAIN_ARGS+=(--prior_only_edge_finetune)
    fi
  fi
  if [[ -n "${PRIOR_LOCAL_DIR}" || -n "${PRIOR_LOCAL_MASK_DIR}" ]]; then
    TRAIN_ARGS+=(
      --prior_local_dir "${PRIOR_LOCAL_DIR}"
      --prior_local_mask_dir "${PRIOR_LOCAL_MASK_DIR}"
      --lambda_prior_local "${LAMBDA_PRIOR_LOCAL}"
      --prior_local_from_iter "${PRIOR_LOCAL_FROM_ITER}"
      --prior_local_min_pixels "${PRIOR_LOCAL_MIN_PIXELS}"
      --lambda_prior_local_surface "${LAMBDA_PRIOR_LOCAL_SURFACE}"
      --prior_local_surface_payload "${PRIOR_LOCAL_SURFACE_PAYLOAD}"
      --prior_local_surface_from_iter "${PRIOR_LOCAL_SURFACE_FROM_ITER}"
      --prior_local_surface_interval "${PRIOR_LOCAL_SURFACE_INTERVAL}"
      --prior_local_surface_min_gaussians "${PRIOR_LOCAL_SURFACE_MIN_GAUSSIANS}"
      --prior_local_surface_max_samples "${PRIOR_LOCAL_SURFACE_MAX_SAMPLES}"
      --prior_local_surface_knn "${PRIOR_LOCAL_SURFACE_KNN}"
      --prior_local_surface_normal_min_cos "${PRIOR_LOCAL_SURFACE_NORMAL_MIN_COS}"
      --prior_local_surface_radius_scale "${PRIOR_LOCAL_SURFACE_RADIUS_SCALE}"
      --prior_local_surface_color_weight "${PRIOR_LOCAL_SURFACE_COLOR_WEIGHT}"
      --prior_local_surface_opacity_weight "${PRIOR_LOCAL_SURFACE_OPACITY_WEIGHT}"
      --prior_local_surface_scale_weight "${PRIOR_LOCAL_SURFACE_SCALE_WEIGHT}"
      --prior_local_surface_thin_weight "${PRIOR_LOCAL_SURFACE_THIN_WEIGHT}"
      --prior_local_surface_thin_ratio "${PRIOR_LOCAL_SURFACE_THIN_RATIO}"
      --prior_local_surface_residual_weight "${PRIOR_LOCAL_SURFACE_RESIDUAL_WEIGHT}"
      --prior_local_surface_residual_clip "${PRIOR_LOCAL_SURFACE_RESIDUAL_CLIP}"
      --prior_local_surface_residual_confidence_power "${PRIOR_LOCAL_SURFACE_RESIDUAL_CONFIDENCE_POWER}"
      --prior_local_surface_residual_min_weight "${PRIOR_LOCAL_SURFACE_RESIDUAL_MIN_WEIGHT}"
      --prior_local_surface_ray_patch_weight "${PRIOR_LOCAL_SURFACE_RAY_PATCH_WEIGHT}"
      --prior_local_surface_ray_patch_target "${PRIOR_LOCAL_SURFACE_RAY_PATCH_TARGET}"
      --prior_local_surface_ray_patch_min_seeds "${PRIOR_LOCAL_SURFACE_RAY_PATCH_MIN_SEEDS}"
      --prior_local_surface_ray_patch_max_seeds "${PRIOR_LOCAL_SURFACE_RAY_PATCH_MAX_SEEDS}"
      --prior_local_surface_ray_patch_max_candidates "${PRIOR_LOCAL_SURFACE_RAY_PATCH_MAX_CANDIDATES}"
      --prior_local_surface_ray_patch_knn "${PRIOR_LOCAL_SURFACE_RAY_PATCH_KNN}"
      --prior_local_surface_ray_patch_target_blend "${PRIOR_LOCAL_SURFACE_RAY_PATCH_TARGET_BLEND}"
      --prior_local_surface_ray_patch_target_delta_clip "${PRIOR_LOCAL_SURFACE_RAY_PATCH_TARGET_DELTA_CLIP}"
      --prior_local_surface_ray_patch_confidence_power "${PRIOR_LOCAL_SURFACE_RAY_PATCH_CONFIDENCE_POWER}"
      --prior_local_surface_ray_patch_min_weight "${PRIOR_LOCAL_SURFACE_RAY_PATCH_MIN_WEIGHT}"
      --prior_local_surface_touch_min_radius_px "${PRIOR_LOCAL_SURFACE_TOUCH_MIN_RADIUS_PX}"
      --prior_local_surface_touch_radius_scale "${PRIOR_LOCAL_SURFACE_TOUCH_RADIUS_SCALE}"
      --prior_local_surface_touch_max_radius_px "${PRIOR_LOCAL_SURFACE_TOUCH_MAX_RADIUS_PX}"
      --prior_local_surface_mask_threshold "${PRIOR_LOCAL_SURFACE_MASK_THRESHOLD}"
    )
  fi
  if [[ -n "${EXTERNAL_PRIOR_ROOT}" ]]; then
    TRAIN_ARGS+=(
      --external_prior_root "${EXTERNAL_PRIOR_ROOT}"
      --external_prior_subdir "${EXTERNAL_PRIOR_SUBDIR}"
      --external_prior_exts "${EXTERNAL_PRIOR_EXTS}"
      --prior_consistency_threshold "${PRIOR_CONSISTENCY_THRESHOLD}"
      --prior_min_valid_ratio "${PRIOR_MIN_VALID_RATIO}"
    )
    if [[ -n "${EXTERNAL_PRIOR_MASK_SUBDIR}" ]]; then
      TRAIN_ARGS+=(--external_prior_mask_subdir "${EXTERNAL_PRIOR_MASK_SUBDIR}")
    fi
  fi
  if [[ "${PRIOR_HF_SEED_ENABLE}" == "1" ]]; then
    TRAIN_ARGS+=(
      --prior_hf_seed_enable
      --prior_hf_seed_source "${PRIOR_HF_SEED_SOURCE}"
      --prior_hf_seed_from_iter "${PRIOR_HF_SEED_FROM_ITER}"
      --prior_hf_seed_until_iter "${PRIOR_HF_SEED_UNTIL_ITER}"
      --prior_hf_seed_interval "${PRIOR_HF_SEED_INTERVAL}"
      --prior_hf_seed_max_per_iter "${PRIOR_HF_SEED_MAX_PER_ITER}"
      --prior_hf_seed_max_total "${PRIOR_HF_SEED_MAX_TOTAL}"
      --prior_hf_seed_scale_multiplier "${PRIOR_HF_SEED_SCALE_MULTIPLIER}"
      --prior_hf_seed_scale_mode "${PRIOR_HF_SEED_SCALE_MODE}"
      --prior_hf_seed_min_pixel_radius "${PRIOR_HF_SEED_MIN_PIXEL_RADIUS}"
      --prior_hf_seed_max_pixel_radius "${PRIOR_HF_SEED_MAX_PIXEL_RADIUS}"
      --prior_hf_seed_max_provider_ratio "${PRIOR_HF_SEED_MAX_PROVIDER_RATIO}"
      --prior_hf_seed_shape_mode "${PRIOR_HF_SEED_SHAPE_MODE}"
      --prior_hf_seed_shape_long_ratio "${PRIOR_HF_SEED_SHAPE_LONG_RATIO}"
      --prior_hf_seed_shape_short_ratio "${PRIOR_HF_SEED_SHAPE_SHORT_RATIO}"
      --prior_hf_seed_shape_normal_ratio "${PRIOR_HF_SEED_SHAPE_NORMAL_RATIO}"
      --prior_hf_seed_shape_confidence_power "${PRIOR_HF_SEED_SHAPE_CONFIDENCE_POWER}"
      --prior_hf_seed_diag_large_scale "${PRIOR_HF_SEED_DIAG_LARGE_SCALE}"
      --prior_hf_seed_opacity "${PRIOR_HF_SEED_OPACITY}"
      --prior_hf_seed_jitter_scale "${PRIOR_HF_SEED_JITTER_SCALE}"
      --prior_hf_seed_color_residual_gain "${PRIOR_HF_SEED_COLOR_RESIDUAL_GAIN}"
      --prior_hf_seed_guidance_threshold "${PRIOR_HF_SEED_GUIDANCE_THRESHOLD}"
      --prior_hf_seed_edge_highpass_kernel "${PRIOR_HF_SEED_EDGE_HIGHPASS_KERNEL}"
      --prior_hf_seed_edge_guidance_power "${PRIOR_HF_SEED_EDGE_GUIDANCE_POWER}"
      --prior_hf_seed_prune_protect_iters "${PRIOR_HF_SEED_PRUNE_PROTECT_ITERS}"
      --prior_hf_seed_recent_hf_boost "${PRIOR_HF_SEED_RECENT_HF_BOOST}"
      --prior_hf_seed_recent_imprint_weight "${PRIOR_HF_SEED_RECENT_IMPRINT_WEIGHT}"
      --prior_hf_seed_recent_imprint_gain "${PRIOR_HF_SEED_RECENT_IMPRINT_GAIN}"
      --prior_hf_seed_recent_imprint_iters "${PRIOR_HF_SEED_RECENT_IMPRINT_ITERS}"
    )
    if [[ "${PRIOR_HF_SEED_FIRST_CYCLE_ONLY}" == "1" ]]; then
      TRAIN_ARGS+=(--prior_hf_seed_first_cycle_only)
    fi
    if [[ "${PRIOR_HF_SEED_ORIGINAL_ONLY}" == "1" ]]; then
      TRAIN_ARGS+=(--prior_hf_seed_original_only)
    fi
    if [[ "${PRIOR_HF_SEED_SCALE_CLAMP_ENABLE}" == "1" ]]; then
      TRAIN_ARGS+=(
        --prior_hf_seed_scale_clamp_enable
        --prior_hf_seed_scale_clamp_max_axis "${PRIOR_HF_SEED_SCALE_CLAMP_MAX_AXIS}"
        --prior_hf_seed_scale_clamp_min_axis "${PRIOR_HF_SEED_SCALE_CLAMP_MIN_AXIS}"
        --prior_hf_seed_scale_clamp_max_anisotropy "${PRIOR_HF_SEED_SCALE_CLAMP_MAX_ANISOTROPY}"
      )
    fi
    if [[ "${PRIOR_HF_SEED_RAY_FAN_ENABLE}" == "1" ]]; then
      TRAIN_ARGS+=(
        --prior_hf_seed_ray_fan_enable
        --prior_hf_seed_ray_fan_rays "${PRIOR_HF_SEED_RAY_FAN_RAYS}"
        --prior_hf_seed_ray_fan_radius_px "${PRIOR_HF_SEED_RAY_FAN_RADIUS_PX}"
        --prior_hf_seed_ray_fan_include_center "${PRIOR_HF_SEED_RAY_FAN_INCLUDE_CENTER}"
        --prior_hf_seed_ray_fan_guidance_min_ratio "${PRIOR_HF_SEED_RAY_FAN_GUIDANCE_MIN_RATIO}"
        --prior_hf_seed_ray_fan_mask_threshold "${PRIOR_HF_SEED_RAY_FAN_MASK_THRESHOLD}"
      )
    fi
  fi
  if [[ "${PRIOR_HF_LOWFREQ_CLEANUP_ENABLE}" == "1" ]]; then
    TRAIN_ARGS+=(
      --prior_hf_lowfreq_cleanup_enable
      --prior_hf_lowfreq_cleanup_from_iter "${PRIOR_HF_LOWFREQ_CLEANUP_FROM_ITER}"
      --prior_hf_lowfreq_cleanup_until_iter "${PRIOR_HF_LOWFREQ_CLEANUP_UNTIL_ITER}"
      --prior_hf_lowfreq_cleanup_interval "${PRIOR_HF_LOWFREQ_CLEANUP_INTERVAL}"
      --prior_hf_lowfreq_cleanup_stale_iters "${PRIOR_HF_LOWFREQ_CLEANUP_STALE_ITERS}"
      --prior_hf_lowfreq_cleanup_opacity_min "${PRIOR_HF_LOWFREQ_CLEANUP_OPACITY_MIN}"
      --prior_hf_lowfreq_cleanup_scale_max_min "${PRIOR_HF_LOWFREQ_CLEANUP_SCALE_MAX_MIN}"
      --prior_hf_lowfreq_cleanup_scale_max_max "${PRIOR_HF_LOWFREQ_CLEANUP_SCALE_MAX_MAX}"
      --prior_hf_lowfreq_cleanup_anisotropy_max "${PRIOR_HF_LOWFREQ_CLEANUP_ANISOTROPY_MAX}"
      --prior_hf_lowfreq_cleanup_opacity_decay "${PRIOR_HF_LOWFREQ_CLEANUP_OPACITY_DECAY}"
      --prior_hf_lowfreq_cleanup_prune_opacity_below "${PRIOR_HF_LOWFREQ_CLEANUP_PRUNE_OPACITY_BELOW}"
      --prior_hf_lowfreq_cleanup_max_prune_fraction "${PRIOR_HF_LOWFREQ_CLEANUP_MAX_PRUNE_FRACTION}"
      --prior_hf_lowfreq_cleanup_max_prune_count "${PRIOR_HF_LOWFREQ_CLEANUP_MAX_PRUNE_COUNT}"
    )
  fi
  if [[ "${LAYER_FREQUENCY_DYNAMIC_ROOTS}" == "1" ]]; then
    TRAIN_ARGS+=(--layer_frequency_dynamic_roots)
  fi
  if [[ "${LAYER_FREQUENCY_START_HF_PROTECT_NON_SURFACE}" == "1" ]]; then
    TRAIN_ARGS+=(--layer_frequency_start_hf_protect_non_surface)
  fi
  if [[ "${SURFACE_NORMAL_LOCK}" == "1" ]]; then
    TRAIN_ARGS+=(
      --surface_normal_lock
      --surface_normal_lock_payload "${SURFACE_STATE_PAYLOAD}"
      --surface_normal_lock_normal_key "${SURFACE_NORMAL_LOCK_NORMAL_KEY}"
      --surface_normal_lock_anchor_key "${SURFACE_NORMAL_LOCK_ANCHOR_KEY}"
    )
    if [[ "${SURFACE_NORMAL_LOCK_DYNAMIC_ROOTS}" == "1" ]]; then
      TRAIN_ARGS+=(--surface_normal_lock_dynamic_roots)
    fi
  fi
  (
    cd "${MIPSPLATTING_ROOT}"
    export PYTHONPATH="${MIP_PYTHONPATH}"
    "${TRAIN_ARGS[@]}"
  )
else
  echo "[nosr-layerfreq-cleanup-v0] checkpoint exists, skipping training: ${CHECKPOINT_PATH}"
fi

echo
if [[ "${RUN_RENDER_AFTER}" == "1" ]]; then
  echo "[3/4] render on real ${TARGET_IMAGES_SUBDIR} test views"
  RENDER_DIR="${MODEL_DIR}/test/ours_${FINAL_ITER}/test_preds_${RENDER_RESOLUTION}"
  if [[ "${FORCE_RERUN}" == "1" || ! -d "${RENDER_DIR}" ]]; then
    (
      cd "${MIPSPLATTING_ROOT}"
      export PYTHONPATH="${MIP_PYTHONPATH}"
      "${PYTHON_BIN}" render.py \
        -m "${MODEL_DIR}" \
        -s "${EVAL_SCENE_ROOT}" \
        -i "${TARGET_IMAGES_SUBDIR}" \
        -r "${RENDER_RESOLUTION}" \
        --iteration "${FINAL_ITER}" \
        --skip_train
    )
  else
    echo "[nosr-layerfreq-cleanup-v0] render exists, skipping: ${RENDER_DIR}"
  fi
else
  echo "[3/4] skip render (RUN_RENDER_AFTER=${RUN_RENDER_AFTER})"
fi

echo
if [[ "${RUN_METRICS_AFTER}" == "1" ]]; then
  echo "[4/4] summarize PSNR/SSIM"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/summarize_mipsplatting_render_metrics.py \
      --model_dir "${MODEL_DIR}" \
      --iteration "${FINAL_ITER}" \
      --resolution "${RENDER_RESOLUTION}"
  )
  if [[ "${RUN_COMPARE_AFTER}" == "1" && -f "${INPUT_MODEL_DIR}/results_psnr_ssim.json" && -f "${RESULTS_JSON}" ]]; then
    (
      cd "${SOF_ROOT}"
      "${PYTHON_BIN}" scripts/compare_mipsplatting_summary_json.py \
        --baseline_json "${INPUT_MODEL_DIR}/results_psnr_ssim.json" \
        --current_json "${RESULTS_JSON}" \
        --output_json "${COMPARE_JSON}"
    )
  fi
else
  echo "[4/4] skip metrics (RUN_METRICS_AFTER=${RUN_METRICS_AFTER})"
fi

echo
echo "[done] surface payload : ${SURFACE_STATE_PAYLOAD}"
echo "[done] model dir       : ${MODEL_DIR}"
echo "[done] checkpoint      : ${CHECKPOINT_PATH}"
echo "[done] metrics         : ${RESULTS_JSON}"

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

SURFACE_NORMAL_LOCK="${SURFACE_NORMAL_LOCK:-1}"
SURFACE_NORMAL_LOCK_NORMAL_KEY="${SURFACE_NORMAL_LOCK_NORMAL_KEY:-anchor_normal}"
SURFACE_NORMAL_LOCK_ANCHOR_KEY="${SURFACE_NORMAL_LOCK_ANCHOR_KEY:-anchor_xyz}"
SURFACE_NORMAL_LOCK_DYNAMIC_ROOTS="${SURFACE_NORMAL_LOCK_DYNAMIC_ROOTS:-0}"

DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-999999999}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-0}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-1000000}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
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
if [[ "${LAMBDA_SURFACE_START_HF_PRESERVE}" != "0" && "${LAMBDA_SURFACE_START_HF_PRESERVE}" != "0.0" ]]; then
  echo "[nosr-layerfreq-cleanup-v0] start HF preserve: checkpoint=${LAYER_FREQUENCY_START_HF_CHECKPOINT} lf_kernel=${LAYER_FREQUENCY_START_HF_LOWFREQ_KERNEL} lf_thr=${LAYER_FREQUENCY_START_HF_LOWFREQ_THRESHOLD} energy_thr=${LAYER_FREQUENCY_START_HF_ENERGY_THRESHOLD} power=${LAYER_FREQUENCY_START_HF_MASK_POWER} protect_ns=${LAYER_FREQUENCY_START_HF_PROTECT_NON_SURFACE}"
fi
echo "[nosr-layerfreq-cleanup-v0] densify/prune     : from=${DENSIFY_FROM_ITER} until=${DENSIFY_UNTIL_ITER} interval=${DENSIFICATION_INTERVAL}"

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
    --opacity_reset_interval "${OPACITY_RESET_INTERVAL}"
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
  )
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

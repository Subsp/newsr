#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MIPSPLATTING_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${DEFAULT_MIPSPLATTING_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"

PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-qwen_steps1_seed42_rcgm_aligned_images2_train244_v0}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}}"
PRIOR_IMAGE_SUBDIR="${PRIOR_IMAGE_SUBDIR:-fused_priors}"
PRIOR_IMAGE_DIR="${PRIOR_IMAGE_DIR:-${PREPARED_SR_PRIOR_ROOT}/${PRIOR_IMAGE_SUBDIR}}"

REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_2}"
FALLBACK_IMAGES_SUBDIR="${FALLBACK_IMAGES_SUBDIR:-images_2}"
LR_ANCHOR_IMAGES_SUBDIR="${LR_ANCHOR_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"

ITERATIONS="${ITERATIONS:-30000}"
LAMBDA_TEX="${LAMBDA_TEX:-0.40}"
SEQUENCE_SUBPIXEL="${SEQUENCE_SUBPIXEL:-bicubic}"
SEQUENCE_SUBPIXEL_SCALE="${SEQUENCE_SUBPIXEL_SCALE:-4.0}"
TEST_ITERATIONS="${TEST_ITERATIONS:-${ITERATIONS}}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-${ITERATIONS}}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-${ITERATIONS}}"
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-1}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-1}"
TRAIN_PORT="${TRAIN_PORT:-6011}"
FORCE_PREPARE_SCENE="${FORCE_PREPARE_SCENE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_RENDER_AFTER="${RUN_RENDER_AFTER:-1}"
RUN_METRICS_AFTER="${RUN_METRICS_AFTER:-1}"
RUN_COMPARE_AFTER="${RUN_COMPARE_AFTER:-1}"

RUN_TAG="${RUN_TAG:-mip30k_r1_qwen_sequence_loss_srprior_v0_ltex${LAMBDA_TEX}_subpx${SEQUENCE_SUBPIXEL}_30k}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/mipsplatting_sequence_loss_srprior_v0/${SCENE_NAME}}"
MODEL_DIR="${MODEL_DIR:-${RUN_ROOT}/${RUN_TAG}}"
ALIAS_ROOT="${ALIAS_ROOT:-${OUTPUT_ROOT}/colmap_sequence_srprior_scene_v0/${SCENE_NAME}}"
ALIAS_TAG="${ALIAS_TAG:-${PREPARED_SR_PRIOR_NAME}_${PRIOR_IMAGE_SUBDIR}_${REFERENCE_IMAGES_SUBDIR}_v0}"
ALIAS_DIR="${ALIAS_DIR:-${ALIAS_ROOT}/${ALIAS_TAG}}"

BASELINE_MODEL_DIR="${BASELINE_MODEL_DIR:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0}"
CHECKPOINT_PATH="${MODEL_DIR}/chkpnt${ITERATIONS}.pth"
RESULTS_JSON="${MODEL_DIR}/results_psnr_ssim.json"
COMPARE_JSON="${COMPARE_JSON:-${MODEL_DIR}/sequence_loss_compare_vs_mip_baseline.json}"
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
  "${SCENE_ROOT}/sparse/0" \
  "${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}" \
  "${SCENE_ROOT}/${FALLBACK_IMAGES_SUBDIR}" \
  "${SCENE_ROOT}/${LR_ANCHOR_IMAGES_SUBDIR}" \
  "${PRIOR_IMAGE_DIR}" \
  "${MIPSPLATTING_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[mip-sequence-loss-srprior-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}" "${MODEL_DIR}" "${ALIAS_ROOT}"

echo "[mip-sequence-loss-srprior-v0] scene root     : ${SCENE_ROOT}"
echo "[mip-sequence-loss-srprior-v0] SR prior dir   : ${PRIOR_IMAGE_DIR}"
echo "[mip-sequence-loss-srprior-v0] LR anchor      : ${SCENE_ROOT}/${LR_ANCHOR_IMAGES_SUBDIR}"
echo "[mip-sequence-loss-srprior-v0] alias scene    : ${ALIAS_DIR}"
echo "[mip-sequence-loss-srprior-v0] output model   : ${MODEL_DIR}"
echo "[mip-sequence-loss-srprior-v0] loss           : lambda_tex=${LAMBDA_TEX} subpixel=${SEQUENCE_SUBPIXEL} scale=${SEQUENCE_SUBPIXEL_SCALE}"
echo "[mip-sequence-loss-srprior-v0] iterations     : ${ITERATIONS}"

echo
echo "[1/4] prepare COLMAP alias scene with SR priors as images/"
if [[ "${FORCE_PREPARE_SCENE}" == "1" || ! -f "${ALIAS_DIR}/prior_supervision_scene_summary.json" ]]; then
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/prepare_colmap_prior_supervision_scene_v0.py \
      --scene_root "${SCENE_ROOT}" \
      --scene_alias_dir "${ALIAS_DIR}" \
      --prior_dir "${PRIOR_IMAGE_DIR}" \
      --reference_images_subdir "${REFERENCE_IMAGES_SUBDIR}" \
      --fallback_images_subdir "${FALLBACK_IMAGES_SUBDIR}" \
      --output_images_subdir images \
      --missing_policy fallback \
      --link_mode symlink
  )
else
  echo "[mip-sequence-loss-srprior-v0] reuse alias scene: ${ALIAS_DIR}"
fi

echo
echo "[2/4] train from scratch with SequenceMatters-style SR texture + LR subpixel loss"
if [[ "${FORCE_RERUN}" == "1" || ! -f "${CHECKPOINT_PATH}" ]]; then
  (
    cd "${MIPSPLATTING_ROOT}"
    export PYTHONPATH="${MIP_PYTHONPATH}"
    "${PYTHON_BIN}" -m hybrid_sdfgs.train \
      -s "${ALIAS_DIR}" \
      -i images \
      -m "${MODEL_DIR}" \
      -r "${TRAIN_RESOLUTION}" \
      --eval \
      --disable_gui \
      --port "${TRAIN_PORT}" \
      --iterations "${ITERATIONS}" \
      --test_iterations "${TEST_ITERATIONS}" \
      --save_iterations "${SAVE_ITERATIONS}" \
      --checkpoint_iterations "${CHECKPOINT_ITERATIONS}" \
      --sequence_loss_enable \
      --sequence_lambda_tex "${LAMBDA_TEX}" \
      --sequence_subpixel "${SEQUENCE_SUBPIXEL}" \
      --sequence_subpixel_scale "${SEQUENCE_SUBPIXEL_SCALE}" \
      --sequence_lr_anchor_root "${SCENE_ROOT}" \
      --sequence_lr_anchor_subdir "${LR_ANCHOR_IMAGES_SUBDIR}"
  )
else
  echo "[mip-sequence-loss-srprior-v0] checkpoint exists, skipping training: ${CHECKPOINT_PATH}"
fi

echo
if [[ "${RUN_RENDER_AFTER}" == "1" ]]; then
  echo "[3/4] render on real ${TARGET_IMAGES_SUBDIR} test views"
  RENDER_DIR="${MODEL_DIR}/test/ours_${ITERATIONS}/test_preds_${RENDER_RESOLUTION}"
  if [[ "${FORCE_RERUN}" == "1" || ! -d "${RENDER_DIR}" ]]; then
    (
      cd "${MIPSPLATTING_ROOT}"
      export PYTHONPATH="${MIP_PYTHONPATH}"
      "${PYTHON_BIN}" render.py \
        -m "${MODEL_DIR}" \
        -s "${SCENE_ROOT}" \
        -i "${TARGET_IMAGES_SUBDIR}" \
        -r "${RENDER_RESOLUTION}" \
        --iteration "${ITERATIONS}" \
        --skip_train
    )
  else
    echo "[mip-sequence-loss-srprior-v0] render exists, skipping: ${RENDER_DIR}"
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
      --iteration "${ITERATIONS}" \
      --resolution "${RENDER_RESOLUTION}"
  )
  if [[ "${RUN_COMPARE_AFTER}" == "1" && -f "${BASELINE_MODEL_DIR}/results_psnr_ssim.json" && -f "${RESULTS_JSON}" ]]; then
    (
      cd "${SOF_ROOT}"
      "${PYTHON_BIN}" scripts/compare_mipsplatting_summary_json.py \
        --baseline_json "${BASELINE_MODEL_DIR}/results_psnr_ssim.json" \
        --current_json "${RESULTS_JSON}" \
        --output_json "${COMPARE_JSON}"
    )
  fi
else
  echo "[4/4] skip metrics (RUN_METRICS_AFTER=${RUN_METRICS_AFTER})"
fi

echo
echo "[done] alias scene : ${ALIAS_DIR}"
echo "[done] model dir   : ${MODEL_DIR}"
echo "[done] checkpoint  : ${CHECKPOINT_PATH}"
echo "[done] metrics     : ${RESULTS_JSON}"

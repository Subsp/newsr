#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

LR_REFERENCE_IMAGES_SUBDIR="${LR_REFERENCE_IMAGES_SUBDIR:-images_8}"
TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR:-${LR_REFERENCE_IMAGES_SUBDIR}}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
REFERENCE_DIR="${REFERENCE_DIR:-${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}}"
PRIOR_SUPERVISION_IMAGES_SUBDIR="${PRIOR_SUPERVISION_IMAGES_SUBDIR:-${TARGET_IMAGES_SUBDIR}}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
BASELINE_MODEL_DIR="${BASELINE_MODEL_DIR:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}}"
BASELINE_ITERATION="${BASELINE_ITERATION:-30000}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BASELINE_MODEL_DIR}/chkpnt${BASELINE_ITERATION}.pth}"

RAW_PRIOR_DIR="${RAW_PRIOR_DIR:-${WORK_ROOT}/test_preds_1_vosr_same/qwen_steps1_seed42_rcgm}"
PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-qwen_steps1_seed42_rcgm_aligned_images2_train244_v0}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}}"
PRIOR_EDGE_SUBDIR="${PRIOR_EDGE_SUBDIR:-fused_priors}"
PRIOR_MASK_SUBDIR="${PRIOR_MASK_SUBDIR:-usable_masks}"
PRIOR_ANCHOR_SUBDIR="${PRIOR_ANCHOR_SUBDIR:-aligned_references}"
FORCE_PREPARE_PRIORS="${FORCE_PREPARE_PRIORS:-0}"

LOWPASS_KERNEL="${LOWPASS_KERNEL:-9}"
LOWPASS_PRIOR_NAME="${LOWPASS_PRIOR_NAME:-${PREPARED_SR_PRIOR_NAME}_lowpassk${LOWPASS_KERNEL}}"
LOWPASS_PRIOR_ROOT="${LOWPASS_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${LOWPASS_PRIOR_NAME}}"
FORCE_BUILD_LOWPASS="${FORCE_BUILD_LOWPASS:-0}"

RUN_TAG="${RUN_TAG:-mip30k_r1_qwen_freqcurriculum_nomesh_v0}"
RUN_ROOT="${RUN_ROOT:-${SOF_ROOT}/output/mipsplatting_frequency_curriculum_v0/${SCENE_NAME}/${RUN_TAG}}"
STAGE_A_DIR="${STAGE_A_DIR:-${RUN_ROOT}/stage_a_lowfreq_bootstrap}"
STAGE_B_DIR="${STAGE_B_DIR:-${RUN_ROOT}/stage_b_transition_detail}"
STAGE_C_DIR="${STAGE_C_DIR:-${RUN_ROOT}/stage_c_detail_release}"

STAGE_A_END_ITER="${STAGE_A_END_ITER:-30600}"
STAGE_B_END_ITER="${STAGE_B_END_ITER:-31400}"
FINAL_ITER="${FINAL_ITER:-32000}"
FORCE_RERUN="${FORCE_RERUN:-0}"

PRIOR_EDGE_MIN_PIXELS="${PRIOR_EDGE_MIN_PIXELS:-64}"
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-1}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GENERIC_PRIOR_L1_WEIGHT="${GENERIC_PRIOR_L1_WEIGHT:-0.0}"
GENERIC_PRIOR_HF_WEIGHT="${GENERIC_PRIOR_HF_WEIGHT:-0.0}"

STAGE_A_LOCAL_LAMBDA="${STAGE_A_LOCAL_LAMBDA:-0.03}"
STAGE_A_LOCAL_DIR="${STAGE_A_LOCAL_DIR:-${LOWPASS_PRIOR_ROOT}/fused_priors}"
STAGE_A_LOCAL_MASK_DIR="${STAGE_A_LOCAL_MASK_DIR:-${LOWPASS_PRIOR_ROOT}/usable_masks}"

STAGE_B_LOCAL_LAMBDA="${STAGE_B_LOCAL_LAMBDA:-0.015}"
STAGE_B_LOCAL_DIR="${STAGE_B_LOCAL_DIR:-${LOWPASS_PRIOR_ROOT}/fused_priors}"
STAGE_B_LOCAL_MASK_DIR="${STAGE_B_LOCAL_MASK_DIR:-${LOWPASS_PRIOR_ROOT}/usable_masks}"
STAGE_B_EDGE_LAMBDA="${STAGE_B_EDGE_LAMBDA:-0.08}"
STAGE_B_EDGE_DIR="${STAGE_B_EDGE_DIR:-${PREPARED_SR_PRIOR_ROOT}/${PRIOR_EDGE_SUBDIR}}"
STAGE_B_EDGE_MASK_DIR="${STAGE_B_EDGE_MASK_DIR:-${PREPARED_SR_PRIOR_ROOT}/${PRIOR_MASK_SUBDIR}}"
STAGE_B_ALPHA="${STAGE_B_ALPHA:-0.15}"
STAGE_B_ALPHA_FINAL="${STAGE_B_ALPHA_FINAL:-0.35}"
STAGE_B_WARMUP_ITERS="${STAGE_B_WARMUP_ITERS:-$(( STAGE_B_END_ITER - STAGE_A_END_ITER ))}"
STAGE_B_DETAIL_WEIGHT="${STAGE_B_DETAIL_WEIGHT:-1.0}"
STAGE_B_LOWFREQ_WEIGHT="${STAGE_B_LOWFREQ_WEIGHT:-0.08}"
STAGE_B_GRAD_WEIGHT="${STAGE_B_GRAD_WEIGHT:-0.0}"
STAGE_B_LOWFREQ_THRESHOLD="${STAGE_B_LOWFREQ_THRESHOLD:-0.08}"
STAGE_B_LOWFREQ_ANCHOR="${STAGE_B_LOWFREQ_ANCHOR:-render}"
STAGE_B_DETAIL_MIN_GAIN="${STAGE_B_DETAIL_MIN_GAIN:-0.003}"
STAGE_B_CONFIDENCE_POWER="${STAGE_B_CONFIDENCE_POWER:-1.25}"
STAGE_B_UPDATE_SCALE="${STAGE_B_UPDATE_SCALE:-0.30}"

STAGE_C_LOCAL_LAMBDA="${STAGE_C_LOCAL_LAMBDA:-0.005}"
STAGE_C_LOCAL_DIR="${STAGE_C_LOCAL_DIR:-${LOWPASS_PRIOR_ROOT}/fused_priors}"
STAGE_C_LOCAL_MASK_DIR="${STAGE_C_LOCAL_MASK_DIR:-${LOWPASS_PRIOR_ROOT}/usable_masks}"
STAGE_C_EDGE_LAMBDA="${STAGE_C_EDGE_LAMBDA:-0.12}"
STAGE_C_EDGE_DIR="${STAGE_C_EDGE_DIR:-${PREPARED_SR_PRIOR_ROOT}/${PRIOR_EDGE_SUBDIR}}"
STAGE_C_EDGE_MASK_DIR="${STAGE_C_EDGE_MASK_DIR:-${PREPARED_SR_PRIOR_ROOT}/${PRIOR_MASK_SUBDIR}}"
STAGE_C_ALPHA="${STAGE_C_ALPHA:-0.35}"
STAGE_C_ALPHA_FINAL="${STAGE_C_ALPHA_FINAL:-0.60}"
STAGE_C_WARMUP_ITERS="${STAGE_C_WARMUP_ITERS:-$(( FINAL_ITER - STAGE_B_END_ITER ))}"
STAGE_C_DETAIL_WEIGHT="${STAGE_C_DETAIL_WEIGHT:-1.0}"
STAGE_C_LOWFREQ_WEIGHT="${STAGE_C_LOWFREQ_WEIGHT:-0.03}"
STAGE_C_GRAD_WEIGHT="${STAGE_C_GRAD_WEIGHT:-0.02}"
STAGE_C_LOWFREQ_THRESHOLD="${STAGE_C_LOWFREQ_THRESHOLD:-0.08}"
STAGE_C_LOWFREQ_ANCHOR="${STAGE_C_LOWFREQ_ANCHOR:-render}"
STAGE_C_DETAIL_MIN_GAIN="${STAGE_C_DETAIL_MIN_GAIN:-0.005}"
STAGE_C_CONFIDENCE_POWER="${STAGE_C_CONFIDENCE_POWER:-1.5}"
STAGE_C_UPDATE_SCALE="${STAGE_C_UPDATE_SCALE:-0.35}"

if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
  echo "[mip-freq-curriculum-v0] missing baseline checkpoint: ${BASELINE_CHECKPOINT}" >&2
  exit 1
fi
if (( STAGE_A_END_ITER <= BASELINE_ITERATION )); then
  echo "[mip-freq-curriculum-v0] STAGE_A_END_ITER must be > BASELINE_ITERATION." >&2
  exit 1
fi
if (( STAGE_B_END_ITER <= STAGE_A_END_ITER )); then
  echo "[mip-freq-curriculum-v0] STAGE_B_END_ITER must be > STAGE_A_END_ITER." >&2
  exit 1
fi
if (( FINAL_ITER <= STAGE_B_END_ITER )); then
  echo "[mip-freq-curriculum-v0] FINAL_ITER must be > STAGE_B_END_ITER." >&2
  exit 1
fi

prepare_priors_if_needed() {
  local manifest_path="${PREPARED_SR_PRIOR_ROOT}/manifest.json"
  local prior_dir="${PREPARED_SR_PRIOR_ROOT}/${PRIOR_EDGE_SUBDIR}"
  local mask_dir="${PREPARED_SR_PRIOR_ROOT}/${PRIOR_MASK_SUBDIR}"
  local anchor_dir="${PREPARED_SR_PRIOR_ROOT}/${PRIOR_ANCHOR_SUBDIR}"
  if [[ "${FORCE_PREPARE_PRIORS}" != "1" && -f "${manifest_path}" && -d "${prior_dir}" && -d "${mask_dir}" && -d "${anchor_dir}" ]]; then
    return
  fi
  if [[ ! -d "${RAW_PRIOR_DIR}" ]]; then
    echo "[mip-freq-curriculum-v0] missing RAW_PRIOR_DIR=${RAW_PRIOR_DIR}" >&2
    exit 1
  fi
  (
    cd "${SOF_ROOT}"
    SCENE_NAME="${SCENE_NAME}" \
    SCENE_ROOT="${SCENE_ROOT}" \
    SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
    PRIOR_DIR="${RAW_PRIOR_DIR}" \
    REFERENCE_DIR="${REFERENCE_DIR}" \
    REFERENCE_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR}" \
    OUTPUT_NAME="${PREPARED_SR_PRIOR_NAME}" \
    OUTPUT_ROOT="${PREPARED_SR_PRIOR_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/run_align_vosr_prior_size_v0_kitchen.sh"
  )
}

build_lowpass_cache_if_needed() {
  local manifest_path="${LOWPASS_PRIOR_ROOT}/manifest.json"
  local prior_dir="${LOWPASS_PRIOR_ROOT}/fused_priors"
  local mask_dir="${LOWPASS_PRIOR_ROOT}/usable_masks"
  if [[ "${FORCE_BUILD_LOWPASS}" != "1" && -f "${manifest_path}" && -d "${prior_dir}" && -d "${mask_dir}" ]]; then
    return
  fi
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_lowpass_prior_cache_v0.py" \
    --source_root "${PREPARED_SR_PRIOR_ROOT}" \
    --output_root "${LOWPASS_PRIOR_ROOT}" \
    --prior_subdir "${PRIOR_EDGE_SUBDIR}" \
    --mask_subdir "${PRIOR_MASK_SUBDIR}" \
    --anchor_subdir "${PRIOR_ANCHOR_SUBDIR}" \
    --kernel_size "${LOWPASS_KERNEL}"
}

run_stage() {
  local stage_name="$1"
  local model_dir="$2"
  local start_checkpoint="$3"
  local stage_start_iter="$4"
  local stage_end_iter="$5"
  local local_dir="$6"
  local local_mask_dir="$7"
  local local_lambda="$8"
  local edge_dir="$9"
  local edge_mask_dir="${10}"
  local edge_lambda="${11}"
  local edge_alpha="${12}"
  local edge_alpha_final="${13}"
  local edge_warmup_iters="${14}"
  local edge_detail_weight="${15}"
  local edge_lowfreq_weight="${16}"
  local edge_grad_weight="${17}"
  local edge_lowfreq_threshold="${18}"
  local edge_lowfreq_anchor="${19}"
  local edge_detail_min_gain="${20}"
  local edge_confidence_power="${21}"
  local edge_update_scale="${22}"
  local run_render_after="${23}"
  local run_metrics_after="${24}"

  if [[ ! -f "${start_checkpoint}" ]]; then
    echo "[mip-freq-curriculum-v0] missing stage start checkpoint: ${start_checkpoint}" >&2
    exit 1
  fi

  local stage_baseline_model_dir="${start_checkpoint%/*}"
  local stage_checkpoint="${model_dir}/chkpnt${stage_end_iter}.pth"
  if [[ "${FORCE_RERUN}" != "1" && -f "${stage_checkpoint}" ]]; then
    echo "[mip-freq-curriculum-v0] reuse stage checkpoint: ${stage_checkpoint}"
    LAST_STAGE_CHECKPOINT="${stage_checkpoint}"
    return
  fi

  echo
  echo "[mip-freq-curriculum-v0] run ${stage_name}"
  echo "  start checkpoint : ${start_checkpoint}"
  echo "  start model dir  : ${stage_baseline_model_dir}"
  echo "  iter range       : ${stage_start_iter} -> ${stage_end_iter}"
  echo "  train images     : ${TRAIN_IMAGES_SUBDIR}"
  echo "  local branch     : lambda=${local_lambda} dir=${local_dir}"
  echo "  edge branch      : lambda=${edge_lambda} dir=${edge_dir}"
  echo "  edge alpha       : ${edge_alpha} -> ${edge_alpha_final} (warmup=${edge_warmup_iters})"
  echo "  edge low/high    : lowfreq=${edge_lowfreq_weight} detail=${edge_detail_weight} grad=${edge_grad_weight}"
  echo "  generic prior    : l1=${GENERIC_PRIOR_L1_WEIGHT} hf=${GENERIC_PRIOR_HF_WEIGHT}"
  echo "  output model     : ${model_dir}"

  (
    cd "${SOF_ROOT}"
    SCENE_NAME="${SCENE_NAME}" \
    SCENE_ROOT="${SCENE_ROOT}" \
    SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT}" \
    LR_REFERENCE_IMAGES_SUBDIR="${LR_REFERENCE_IMAGES_SUBDIR}" \
    TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR}" \
    TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR}" \
    PRIOR_SUPERVISION_IMAGES_SUBDIR="${PRIOR_SUPERVISION_IMAGES_SUBDIR}" \
    BASELINE_MODEL_DIR="${BASELINE_MODEL_DIR}" \
    BASELINE_ITERATION="${stage_start_iter}" \
    START_CHECKPOINT="${start_checkpoint}" \
    PREPARED_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT}" \
    PRIOR_DIR="${RAW_PRIOR_DIR}" \
    FORCE_PREPARE_PRIORS=0 \
    PREPARED_PRIOR_EXPECTED_REFERENCE_DIR="${REFERENCE_DIR}" \
    PRIOR_SUBDIR="${PRIOR_EDGE_SUBDIR}" \
    PRIOR_MASK_SUBDIR="${PRIOR_MASK_SUBDIR}" \
    PRIOR_L1_WEIGHT="${GENERIC_PRIOR_L1_WEIGHT}" \
    PRIOR_HF_WEIGHT="${GENERIC_PRIOR_HF_WEIGHT}" \
    PRIOR_LOCAL_DIR="${local_dir}" \
    PRIOR_LOCAL_MASK_DIR="${local_mask_dir}" \
    LAMBDA_PRIOR_LOCAL="${local_lambda}" \
    PRIOR_LOCAL_FROM_ITER="${stage_start_iter}" \
    PRIOR_EDGE_DIR="${edge_dir}" \
    PRIOR_EDGE_MASK_DIR="${edge_mask_dir}" \
    LAMBDA_PRIOR_EDGE="${edge_lambda}" \
    PRIOR_EDGE_LOSS_MODE="detail_v1" \
    PRIOR_EDGE_MIN_PIXELS="${PRIOR_EDGE_MIN_PIXELS}" \
    PRIOR_EDGE_FROM_ITER="${stage_start_iter}" \
    PRIOR_EDGE_DETAIL_ALPHA="${edge_alpha}" \
    PRIOR_EDGE_DETAIL_ALPHA_FINAL="${edge_alpha_final}" \
    PRIOR_EDGE_DETAIL_WARMUP_ITERS="${edge_warmup_iters}" \
    PRIOR_EDGE_DETAIL_WEIGHT="${edge_detail_weight}" \
    PRIOR_EDGE_LOWFREQ_WEIGHT="${edge_lowfreq_weight}" \
    PRIOR_EDGE_GRAD_WEIGHT="${edge_grad_weight}" \
    PRIOR_EDGE_LOWFREQ_THRESHOLD="${edge_lowfreq_threshold}" \
    PRIOR_EDGE_LOWFREQ_ANCHOR="${edge_lowfreq_anchor}" \
    PRIOR_EDGE_DETAIL_MIN_GAIN="${edge_detail_min_gain}" \
    PRIOR_EDGE_CONFIDENCE_POWER="${edge_confidence_power}" \
    PRIOR_EDGE_UPDATE_SCALE="${edge_update_scale}" \
    RUN_BASELINE_IF_MISSING=0 \
    ITERATIONS="${stage_end_iter}" \
    TRAIN_RESOLUTION="${TRAIN_RESOLUTION}" \
    RENDER_RESOLUTION="${RENDER_RESOLUTION}" \
    EXPERIMENT_NAME="${stage_name}" \
    RUN_ROOT="${RUN_ROOT}" \
    PRIOR_MODEL_DIR="${model_dir}" \
    RUN_RENDER_AFTER="${run_render_after}" \
    RUN_METRICS_AFTER="${run_metrics_after}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/run_mipsplatting_stablesr_prior_scene.sh"
  )

  LAST_STAGE_CHECKPOINT="${stage_checkpoint}"
}

mkdir -p "${RUN_ROOT}"

prepare_priors_if_needed
build_lowpass_cache_if_needed

echo "[mip-freq-curriculum-v0] scene              : ${SCENE_ROOT}"
echo "[mip-freq-curriculum-v0] baseline model     : ${BASELINE_MODEL_DIR} iter=${BASELINE_ITERATION}"
echo "[mip-freq-curriculum-v0] train images       : ${TRAIN_IMAGES_SUBDIR}"
echo "[mip-freq-curriculum-v0] eval target images : ${TARGET_IMAGES_SUBDIR}"
if [[ -n "${PRIOR_SUPERVISION_IMAGES_SUBDIR}" ]]; then
  echo "[mip-freq-curriculum-v0] prior supervision  : ${PRIOR_SUPERVISION_IMAGES_SUBDIR}"
else
  echo "[mip-freq-curriculum-v0] prior supervision  : disabled"
fi
echo "[mip-freq-curriculum-v0] raw prior dir      : ${RAW_PRIOR_DIR}"
echo "[mip-freq-curriculum-v0] prepared prior     : ${PREPARED_SR_PRIOR_ROOT}"
echo "[mip-freq-curriculum-v0] lowpass prior      : ${LOWPASS_PRIOR_ROOT} (kernel=${LOWPASS_KERNEL})"
echo "[mip-freq-curriculum-v0] iter split         : ${BASELINE_ITERATION} -> ${STAGE_A_END_ITER} -> ${STAGE_B_END_ITER} -> ${FINAL_ITER}"
echo "[mip-freq-curriculum-v0] output root        : ${RUN_ROOT}"

LAST_STAGE_CHECKPOINT="${BASELINE_CHECKPOINT}"
run_stage \
  "stage_a_lowfreq_bootstrap" \
  "${STAGE_A_DIR}" \
  "${LAST_STAGE_CHECKPOINT}" \
  "${BASELINE_ITERATION}" \
  "${STAGE_A_END_ITER}" \
  "${STAGE_A_LOCAL_DIR}" \
  "${STAGE_A_LOCAL_MASK_DIR}" \
  "${STAGE_A_LOCAL_LAMBDA}" \
  "" \
  "" \
  "0.0" \
  "0.0" \
  "0.0" \
  "0" \
  "0.0" \
  "0.0" \
  "0.0" \
  "0.0" \
  "render" \
  "0.0" \
  "1.0" \
  "1.0" \
  "0" \
  "0"

run_stage \
  "stage_b_transition_detail" \
  "${STAGE_B_DIR}" \
  "${LAST_STAGE_CHECKPOINT}" \
  "${STAGE_A_END_ITER}" \
  "${STAGE_B_END_ITER}" \
  "${STAGE_B_LOCAL_DIR}" \
  "${STAGE_B_LOCAL_MASK_DIR}" \
  "${STAGE_B_LOCAL_LAMBDA}" \
  "${STAGE_B_EDGE_DIR}" \
  "${STAGE_B_EDGE_MASK_DIR}" \
  "${STAGE_B_EDGE_LAMBDA}" \
  "${STAGE_B_ALPHA}" \
  "${STAGE_B_ALPHA_FINAL}" \
  "${STAGE_B_WARMUP_ITERS}" \
  "${STAGE_B_DETAIL_WEIGHT}" \
  "${STAGE_B_LOWFREQ_WEIGHT}" \
  "${STAGE_B_GRAD_WEIGHT}" \
  "${STAGE_B_LOWFREQ_THRESHOLD}" \
  "${STAGE_B_LOWFREQ_ANCHOR}" \
  "${STAGE_B_DETAIL_MIN_GAIN}" \
  "${STAGE_B_CONFIDENCE_POWER}" \
  "${STAGE_B_UPDATE_SCALE}" \
  "0" \
  "0"

run_stage \
  "stage_c_detail_release" \
  "${STAGE_C_DIR}" \
  "${LAST_STAGE_CHECKPOINT}" \
  "${STAGE_B_END_ITER}" \
  "${FINAL_ITER}" \
  "${STAGE_C_LOCAL_DIR}" \
  "${STAGE_C_LOCAL_MASK_DIR}" \
  "${STAGE_C_LOCAL_LAMBDA}" \
  "${STAGE_C_EDGE_DIR}" \
  "${STAGE_C_EDGE_MASK_DIR}" \
  "${STAGE_C_EDGE_LAMBDA}" \
  "${STAGE_C_ALPHA}" \
  "${STAGE_C_ALPHA_FINAL}" \
  "${STAGE_C_WARMUP_ITERS}" \
  "${STAGE_C_DETAIL_WEIGHT}" \
  "${STAGE_C_LOWFREQ_WEIGHT}" \
  "${STAGE_C_GRAD_WEIGHT}" \
  "${STAGE_C_LOWFREQ_THRESHOLD}" \
  "${STAGE_C_LOWFREQ_ANCHOR}" \
  "${STAGE_C_DETAIL_MIN_GAIN}" \
  "${STAGE_C_CONFIDENCE_POWER}" \
  "${STAGE_C_UPDATE_SCALE}" \
  "1" \
  "1"

echo
echo "[done] final model      : ${STAGE_C_DIR}"
echo "[done] final checkpoint : ${STAGE_C_DIR}/chkpnt${FINAL_ITER}.pth"
echo "[done] final metrics    : ${STAGE_C_DIR}/results_psnr_ssim.json"

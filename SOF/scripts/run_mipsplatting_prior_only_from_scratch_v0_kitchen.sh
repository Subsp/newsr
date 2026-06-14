#!/usr/bin/env bash
set -euo pipefail

# Canonical prior-from-scratch mainline for new models trained from priors.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MIPSPLATTING_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-qwen_steps1_seed42_rcgm_aligned_images2_train244_v0}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}}"
PRIOR_IMAGE_SUBDIR="${PRIOR_IMAGE_SUBDIR:-fused_priors}"
PRIOR_IMAGE_DIR="${PRIOR_IMAGE_DIR:-${PREPARED_SR_PRIOR_ROOT}/${PRIOR_IMAGE_SUBDIR}}"

REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_2}"
FALLBACK_IMAGES_SUBDIR="${FALLBACK_IMAGES_SUBDIR:-images_2}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"

MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${DEFAULT_MIPSPLATTING_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"
RUN_TAG="${RUN_TAG:-mip30k_r1_qwen_prioronly_scratch_v0}"
ALIAS_ROOT="${ALIAS_ROOT:-${OUTPUT_ROOT}/colmap_prior_supervision_scene_v0/${SCENE_NAME}}"
ALIAS_DIR="${ALIAS_DIR:-${ALIAS_ROOT}/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/mipsplatting_prior_only_from_scratch_v0/${SCENE_NAME}}"
MODEL_DIR="${MODEL_DIR:-${RUN_ROOT}/${RUN_TAG}}"

ITERATIONS="${ITERATIONS:-30000}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_RENDER_AFTER="${RUN_RENDER_AFTER:-1}"
RUN_METRICS_AFTER="${RUN_METRICS_AFTER:-1}"
LINK_MODE="${LINK_MODE:-symlink}"

CHECKPOINT_PATH="${MODEL_DIR}/chkpnt${ITERATIONS}.pth"
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
  "${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}" \
  "${PRIOR_IMAGE_DIR}" \
  "${MIPSPLATTING_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[mip-prioronly-scratch-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${ALIAS_ROOT}" "${RUN_ROOT}" "${MODEL_DIR}"

echo "[mip-prioronly-scratch-v0] scene            : ${SCENE_NAME}"
echo "[mip-prioronly-scratch-v0] scene root       : ${SCENE_ROOT}"
echo "[mip-prioronly-scratch-v0] prior dir        : ${PRIOR_IMAGE_DIR}"
echo "[mip-prioronly-scratch-v0] reference images : ${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}"
echo "[mip-prioronly-scratch-v0] fallback images  : ${SCENE_ROOT}/${FALLBACK_IMAGES_SUBDIR}"
echo "[mip-prioronly-scratch-v0] alias scene      : ${ALIAS_DIR}"
echo "[mip-prioronly-scratch-v0] mip root         : ${MIPSPLATTING_ROOT}"
echo "[mip-prioronly-scratch-v0] model dir        : ${MODEL_DIR}"
echo "[mip-prioronly-scratch-v0] iterations       : ${ITERATIONS}"
echo "[mip-prioronly-scratch-v0] resolution flag  : ${RENDER_RESOLUTION}"
echo "[mip-prioronly-scratch-v0] from checkpoint  : none (scratch)"

echo
echo "[1/4] prepare COLMAP prior-supervision alias"
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
    --link_mode "${LINK_MODE}"
)

echo
echo "[2/4] train mip-splatting from scratch on prior images"
if [[ "${FORCE_RERUN}" == "1" || ! -f "${CHECKPOINT_PATH}" ]]; then
  (
    cd "${MIPSPLATTING_ROOT}"
    export PYTHONPATH="${MIP_PYTHONPATH}"
    "${PYTHON_BIN}" train.py \
      -s "${ALIAS_DIR}" \
      -i images \
      -m "${MODEL_DIR}" \
      -r "${RENDER_RESOLUTION}" \
      --eval \
      --iterations "${ITERATIONS}" \
      --test_iterations "${ITERATIONS}" \
      --save_iterations "${ITERATIONS}" \
      --checkpoint_iterations "${ITERATIONS}"
  )
else
  echo "[mip-prioronly-scratch-v0] checkpoint exists, skipping training: ${CHECKPOINT_PATH}"
fi

if [[ "${RUN_RENDER_AFTER}" == "1" ]]; then
  echo
  echo "[3/4] render on real ${TARGET_IMAGES_SUBDIR} test views"
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
  echo
  echo "[3/4] skip render (RUN_RENDER_AFTER=${RUN_RENDER_AFTER})"
fi

if [[ "${RUN_METRICS_AFTER}" == "1" ]]; then
  echo
  echo "[4/4] summarize PSNR/SSIM against real ${TARGET_IMAGES_SUBDIR} test views"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/summarize_mipsplatting_render_metrics.py \
      --model_dir "${MODEL_DIR}" \
      --iteration "${ITERATIONS}" \
      --resolution "${RENDER_RESOLUTION}"
  )
else
  echo
  echo "[4/4] skip metrics (RUN_METRICS_AFTER=${RUN_METRICS_AFTER})"
fi

echo
echo "[done] alias scene: ${ALIAS_DIR}"
echo "[done] model dir  : ${MODEL_DIR}"
echo "[done] checkpoint : ${CHECKPOINT_PATH}"
echo "[done] metrics    : ${MODEL_DIR}/results_psnr_ssim.json"

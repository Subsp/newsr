#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MIPSPLATTING_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${WORK_ROOT}/archive}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
if [[ -z "${SCENE_ROOT:-}" ]]; then
  DIRECT_SCENE_ROOT="${WORK_ROOT}/${SCENE_NAME}"
  ARCHIVE_SCENE_ROOT="${ARCHIVE_ROOT}/${SCENE_NAME}"
  if [[ -d "${DIRECT_SCENE_ROOT}" ]]; then
    SCENE_ROOT="${DIRECT_SCENE_ROOT}"
  else
    SCENE_ROOT="${ARCHIVE_SCENE_ROOT}"
  fi
else
  SCENE_ROOT="${SCENE_ROOT}"
fi

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
SOURCE_IMAGES_DIR="${SOURCE_IMAGES_DIR:-${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}}"
TARGET_IMAGES_DIR="${TARGET_IMAGES_DIR:-${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}}"
USE_ALIAS_TRAINING="${USE_ALIAS_TRAINING:-0}"

ALIAS_ROOT="${ALIAS_ROOT:-${WORK_ROOT}/aliases}"
ALIAS_DIR="${ALIAS_DIR:-${ALIAS_ROOT}/${SCENE_NAME}_images8bicubic_to_images2}"

MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${DEFAULT_MIPSPLATTING_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-${SCENE_NAME}_mipsplatting_lr_ablation_v1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mipsplatting_x8to2_baseline_v0}"
MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_GROUP}/${EXPERIMENT_NAME}}"

ITERATIONS="${ITERATIONS:-30000}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${MODEL_DIR}/chkpnt${ITERATIONS}.pth}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
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

MIP_PYTHONPATH="${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${ALIAS_ROOT}" "${MODEL_DIR}"

for path in "${SCENE_ROOT}" "${TARGET_IMAGES_DIR}" "${MIPSPLATTING_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[mipsplatting-x8to2-baseline] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${SCENE_ROOT}/sparse/0" ]]; then
  echo "[mipsplatting-x8to2-baseline] missing sparse/0: ${SCENE_ROOT}/sparse/0" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_IMAGES_DIR}" ]]; then
  echo "[mipsplatting-x8to2-baseline] ${SOURCE_IMAGES_SUBDIR} missing, generating from ${TARGET_IMAGES_SUBDIR}"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/generate_downsampled_images.py \
      --source_dir "${TARGET_IMAGES_DIR}" \
      --output_dir "${SOURCE_IMAGES_DIR}" \
      --scale 4 \
      --resize_filter bicubic
  )
fi

TRAIN_SCENE_ROOT="${SCENE_ROOT}"
TRAIN_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR}"
if [[ "${USE_ALIAS_TRAINING}" == "1" ]]; then
  TRAIN_SCENE_ROOT="${ALIAS_DIR}"
  TRAIN_IMAGES_SUBDIR="images"
fi

echo "[mipsplatting-x8to2-baseline] scene            : ${SCENE_NAME}"
echo "[mipsplatting-x8to2-baseline] scene root       : ${SCENE_ROOT}"
echo "[mipsplatting-x8to2-baseline] source images    : ${SOURCE_IMAGES_DIR}"
echo "[mipsplatting-x8to2-baseline] target images    : ${TARGET_IMAGES_DIR}"
echo "[mipsplatting-x8to2-baseline] alias dir        : ${ALIAS_DIR}"
echo "[mipsplatting-x8to2-baseline] train scene root : ${TRAIN_SCENE_ROOT}"
echo "[mipsplatting-x8to2-baseline] train images     : ${TRAIN_IMAGES_SUBDIR}"
echo "[mipsplatting-x8to2-baseline] mipsplatting dir : ${MIPSPLATTING_ROOT}"
echo "[mipsplatting-x8to2-baseline] model dir        : ${MODEL_DIR}"
echo "[mipsplatting-x8to2-baseline] iterations       : ${ITERATIONS}"
echo "[mipsplatting-x8to2-baseline] resolution flag  : ${RENDER_RESOLUTION}"

if [[ "${USE_ALIAS_TRAINING}" == "1" ]]; then
  echo
  echo "[1/4] prepare pseudo-scene alias"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/prepare_colmap_pseudo_sr_scene.py \
      --scene_root "${SCENE_ROOT}" \
      --scene_alias_dir "${ALIAS_DIR}" \
      --source_images_subdir "${SOURCE_IMAGES_SUBDIR}" \
      --target_images_subdir "${TARGET_IMAGES_SUBDIR}" \
      --resize_filter bicubic
  )
else
  echo
  echo "[1/4] skip alias preparation (training directly from ${SOURCE_IMAGES_SUBDIR})"
fi

echo
echo "[2/4] train mip-splatting baseline"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  (
    cd "${MIPSPLATTING_ROOT}"
    export PYTHONPATH="${MIP_PYTHONPATH}"
    "${PYTHON_BIN}" train.py \
      -s "${TRAIN_SCENE_ROOT}" \
      -i "${TRAIN_IMAGES_SUBDIR}" \
      -m "${MODEL_DIR}" \
      -r "${RENDER_RESOLUTION}" \
      --eval \
      --iterations "${ITERATIONS}" \
      --test_iterations "${ITERATIONS}" \
      --save_iterations "${ITERATIONS}" \
      --checkpoint_iterations "${ITERATIONS}"
  )
fi

echo
echo "[3/4] render on ${TARGET_IMAGES_SUBDIR}"
RENDER_DIR="${MODEL_DIR}/test/ours_${ITERATIONS}/test_preds_${RENDER_RESOLUTION}"
if [[ ! -d "${RENDER_DIR}" ]]; then
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
fi

echo
echo "[4/4] summarize PSNR/SSIM"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" scripts/summarize_mipsplatting_render_metrics.py \
    --model_dir "${MODEL_DIR}" \
    --iteration "${ITERATIONS}" \
    --resolution "${RENDER_RESOLUTION}"
)

echo
echo "[done] model dir : ${MODEL_DIR}"
echo "[done] render dir: ${MODEL_DIR}/test/ours_${ITERATIONS}/test_preds_${RENDER_RESOLUTION}"
echo "[done] gt dir    : ${MODEL_DIR}/test/ours_${ITERATIONS}/gt_${RENDER_RESOLUTION}"
echo "[done] metrics   : ${MODEL_DIR}/results_psnr_ssim.json"

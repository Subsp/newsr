#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VGGTSR_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"
DEFAULT_MIPSPLATTING_ROOT="${VGGTSR_ROOT}/mip-splatting"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

TRAIN_IMAGES_SUBDIR="${TRAIN_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
TRAIN_IMAGES_DIR="${TRAIN_IMAGES_DIR:-${SCENE_ROOT}/${TRAIN_IMAGES_SUBDIR}}"
TARGET_IMAGES_DIR="${TARGET_IMAGES_DIR:-${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}}"

MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${DEFAULT_MIPSPLATTING_ROOT}}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mip30k}"
MODEL_DIR="${MODEL_DIR:-${SCENE_ASSET_ROOT}/${EXPERIMENT_GROUP}/${EXPERIMENT_NAME}}"

ITERATIONS="${ITERATIONS:-30000}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

SWITCH_RASTERIZER="${SWITCH_RASTERIZER:-1}"
RESTORE_SOF_RASTERIZER="${RESTORE_SOF_RASTERIZER:-1}"
MIP_RASTERIZER_DIR="${MIP_RASTERIZER_DIR:-${MIPSPLATTING_ROOT}/submodules/diff-gaussian-rasterization}"
SOF_RASTERIZER_DIR="${SOF_RASTERIZER_DIR:-${SOF_ROOT}/submodules/diff-gaussian-rasterization}"

RUN_RENDER_AFTER="${RUN_RENDER_AFTER:-0}"
RUN_METRICS_AFTER="${RUN_METRICS_AFTER:-0}"
START_CHECKPOINT="${START_CHECKPOINT:-}"

TEST_ITERATIONS="${TEST_ITERATIONS:-${ITERATIONS}}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-${ITERATIONS}}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-${ITERATIONS}}"
CHECKPOINT_PATH="${MODEL_DIR}/chkpnt${ITERATIONS}.pth"

export PYTHONUNBUFFERED=1
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "${MODEL_DIR}"

for path in "${SCENE_ROOT}" "${TRAIN_IMAGES_DIR}" "${TARGET_IMAGES_DIR}" "${SCENE_ROOT}/sparse/0" "${MIPSPLATTING_ROOT}" "${MIP_RASTERIZER_DIR}" "${SOF_RASTERIZER_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[mip-images8-30k] required path not found: ${path}" >&2
    exit 1
  fi
done

SWITCHED_RASTERIZER=0
restore_sof_rasterizer() {
  if [[ "${SWITCHED_RASTERIZER}" != "1" || "${RESTORE_SOF_RASTERIZER}" != "1" ]]; then
    return 0
  fi
  echo
  echo "[cleanup] restoring SOF rasterizer"
  "${PYTHON_BIN}" -m pip install --force-reinstall --no-build-isolation --no-cache-dir "${SOF_RASTERIZER_DIR}"
}
trap restore_sof_rasterizer EXIT

echo "[mip-images8-30k] scene             : ${SCENE_NAME}"
echo "[mip-images8-30k] scene root        : ${SCENE_ROOT}"
echo "[mip-images8-30k] asset root        : ${SCENE_ASSET_ROOT}"
echo "[mip-images8-30k] train images      : ${TRAIN_IMAGES_DIR}"
echo "[mip-images8-30k] target images     : ${TARGET_IMAGES_DIR}"
echo "[mip-images8-30k] mip root          : ${MIPSPLATTING_ROOT}"
echo "[mip-images8-30k] model dir         : ${MODEL_DIR}"
echo "[mip-images8-30k] iterations        : ${ITERATIONS}"
echo "[mip-images8-30k] render resolution : ${RENDER_RESOLUTION}"
if [[ -n "${START_CHECKPOINT}" ]]; then
  echo "[mip-images8-30k] start checkpoint  : ${START_CHECKPOINT}"
fi

if [[ "${SWITCH_RASTERIZER}" == "1" ]]; then
  echo
  echo "[0/3] switch active rasterizer to mip-splatting build"
  "${PYTHON_BIN}" -m pip install --force-reinstall --no-build-isolation --no-cache-dir "${MIP_RASTERIZER_DIR}"
  SWITCHED_RASTERIZER=1
fi

echo
echo "[1/3] train mip-splatting on ${TRAIN_IMAGES_SUBDIR}"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  TRAIN_CMD=(
    "${PYTHON_BIN}" train.py
    -s "${SCENE_ROOT}"
    -i "${TRAIN_IMAGES_SUBDIR}"
    -m "${MODEL_DIR}"
    -r "${RENDER_RESOLUTION}"
    --eval
    --iterations "${ITERATIONS}"
    --test_iterations "${TEST_ITERATIONS}"
    --save_iterations "${SAVE_ITERATIONS}"
    --checkpoint_iterations "${CHECKPOINT_ITERATIONS}"
  )
  if [[ -n "${START_CHECKPOINT}" ]]; then
    TRAIN_CMD+=(--start_checkpoint "${START_CHECKPOINT}")
  fi
  (
    cd "${MIPSPLATTING_ROOT}"
    "${TRAIN_CMD[@]}"
  )
else
  echo "[mip-images8-30k] checkpoint already exists, skipping train: ${CHECKPOINT_PATH}"
fi

if [[ "${RUN_RENDER_AFTER}" == "1" ]]; then
  echo
  echo "[2/3] render on ${TARGET_IMAGES_SUBDIR}"
  (
    cd "${MIPSPLATTING_ROOT}"
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
  echo "[2/3] skip render (RUN_RENDER_AFTER=${RUN_RENDER_AFTER})"
fi

if [[ "${RUN_METRICS_AFTER}" == "1" ]]; then
  echo
  echo "[3/3] summarize PSNR/SSIM"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/summarize_mipsplatting_render_metrics.py \
      --model_dir "${MODEL_DIR}" \
      --iteration "${ITERATIONS}" \
      --resolution "${RENDER_RESOLUTION}"
  )
else
  echo
  echo "[3/3] skip metrics (RUN_METRICS_AFTER=${RUN_METRICS_AFTER})"
fi

echo
echo "[done] model dir : ${MODEL_DIR}"
echo "[done] checkpoint: ${CHECKPOINT_PATH}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-/root/autodl-tmp/archive}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${ARCHIVE_ROOT}/${SCENE_NAME}}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-images_2}"
SOURCE_IMAGES_DIR="${SOURCE_IMAGES_DIR:-${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}}"
TARGET_IMAGES_DIR="${TARGET_IMAGES_DIR:-${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}}"

ALIAS_ROOT="${ALIAS_ROOT:-${ARCHIVE_ROOT}/aliases}"
ALIAS_DIR="${ALIAS_DIR:-${ALIAS_ROOT}/${SCENE_NAME}_images8bicubic_to_images2}"

ITERATIONS="${ITERATIONS:-30000}"
SPLATTING_CONFIG="${SPLATTING_CONFIG:-configs/hierarchical.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vanilla3dgs_no_sof_reg}"
MODEL_DIR="${MODEL_DIR:-${SOF_ROOT}/output/${SCENE_NAME}_vanilla3dgs_lr_ablation_v1/${EXPERIMENT_NAME}}"
CHECKPOINT_PATH="${MODEL_DIR}/chkpnt${ITERATIONS}.pth"

PYTHON_BIN="${PYTHON_BIN:-python}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sof

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${ALIAS_ROOT}" "${MODEL_DIR}"

for path in "${SCENE_ROOT}" "${TARGET_IMAGES_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[vanilla3dgs-no-sof-reg] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${SCENE_ROOT}/sparse/0" ]]; then
  echo "[vanilla3dgs-no-sof-reg] missing sparse/0: ${SCENE_ROOT}/sparse/0" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_IMAGES_DIR}" ]]; then
  echo "[vanilla3dgs-no-sof-reg] ${SOURCE_IMAGES_SUBDIR} missing, generating from ${TARGET_IMAGES_SUBDIR}"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/generate_downsampled_images.py \
      --source_dir "${TARGET_IMAGES_DIR}" \
      --output_dir "${SOURCE_IMAGES_DIR}" \
      --scale 4 \
      --resize_filter bicubic
  )
fi

echo "[vanilla3dgs-no-sof-reg] scene          : ${SCENE_NAME}"
echo "[vanilla3dgs-no-sof-reg] scene root     : ${SCENE_ROOT}"
echo "[vanilla3dgs-no-sof-reg] source images  : ${SOURCE_IMAGES_DIR}"
echo "[vanilla3dgs-no-sof-reg] target images  : ${TARGET_IMAGES_DIR}"
echo "[vanilla3dgs-no-sof-reg] alias dir      : ${ALIAS_DIR}"
echo "[vanilla3dgs-no-sof-reg] model dir      : ${MODEL_DIR}"
echo "[vanilla3dgs-no-sof-reg] iterations     : ${ITERATIONS}"

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

echo
echo "[2/4] train vanilla3dgs_no_sof_reg"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" train.py \
    --splatting_config "${SPLATTING_CONFIG}" \
    -s "${ALIAS_DIR}" \
    -m "${MODEL_DIR}" \
    --eval \
    --iterations "${ITERATIONS}" \
    --test_iterations "${ITERATIONS}" \
    --save_iterations "${ITERATIONS}" \
    --checkpoint_iterations "${ITERATIONS}" \
    --lambda_distortion 0.0 \
    --lambda_depth_normal 0.0 \
    --lambda_smoothness 0.0 \
    --lambda_opacity_field 0.0 \
    --lambda_extent 0.0 \
    --distortion_from_iter 99999999 \
    --depth_normal_from_iter 99999999 \
    --scale_reg 0.0 \
    --opacity_reg 0.0 \
    --min_scale_reg 0.0
)

echo
echo "[3/4] render on ${TARGET_IMAGES_SUBDIR}"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" render.py \
    -m "${MODEL_DIR}" \
    -s "${SCENE_ROOT}" \
    -i "${TARGET_IMAGES_SUBDIR}" \
    --iteration "${ITERATIONS}" \
    --eval \
    --skip_train \
    --data_device cpu
)

echo
echo "[4/4] metrics"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" metrics.py -m "${MODEL_DIR}"
)

echo
echo "[done] model dir : ${MODEL_DIR}"
echo "[done] render dir: ${MODEL_DIR}/test/ours_${ITERATIONS}"
echo "[done] metrics   : ${MODEL_DIR}/results_full.json"

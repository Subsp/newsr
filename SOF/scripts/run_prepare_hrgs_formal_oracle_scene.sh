#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VGGTSR_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${WORK_ROOT}/mip-splatting}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

HR_IMAGE_SUBDIR="${HR_IMAGE_SUBDIR:-images_2}"
ORACLE_ITER="${ORACLE_ITER:-30000}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgsrefiner_formal_scene_v0/${SCENE_NAME}/oracle/formal_oracle_v0}"
ORACLE_MODEL_DIR="${ORACLE_MODEL_DIR:-${OUTPUT_ROOT}/model}"
ORACLE_DEPTH_ROOT="${ORACLE_DEPTH_ROOT:-${OUTPUT_ROOT}}"
ORACLE_DEPTH_DIR="${ORACLE_DEPTH_DIR:-${ORACLE_DEPTH_ROOT}/depth}"

TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
FORCE_RENDER="${FORCE_RENDER:-0}"
SWITCH_RASTERIZER="${SWITCH_RASTERIZER:-1}"
RESTORE_SOF_RASTERIZER="${RESTORE_SOF_RASTERIZER:-1}"
MIP_RASTERIZER_DIR="${MIP_RASTERIZER_DIR:-${MIPSPLATTING_ROOT}/submodules/diff-gaussian-rasterization}"
SOF_RASTERIZER_DIR="${SOF_RASTERIZER_DIR:-${SOF_ROOT}/submodules/diff-gaussian-rasterization}"
RESUME_IF_AVAILABLE="${RESUME_IF_AVAILABLE:-1}"
ORACLE_LAMBDA_DSSIM="${ORACLE_LAMBDA_DSSIM:-0.2}"
ORACLE_TEST_ITERS="${ORACLE_TEST_ITERS:--1}"
ORACLE_SAVE_ITERS="${ORACLE_SAVE_ITERS:-1000 2000 4000 6000 7000 10000 14000 21000 30000}"
ORACLE_CKPT_ITERS="${ORACLE_CKPT_ITERS:-1000 2000 4000 6000 7000 10000 14000 21000 30000}"
START_CHECKPOINT="${START_CHECKPOINT:-}"

export PYTHONUNBUFFERED=1
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUTPUT_ROOT}" "${ORACLE_MODEL_DIR}" "${ORACLE_DEPTH_DIR}"

for path in "${SCENE_ROOT}" "${MIPSPLATTING_ROOT}" "${VGGTSR_ROOT}" "${MIP_RASTERIZER_DIR}" "${SOF_RASTERIZER_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[prepare-formal-oracle] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${SCENE_ROOT}/${HR_IMAGE_SUBDIR}" ]]; then
  echo "[prepare-formal-oracle] image directory not found: ${SCENE_ROOT}/${HR_IMAGE_SUBDIR}" >&2
  exit 1
fi

CHECKPOINT_PATH="${ORACLE_MODEL_DIR}/chkpnt${ORACLE_ITER}.pth"
POINT_CLOUD_DIR="${ORACLE_MODEL_DIR}/point_cloud/iteration_${ORACLE_ITER}"
HAS_ORACLE_CHECKPOINT=0
HAS_ORACLE_POINT_CLOUD=0
HAS_ORACLE_STATE=0
if [[ -f "${CHECKPOINT_PATH}" ]]; then
  HAS_ORACLE_CHECKPOINT=1
  HAS_ORACLE_STATE=1
fi
if [[ -d "${POINT_CLOUD_DIR}" ]]; then
  HAS_ORACLE_POINT_CLOUD=1
  HAS_ORACLE_STATE=1
fi
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

LATEST_PARTIAL_CHECKPOINT=""
if compgen -G "${ORACLE_MODEL_DIR}/chkpnt*.pth" > /dev/null; then
  LATEST_PARTIAL_CHECKPOINT="$(find "${ORACLE_MODEL_DIR}" -maxdepth 1 -name 'chkpnt*.pth' | sort -V | tail -n 1)"
fi
if [[ -z "${START_CHECKPOINT}" && "${RESUME_IF_AVAILABLE}" == "1" && -n "${LATEST_PARTIAL_CHECKPOINT}" && "${HAS_ORACLE_STATE}" != "1" ]]; then
  START_CHECKPOINT="${LATEST_PARTIAL_CHECKPOINT}"
fi

echo "[prepare-formal-oracle] scene root        : ${SCENE_ROOT}"
echo "[prepare-formal-oracle] image subdir      : ${HR_IMAGE_SUBDIR}"
echo "[prepare-formal-oracle] mip root          : ${MIPSPLATTING_ROOT}"
echo "[prepare-formal-oracle] oracle model dir  : ${ORACLE_MODEL_DIR}"
echo "[prepare-formal-oracle] oracle depth root : ${ORACLE_DEPTH_ROOT}"
echo "[prepare-formal-oracle] oracle depth dir  : ${ORACLE_DEPTH_DIR}"
echo "[prepare-formal-oracle] oracle iteration  : ${ORACLE_ITER}"
echo "[prepare-formal-oracle] oracle checkpoint : ${HAS_ORACLE_CHECKPOINT}"
echo "[prepare-formal-oracle] oracle pointcloud : ${HAS_ORACLE_POINT_CLOUD}"
echo "[prepare-formal-oracle] switch rasterizer : ${SWITCH_RASTERIZER}"
echo "[prepare-formal-oracle] lambda_dssim      : ${ORACLE_LAMBDA_DSSIM}"
echo "[prepare-formal-oracle] test iterations   : ${ORACLE_TEST_ITERS}"
echo "[prepare-formal-oracle] save iterations   : ${ORACLE_SAVE_ITERS}"
echo "[prepare-formal-oracle] ckpt iterations   : ${ORACLE_CKPT_ITERS}"
if [[ -n "${START_CHECKPOINT}" ]]; then
  echo "[prepare-formal-oracle] resume checkpoint : ${START_CHECKPOINT}"
fi

if [[ "${SWITCH_RASTERIZER}" == "1" ]]; then
  echo
  echo "[0/3] switch active rasterizer to mip-splatting build"
  "${PYTHON_BIN}" -m pip install --force-reinstall --no-build-isolation --no-cache-dir "${MIP_RASTERIZER_DIR}"
  SWITCHED_RASTERIZER=1
fi

if [[ "${HAS_ORACLE_STATE}" != "1" ]]; then
  if [[ "${TRAIN_IF_MISSING}" != "1" ]]; then
    echo "[prepare-formal-oracle] missing oracle state for iter ${ORACLE_ITER} and TRAIN_IF_MISSING=0" >&2
    echo "[prepare-formal-oracle] expected checkpoint: ${CHECKPOINT_PATH}" >&2
    echo "[prepare-formal-oracle] expected point cloud: ${POINT_CLOUD_DIR}" >&2
    exit 1
  fi
  echo
  echo "[1/3] train mip-splatting oracle model on ${HR_IMAGE_SUBDIR}"
  read -r -a TEST_ITERS_ARR <<< "${ORACLE_TEST_ITERS}"
  read -r -a SAVE_ITERS_ARR <<< "${ORACLE_SAVE_ITERS}"
  read -r -a CKPT_ITERS_ARR <<< "${ORACLE_CKPT_ITERS}"
  CMD=(
    "${PYTHON_BIN}" train.py
    -s "${SCENE_ROOT}"
    -i "${HR_IMAGE_SUBDIR}"
    -m "${ORACLE_MODEL_DIR}"
    --eval
    --iterations "${ORACLE_ITER}"
    --lambda_dssim "${ORACLE_LAMBDA_DSSIM}"
    --test_iterations "${TEST_ITERS_ARR[@]}"
    --save_iterations "${SAVE_ITERS_ARR[@]}"
    --checkpoint_iterations "${CKPT_ITERS_ARR[@]}"
  )
  if [[ -n "${START_CHECKPOINT}" ]]; then
    CMD+=(--start_checkpoint "${START_CHECKPOINT}")
  fi
  (
    cd "${MIPSPLATTING_ROOT}"
    "${CMD[@]}"
  )
else
  echo
  if [[ "${HAS_ORACLE_CHECKPOINT}" == "1" ]]; then
    echo "[1/3] reuse existing oracle checkpoint: ${CHECKPOINT_PATH}"
  else
    echo "[1/3] reuse existing oracle point cloud: ${POINT_CLOUD_DIR}"
  fi
fi

DEPTH_COUNT=0
if compgen -G "${ORACLE_DEPTH_DIR}/*.npy" > /dev/null; then
  DEPTH_COUNT="$(find "${ORACLE_DEPTH_DIR}" -maxdepth 1 -name '*.npy' | wc -l | tr -d ' ')"
fi

if [[ "${FORCE_RENDER}" == "1" || "${DEPTH_COUNT}" == "0" ]]; then
  echo
  echo "[2/3] render oracle depth maps"
  (
    cd "${VGGTSR_ROOT}/experiments"
    "${PYTHON_BIN}" task02_render_oracle_depth.py \
      --mip_root "${MIPSPLATTING_ROOT}" \
      --model_path "${ORACLE_MODEL_DIR}" \
      --source_path "${SCENE_ROOT}" \
      --output_dir "${ORACLE_DEPTH_DIR}" \
      --images "${HR_IMAGE_SUBDIR}" \
      --iteration "${ORACLE_ITER}"
  )
else
  echo
  echo "[2/3] reuse existing oracle depths (${DEPTH_COUNT} npy files)"
fi

echo
echo "[3/3] summarize oracle coverage and depth stats"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" scripts/summarize_oracle_depth_dir.py \
    --oracle_root "${ORACLE_DEPTH_ROOT}" \
    --scene_root "${SCENE_ROOT}" \
    --images_subdir "${HR_IMAGE_SUBDIR}" \
    --output_dir "${ORACLE_DEPTH_ROOT}"
)

echo
echo "[done] formal oracle root : ${ORACLE_DEPTH_ROOT}"
echo "[done] trainer can use     : ORACLE_ROOT=${ORACLE_DEPTH_ROOT}"
echo "[done] depth files         : ${ORACLE_DEPTH_DIR}"

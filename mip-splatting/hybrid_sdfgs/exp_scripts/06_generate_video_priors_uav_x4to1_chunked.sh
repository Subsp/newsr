#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

pick_first_existing_dir() {
  for p in "$@"; do
    if [[ -d "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

HBSR_ROOT="${HBSR_ROOT:-${WORKSPACE_ROOT}}"
PYTHON_EXE="${PYTHON_EXE:-python}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/kitchen}"
INPUT_DIR="${INPUT_DIR:-${DATASET_ROOT}/images_4}"
GT_DIR="${GT_DIR:-${DATASET_ROOT}/images}"
GT2_DIR="${GT2_DIR:-${DATASET_ROOT}/images_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_video_uav_x4to1_chunked}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
UAV_PERFORM_TILE="${UAV_PERFORM_TILE:-1}"
UAV_TILE_SIZE="${UAV_TILE_SIZE:-384}"

UAV_REPO="${UAV_REPO:-$(pick_first_existing_dir \
  /root/autodl-tmp/Upscale-A-Video \
  /root/autodl-tmp/HBSR/video_sr_models/Upscale-A-Video \
  /Users/ltl/Desktop/codex_playground/video_sr_models/Upscale-A-Video)}"

UAV_CKPT_SRC="${UAV_CKPT_SRC:-$(pick_first_existing_dir \
  /root/autodl-tmp/Upscale-A-Video/pretrained_models/upscale_a_video \
  /root/autodl-tmp/upscale_a_video \
  /Users/ltl/Desktop/codex_playground/upscale_a_video)}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[uav-x4to1] input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${GT_DIR}" ]]; then
  echo "[uav-x4to1] GT dir not found: ${GT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${GT2_DIR}" ]]; then
  echo "[uav-x4to1] GT images_2 dir not found: ${GT2_DIR}" >&2
  exit 1
fi
if [[ -z "${UAV_REPO}" || ! -d "${UAV_REPO}" ]]; then
  echo "[uav-x4to1] UAV repo not found. Set UAV_REPO=/path/to/Upscale-A-Video" >&2
  exit 1
fi
if [[ -z "${UAV_CKPT_SRC}" || ! -d "${UAV_CKPT_SRC}" ]]; then
  echo "[uav-x4to1] UAV checkpoint folder not found. Set UAV_CKPT_SRC=/path/to/upscale_a_video" >&2
  exit 1
fi

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=1
fi

# Keep UAV in a plain single-process mode so launching another job on the same GPU
# does not run into accidental distributed port collisions.
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
if [[ ! "${MASTER_PORT:-}" =~ ^[0-9]+$ ]]; then
  export MASTER_PORT="$((29500 + ($$ % 1000)))"
fi

TARGET_CKPT_ROOT="${UAV_REPO}/pretrained_models"
TARGET_CKPT_LINK="${TARGET_CKPT_ROOT}/upscale_a_video"
mkdir -p "${TARGET_CKPT_ROOT}"
ln -sfn "${UAV_CKPT_SRC}" "${TARGET_CKPT_LINK}"

echo "[uav-x4to1] HBSR_ROOT=${HBSR_ROOT}"
echo "[uav-x4to1] UAV_REPO=${UAV_REPO}"
echo "[uav-x4to1] UAV_CKPT_SRC=${UAV_CKPT_SRC}"
echo "[uav-x4to1] INPUT_DIR=${INPUT_DIR}"
echo "[uav-x4to1] GT_DIR=${GT_DIR}"
echo "[uav-x4to1] GT2_DIR=${GT2_DIR}"
echo "[uav-x4to1] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[uav-x4to1] CHUNK_SIZE=${CHUNK_SIZE}"
echo "[uav-x4to1] UAV_PERFORM_TILE=${UAV_PERFORM_TILE}"
echo "[uav-x4to1] UAV_TILE_SIZE=${UAV_TILE_SIZE}"
echo "[uav-x4to1] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[uav-x4to1] MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "[uav-x4to1] MASTER_ADDR=${MASTER_ADDR}"
echo "[uav-x4to1] MASTER_PORT=${MASTER_PORT}"

"${PYTHON_EXE}" "${HBSR_ROOT}/hybrid_sdfgs/tools/check_upscale_a_video_checkpoints.py" \
  --root "${TARGET_CKPT_LINK}"

mkdir -p "${OUTPUT_ROOT}"
ln -sfn "${GT_DIR}" "${OUTPUT_ROOT}/ref_images"
ln -sfn "${GT2_DIR}" "${OUTPUT_ROOT}/ref_images_2"
ln -sfn "${INPUT_DIR}" "${OUTPUT_ROOT}/input_images_4"

CHUNKED_SCRIPT="${HBSR_ROOT}/hybrid_sdfgs/exp_scripts/06_generate_video_priors_uav_x8to2_chunked.sh"
if [[ ! -f "${CHUNKED_SCRIPT}" ]]; then
  echo "[uav-x4to1] chunked UAV script not found: ${CHUNKED_SCRIPT}" >&2
  exit 1
fi

HBSR_ROOT="${HBSR_ROOT}" \
UAV_REPO="${UAV_REPO}" \
UAV_CKPT_SRC="${UAV_CKPT_SRC}" \
INPUT_DIR="${INPUT_DIR}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
PYTHON_EXE="${PYTHON_EXE}" \
CHUNK_SIZE="${CHUNK_SIZE}" \
UAV_PERFORM_TILE="${UAV_PERFORM_TILE}" \
UAV_TILE_SIZE="${UAV_TILE_SIZE}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
bash "${CHUNKED_SCRIPT}"

echo "[uav-x4to1] done: ${OUTPUT_ROOT}"
echo "[uav-x4to1] refs:"
echo "  input images_4   : ${OUTPUT_ROOT}/input_images_4"
echo "  GT images        : ${OUTPUT_ROOT}/ref_images"
echo "  GT images_2      : ${OUTPUT_ROOT}/ref_images_2"

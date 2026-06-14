#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HBSR_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_VANILLA_2DGS_ROOT="${HBSR_ROOT}/../2d-gaussian-splatting"
if [[ ! -d "${DEFAULT_VANILLA_2DGS_ROOT}" ]]; then
  DEFAULT_VANILLA_2DGS_ROOT="/root/autodl-tmp/2d-gaussian-splatting"
fi

DEFAULT_SCENE_ROOT="${HBSR_ROOT}/../kitchen"
if [[ ! -d "${DEFAULT_SCENE_ROOT}" ]]; then
  DEFAULT_SCENE_ROOT="/root/autodl-tmp/kitchen"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VANILLA_2DGS_ROOT="${VANILLA_2DGS_ROOT:-${DEFAULT_VANILLA_2DGS_ROOT}}"
SCENE_ROOT="${SCENE_ROOT:-${DEFAULT_SCENE_ROOT}}"
SCENE_NAME="${SCENE_NAME:-$(basename "${SCENE_ROOT}")}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
OUTPUT_PATH="${OUTPUT_PATH:-${HBSR_ROOT}/outputs/vanilla_2dgs_lr_${SCENE_NAME}_${IMAGES_SUBDIR}}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
OMP_THREADS="${OMP_THREADS:-4}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_METRICS="${RUN_METRICS:-0}"

ITERATIONS="${ITERATIONS:-30000}"
RENDER_ITERATION="${RENDER_ITERATION:-${ITERATIONS}}"
RESOLUTION="${RESOLUTION:-1}"
EVAL_SPLIT="${EVAL_SPLIT:-1}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"
QUIET="${QUIET:-0}"
PORT="${PORT:-6019}"

DEPTH_RATIO="${DEPTH_RATIO:-0.0}"
LAMBDA_NORMAL="${LAMBDA_NORMAL:-0.05}"
LAMBDA_DIST="${LAMBDA_DIST:-10.0}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
OPACITY_CULL="${OPACITY_CULL:-0.05}"

TEST_ITERS="${TEST_ITERS:--1}"
SAVE_ITERS="${SAVE_ITERS:-7000 30000}"

SKIP_TRAIN_EXPORT="${SKIP_TRAIN_EXPORT:-1}"
SKIP_TEST_EXPORT="${SKIP_TEST_EXPORT:-0}"
EXPORT_MESH="${EXPORT_MESH:-1}"
UNBOUNDED="${UNBOUNDED:-0}"
MESH_RES="${MESH_RES:-1024}"
VOXEL_SIZE="${VOXEL_SIZE:--1}"
SDF_TRUNC="${SDF_TRUNC:--1}"
DEPTH_TRUNC="${DEPTH_TRUNC:--1}"
NUM_CLUSTER="${NUM_CLUSTER:-50}"

resolve_executable() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
    fi
    return 0
  fi

  command -v "${candidate}" 2>/dev/null || true
}

PYTHON_EXE="$(resolve_executable "${PYTHON_BIN}")"
if [[ -z "${PYTHON_EXE}" && "${PYTHON_BIN}" == "python3" ]]; then
  PYTHON_BIN="python"
  PYTHON_EXE="$(resolve_executable "${PYTHON_BIN}")"
fi

if [[ ! -d "${VANILLA_2DGS_ROOT}" ]]; then
  echo "[vanilla-2dgs-lr] repo not found: ${VANILLA_2DGS_ROOT}" >&2
  exit 1
fi

if [[ -z "${PYTHON_EXE}" ]]; then
  echo "[vanilla-2dgs-lr] python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[vanilla-2dgs-lr] scene root not found: ${SCENE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/${IMAGES_SUBDIR}" ]]; then
  echo "[vanilla-2dgs-lr] image subdir not found: ${SCENE_ROOT}/${IMAGES_SUBDIR}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/sparse/0" ]]; then
  echo "[vanilla-2dgs-lr] COLMAP sparse dir not found: ${SCENE_ROOT}/sparse/0" >&2
  exit 1
fi

mkdir -p "${OUTPUT_PATH}"
cd "${VANILLA_2DGS_ROOT}"

read -r -a TEST_ITERS_ARR <<< "${TEST_ITERS}"
read -r -a SAVE_ITERS_ARR <<< "${SAVE_ITERS}"

COMMON_TRAIN_ARGS=(
  -s "${SCENE_ROOT}"
  -i "${IMAGES_SUBDIR}"
  -m "${OUTPUT_PATH}"
  -r "${RESOLUTION}"
  --iterations "${ITERATIONS}"
  --depth_ratio "${DEPTH_RATIO}"
  --lambda_normal "${LAMBDA_NORMAL}"
  --lambda_dist "${LAMBDA_DIST}"
  --densify_grad_threshold "${DENSIFY_GRAD_THRESHOLD}"
  --opacity_cull "${OPACITY_CULL}"
  --port "${PORT}"
  --test_iterations "${TEST_ITERS_ARR[@]}"
  --save_iterations "${SAVE_ITERS_ARR[@]}"
)

if [[ "${EVAL_SPLIT}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--eval)
fi

if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--white_background)
fi

if [[ "${QUIET}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--quiet)
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  TRAIN_CMD=("${PYTHON_EXE}" train.py "${COMMON_TRAIN_ARGS[@]}")
  echo "[vanilla-2dgs-lr] training:"
  printf '  CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q' "${CUDA_DEVICE}" "${OMP_THREADS}"
  printf ' %q' "${TRAIN_CMD[@]}"
  printf '\n'
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" OMP_NUM_THREADS="${OMP_THREADS}" "${TRAIN_CMD[@]}"
fi

if [[ "${RUN_RENDER}" == "1" ]]; then
  RENDER_CMD=(
    "${PYTHON_EXE}" render.py
    -s "${SCENE_ROOT}"
    -i "${IMAGES_SUBDIR}"
    -m "${OUTPUT_PATH}"
    -r "${RESOLUTION}"
    --iteration "${RENDER_ITERATION}"
    --depth_ratio "${DEPTH_RATIO}"
    --mesh_res "${MESH_RES}"
    --num_cluster "${NUM_CLUSTER}"
  )

  if [[ "${EVAL_SPLIT}" == "1" ]]; then
    RENDER_CMD+=(--eval)
  fi

  if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
    RENDER_CMD+=(--white_background)
  fi

  if [[ "${QUIET}" == "1" ]]; then
    RENDER_CMD+=(--quiet)
  fi

  if [[ "${SKIP_TRAIN_EXPORT}" == "1" ]]; then
    RENDER_CMD+=(--skip_train)
  fi

  if [[ "${SKIP_TEST_EXPORT}" == "1" ]]; then
    RENDER_CMD+=(--skip_test)
  fi

  if [[ "${EXPORT_MESH}" == "0" ]]; then
    RENDER_CMD+=(--skip_mesh)
  fi

  if [[ "${UNBOUNDED}" == "1" ]]; then
    RENDER_CMD+=(--unbounded)
  else
    if [[ "${VOXEL_SIZE}" != "-1" ]]; then
      RENDER_CMD+=(--voxel_size "${VOXEL_SIZE}")
    fi
    if [[ "${SDF_TRUNC}" != "-1" ]]; then
      RENDER_CMD+=(--sdf_trunc "${SDF_TRUNC}")
    fi
    if [[ "${DEPTH_TRUNC}" != "-1" ]]; then
      RENDER_CMD+=(--depth_trunc "${DEPTH_TRUNC}")
    fi
  fi

  echo "[vanilla-2dgs-lr] rendering/meshing:"
  printf '  CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q' "${CUDA_DEVICE}" "${OMP_THREADS}"
  printf ' %q' "${RENDER_CMD[@]}"
  printf '\n'
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" OMP_NUM_THREADS="${OMP_THREADS}" "${RENDER_CMD[@]}"
fi

if [[ "${RUN_METRICS}" == "1" ]]; then
  METRICS_CMD=("${PYTHON_EXE}" metrics.py -m "${OUTPUT_PATH}")
  echo "[vanilla-2dgs-lr] metrics:"
  printf '  %q' "${METRICS_CMD[@]}"
  printf '\n'
  "${METRICS_CMD[@]}"
fi

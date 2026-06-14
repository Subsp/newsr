#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HBSR_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_VANILLA_2DGS_ROOT="/root/autodl-tmp/2d-gaussian-splatting"
DEFAULT_SWINIR_ROOT="/root/autodl-tmp/SwinIR"
DEFAULT_SUGAR_ENV_DIR="${HBSR_ROOT}/.venvs/sugar-system-py"

PYTHON_BIN="${PYTHON_BIN:-}"
VANILLA_2DGS_ROOT="${VANILLA_2DGS_ROOT:-${DEFAULT_VANILLA_2DGS_ROOT}}"
SWINIR_ROOT="${SWINIR_ROOT:-${DEFAULT_SWINIR_ROOT}}"
SUGAR_ENV_DIR="${SUGAR_ENV_DIR:-${DEFAULT_SUGAR_ENV_DIR}}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_NAME="${SCENE_NAME:-$(basename "${SCENE_ROOT}")}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
OUTPUT_PATH="${OUTPUT_PATH:-${HBSR_ROOT}/outputs/vanilla_2dgs_lr_${SCENE_NAME}_${IMAGES_SUBDIR}_swinir_prior}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
OMP_THREADS="${OMP_THREADS:-4}"

RUN_GENERATE_PRIORS="${RUN_GENERATE_PRIORS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_METRICS="${RUN_METRICS:-0}"

ITERATIONS="${ITERATIONS:-30000}"
RENDER_ITERATION="${RENDER_ITERATION:-${ITERATIONS}}"
RESOLUTION="${RESOLUTION:-1}"
EVAL_SPLIT="${EVAL_SPLIT:-1}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"
QUIET="${QUIET:-0}"
PORT="${PORT:-6020}"

DEPTH_RATIO="${DEPTH_RATIO:-0.0}"
LAMBDA_NORMAL="${LAMBDA_NORMAL:-0.05}"
LAMBDA_DIST="${LAMBDA_DIST:-10.0}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
OPACITY_CULL="${OPACITY_CULL:-0.05}"
INIT_POINT_LIMIT="${INIT_POINT_LIMIT:-120000}"

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

PRIOR_ROOT="${PRIOR_ROOT:-/root/autodl-tmp/priors/${SCENE_NAME}_swinir_x8to2_classical}"
PRIOR_SUBDIR="${PRIOR_SUBDIR:-fused_priors}"
PRIOR_MASK_SUBDIR="${PRIOR_MASK_SUBDIR:-usable_masks}"
PRIOR_L1_WEIGHT="${PRIOR_L1_WEIGHT:-0.02}"
PRIOR_HF_WEIGHT="${PRIOR_HF_WEIGHT:-0.01}"
PRIOR_MASK_FLOOR="${PRIOR_MASK_FLOOR:-0.0}"

REFERENCE_DIR="${REFERENCE_DIR:-}"
TASK="${TASK:-classical_sr}"
SCALE="${SCALE:-4}"
TRAINING_PATCH_SIZE="${TRAINING_PATCH_SIZE:-64}"
MODEL_PATH="${MODEL_PATH:-}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.12}"
MASK_MODE="${MASK_MODE:-soft}"
DISCREPANCY_FLOOR="${DISCREPANCY_FLOOR:-0.05}"

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

is_conda_python() {
  local candidate="$1"
  case "${candidate}" in
    *"/conda/"*|*"/miniconda"*|*"/anaconda"*|*"/micromamba"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

discover_python_for_swinir() {
  local candidates=()
  local candidate
  local resolved

  if [[ -n "${PYTHON_BIN}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi

  candidates+=(
    "${SUGAR_ENV_DIR}/bin/python"
    "/usr/bin/python3.10"
    "/usr/local/bin/python3.10"
    "/usr/bin/python3.9"
    "/usr/local/bin/python3.9"
    "/usr/bin/python3"
    "/usr/local/bin/python3"
    "python3"
    "python"
  )

  for candidate in "${candidates[@]}"; do
    resolved="$(resolve_executable "${candidate}")"
    if [[ -z "${resolved}" ]]; then
      continue
    fi
    if [[ "${resolved}" == "${SUGAR_ENV_DIR}/bin/python" ]]; then
      printf '%s\n' "${resolved}"
      return 0
    fi
    if is_conda_python "${resolved}"; then
      continue
    fi
    printf '%s\n' "${resolved}"
    return 0
  done

  return 1
}

PYTHON_EXE="$(discover_python_for_swinir || true)"

if [[ ! -d "${VANILLA_2DGS_ROOT}" ]]; then
  echo "[vanilla-2dgs-swinir] repo not found: ${VANILLA_2DGS_ROOT}" >&2
  exit 1
fi

if [[ -z "${PYTHON_EXE}" ]]; then
  echo "[vanilla-2dgs-swinir] no usable python was found for SwinIR / vanilla 2DGS." >&2
  echo "[vanilla-2dgs-swinir] run 16_install_sugar_system_python.sh first, or pass a system python path via PYTHON_BIN." >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[vanilla-2dgs-swinir] scene root not found: ${SCENE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/${IMAGES_SUBDIR}" ]]; then
  echo "[vanilla-2dgs-swinir] image subdir not found: ${SCENE_ROOT}/${IMAGES_SUBDIR}" >&2
  exit 1
fi

if [[ -z "${REFERENCE_DIR}" ]]; then
  if [[ "${PRIOR_SUBDIR}" == "fused_priors" ]]; then
    echo "[vanilla-2dgs-swinir] REFERENCE_DIR is empty, fallback to raw priors"
    PRIOR_SUBDIR="priors"
  fi
  if [[ -n "${PRIOR_MASK_SUBDIR}" ]]; then
    echo "[vanilla-2dgs-swinir] REFERENCE_DIR is empty, disabling prior mask"
    PRIOR_MASK_SUBDIR=""
  fi
fi

if [[ "${RUN_GENERATE_PRIORS}" == "1" ]]; then
  GEN_CMD=(
    bash "${HBSR_ROOT}/hybrid_sdfgs/exp_scripts/14_generate_swinir_priors_and_masks_x8to2.sh"
  )
  echo "[vanilla-2dgs-swinir] generating priors/masks"
  HBSR_ROOT="${HBSR_ROOT}" \
  SWINIR_ROOT="${SWINIR_ROOT}" \
  PYTHON_EXE="${PYTHON_EXE}" \
  SCENE_ROOT="${SCENE_ROOT}" \
  INPUT_SUBDIR="${IMAGES_SUBDIR}" \
  REFERENCE_DIR="${REFERENCE_DIR}" \
  OUTPUT_ROOT="${PRIOR_ROOT}" \
  TASK="${TASK}" \
  SCALE="${SCALE}" \
  TRAINING_PATCH_SIZE="${TRAINING_PATCH_SIZE}" \
  MODEL_PATH="${MODEL_PATH}" \
  MASK_THRESHOLD="${MASK_THRESHOLD}" \
  MASK_MODE="${MASK_MODE}" \
  DISCREPANCY_FLOOR="${DISCREPANCY_FLOOR}" \
  SAVE_FUSED_PRIORS=1 \
  "${GEN_CMD[@]}"
fi

if [[ ! -d "${PRIOR_ROOT}" ]]; then
  echo "[vanilla-2dgs-swinir] prior root not found: ${PRIOR_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${PRIOR_ROOT}/${PRIOR_SUBDIR}" ]]; then
  if [[ "${PRIOR_SUBDIR}" == "fused_priors" && -d "${PRIOR_ROOT}/priors" ]]; then
    echo "[vanilla-2dgs-swinir] fused priors missing, fallback to raw priors"
    PRIOR_SUBDIR="priors"
  else
    echo "[vanilla-2dgs-swinir] prior subdir not found: ${PRIOR_ROOT}/${PRIOR_SUBDIR}" >&2
    exit 1
  fi
fi

if [[ -n "${PRIOR_MASK_SUBDIR}" && ! -d "${PRIOR_ROOT}/${PRIOR_MASK_SUBDIR}" ]]; then
  echo "[vanilla-2dgs-swinir] prior mask subdir not found, disabling mask: ${PRIOR_ROOT}/${PRIOR_MASK_SUBDIR}"
  PRIOR_MASK_SUBDIR=""
fi

echo "[vanilla-2dgs-swinir] python      : ${PYTHON_EXE}"
echo "[vanilla-2dgs-swinir] prior subdir: ${PRIOR_SUBDIR}"
if [[ -n "${PRIOR_MASK_SUBDIR}" ]]; then
  echo "[vanilla-2dgs-swinir] prior mask  : ${PRIOR_MASK_SUBDIR}"
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
  --init_point_limit "${INIT_POINT_LIMIT}"
  --port "${PORT}"
  --test_iterations "${TEST_ITERS_ARR[@]}"
  --save_iterations "${SAVE_ITERS_ARR[@]}"
  --external_prior_root "${PRIOR_ROOT}"
  --external_prior_subdir "${PRIOR_SUBDIR}"
  --prior_l1_weight "${PRIOR_L1_WEIGHT}"
  --prior_hf_weight "${PRIOR_HF_WEIGHT}"
  --prior_mask_floor "${PRIOR_MASK_FLOOR}"
)

if [[ -n "${PRIOR_MASK_SUBDIR}" ]]; then
  COMMON_TRAIN_ARGS+=(--external_prior_mask_subdir "${PRIOR_MASK_SUBDIR}")
fi

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
  echo "[vanilla-2dgs-swinir] training:"
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

  echo "[vanilla-2dgs-swinir] rendering/meshing:"
  printf '  CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q' "${CUDA_DEVICE}" "${OMP_THREADS}"
  printf ' %q' "${RENDER_CMD[@]}"
  printf '\n'
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" OMP_NUM_THREADS="${OMP_THREADS}" "${RENDER_CMD[@]}"
fi

if [[ "${RUN_METRICS}" == "1" ]]; then
  METRICS_CMD=("${PYTHON_EXE}" metrics.py -m "${OUTPUT_PATH}")
  echo "[vanilla-2dgs-swinir] metrics:"
  printf '  %q' "${METRICS_CMD[@]}"
  printf '\n'
  "${METRICS_CMD[@]}"
fi

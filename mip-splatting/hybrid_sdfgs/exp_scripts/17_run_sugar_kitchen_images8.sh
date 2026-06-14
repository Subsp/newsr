#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HBSR_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SUGAR_ROOT="${HBSR_ROOT}/../SuGaR"
if [[ ! -d "${DEFAULT_SUGAR_ROOT}" ]]; then
  DEFAULT_SUGAR_ROOT="/root/autodl-tmp/SuGaR"
fi

DEFAULT_SCENE_ROOT="${HBSR_ROOT}/../kitchen"
if [[ ! -d "${DEFAULT_SCENE_ROOT}" ]]; then
  DEFAULT_SCENE_ROOT="/root/autodl-tmp/kitchen"
fi

DEFAULT_SUGAR_ENV_DIR="${HBSR_ROOT}/.venvs/sugar-system-py"

PYTHON_BIN="${PYTHON_BIN:-}"

SUGAR_ROOT="${SUGAR_ROOT:-${DEFAULT_SUGAR_ROOT}}"
SCENE_ROOT="${SCENE_ROOT:-${DEFAULT_SCENE_ROOT}}"
SCENE_NAME="${SCENE_NAME:-$(basename "${SCENE_ROOT}")}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_8}"
SCENE_ALIAS_ROOT="${SCENE_ALIAS_ROOT:-${HBSR_ROOT}/outputs/sugar_scene_aliases}"
SCENE_ALIAS_NAME="${SCENE_ALIAS_NAME:-${SCENE_NAME}_${IMAGES_SUBDIR}}"
SCENE_ALIAS_PATH="${SCENE_ALIAS_ROOT}/${SCENE_ALIAS_NAME}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
GPU_INDEX="${GPU_INDEX:-0}"
OMP_THREADS="${OMP_THREADS:-4}"

REGULARIZATION_TYPE="${REGULARIZATION_TYPE:-dn_consistency}"
POLY_MODE="${POLY_MODE:-high}"
REFINEMENT_TIME="${REFINEMENT_TIME:-short}"
SURFACE_LEVEL="${SURFACE_LEVEL:-0.3}"
SQUARE_SIZE="${SQUARE_SIZE:-8}"
EXPORT_OBJ="${EXPORT_OBJ:-1}"
EXPORT_PLY="${EXPORT_PLY:-1}"
POSTPROCESS_MESH="${POSTPROCESS_MESH:-0}"
POSTPROCESS_DENSITY_THRESHOLD="${POSTPROCESS_DENSITY_THRESHOLD:-0.1}"
POSTPROCESS_ITERATIONS="${POSTPROCESS_ITERATIONS:-5}"
EVAL_SPLIT="${EVAL_SPLIT:-1}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"
GS_OUTPUT_DIR="${GS_OUTPUT_DIR:-}"
BBOX_MIN="${BBOX_MIN:-}"
BBOX_MAX="${BBOX_MAX:-}"
CENTER_BBOX="${CENTER_BBOX:-1}"
N_VERTICES_IN_MESH="${N_VERTICES_IN_MESH:-1000000}"
GAUSSIANS_PER_TRIANGLE="${GAUSSIANS_PER_TRIANGLE:-1}"

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

discover_python_for_run() {
  local candidates=()
  local candidate
  local resolved

  if [[ -n "${PYTHON_BIN}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi

  candidates+=(
    "${DEFAULT_SUGAR_ENV_DIR}/bin/python"
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
    if [[ "${resolved}" == "${DEFAULT_SUGAR_ENV_DIR}/bin/python" ]]; then
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

bool_word() {
  case "${1}" in
    1|true|TRUE|True|yes|YES|on|ON)
      printf 'True\n'
      ;;
    0|false|FALSE|False|no|NO|off|OFF)
      printf 'False\n'
      ;;
    *)
      printf '%s\n' "${1}"
      ;;
  esac
}

PYTHON_EXE="$(discover_python_for_run || true)"
if [[ -z "${PYTHON_EXE}" ]]; then
  echo "[sugar-run] no usable python was found for SuGaR." >&2
  echo "[sugar-run] run 16_install_sugar_system_python.sh first, or pass a system python path via PYTHON_BIN." >&2
  exit 1
fi

if [[ ! -d "${SUGAR_ROOT}" ]]; then
  echo "[sugar-run] repo not found: ${SUGAR_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[sugar-run] scene root not found: ${SCENE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/${IMAGES_SUBDIR}" ]]; then
  echo "[sugar-run] image subdir not found: ${SCENE_ROOT}/${IMAGES_SUBDIR}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/sparse/0" ]]; then
  echo "[sugar-run] COLMAP sparse dir not found: ${SCENE_ROOT}/sparse/0" >&2
  exit 1
fi

if [[ -n "${BBOX_MIN}" || -n "${BBOX_MAX}" ]]; then
  if [[ -z "${BBOX_MIN}" || -z "${BBOX_MAX}" ]]; then
    echo "[sugar-run] both BBOX_MIN and BBOX_MAX must be provided together" >&2
    exit 1
  fi
fi

mkdir -p "${SCENE_ALIAS_PATH}"

shopt -s nullglob
for entry in "${SCENE_ROOT}"/*; do
  base_name="$(basename "${entry}")"
  if [[ "${base_name}" == "${IMAGES_SUBDIR}" || "${base_name}" == "images" ]]; then
    continue
  fi
  ln -sfn "${entry}" "${SCENE_ALIAS_PATH}/${base_name}"
done
shopt -u nullglob

ln -sfn "${SCENE_ROOT}/${IMAGES_SUBDIR}" "${SCENE_ALIAS_PATH}/images"

export PATH="$(dirname "${PYTHON_EXE}"):${PATH}"

RUN_CMD=(
  "${PYTHON_EXE}" train_full_pipeline.py
  -s "${SCENE_ALIAS_PATH}"
  -r "${REGULARIZATION_TYPE}"
  -l "${SURFACE_LEVEL}"
  --square_size "${SQUARE_SIZE}"
  --postprocess_mesh "$(bool_word "${POSTPROCESS_MESH}")"
  --postprocess_density_threshold "${POSTPROCESS_DENSITY_THRESHOLD}"
  --postprocess_iterations "${POSTPROCESS_ITERATIONS}"
  --export_obj "$(bool_word "${EXPORT_OBJ}")"
  --export_ply "$(bool_word "${EXPORT_PLY}")"
  --eval "$(bool_word "${EVAL_SPLIT}")"
  --gpu "${GPU_INDEX}"
  --white_background "$(bool_word "${WHITE_BACKGROUND}")"
)

if [[ -n "${REFINEMENT_TIME}" ]]; then
  RUN_CMD+=(--refinement_time "${REFINEMENT_TIME}")
fi

case "${POLY_MODE}" in
  high)
    RUN_CMD+=(--high_poly True --low_poly False)
    ;;
  low)
    RUN_CMD+=(--low_poly True --high_poly False)
    ;;
  custom)
    RUN_CMD+=(
      --high_poly False
      --low_poly False
      -v "${N_VERTICES_IN_MESH}"
      -g "${GAUSSIANS_PER_TRIANGLE}"
    )
    ;;
  *)
    echo "[sugar-run] unsupported POLY_MODE: ${POLY_MODE} (expected high, low, or custom)" >&2
    exit 1
    ;;
esac

if [[ -n "${GS_OUTPUT_DIR}" ]]; then
  RUN_CMD+=(--gs_output_dir "${GS_OUTPUT_DIR}")
fi

if [[ -n "${BBOX_MIN}" ]]; then
  RUN_CMD+=(
    --bboxmin "${BBOX_MIN}"
    --bboxmax "${BBOX_MAX}"
    --center_bbox "$(bool_word "${CENTER_BBOX}")"
  )
fi

echo "[sugar-run] scene root: ${SCENE_ROOT}"
echo "[sugar-run] scene alias: ${SCENE_ALIAS_PATH}"
echo "[sugar-run] sugar root: ${SUGAR_ROOT}"
echo "[sugar-run] python: ${PYTHON_EXE}"
echo "[sugar-run] note: upstream SuGaR outputs are written under ${SUGAR_ROOT}/output"
echo "[sugar-run] launching:"
printf '  CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q' "${CUDA_DEVICE}" "${OMP_THREADS}"
printf ' %q' "${RUN_CMD[@]}"
printf '\n'

cd "${SUGAR_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" OMP_NUM_THREADS="${OMP_THREADS}" "${RUN_CMD[@]}"

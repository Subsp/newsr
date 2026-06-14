#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="${HBSR_ROOT:-/root/autodl-tmp/HBSR}"
SR_BACKEND="${SR_BACKEND:-swinir}"
SWINIR_ROOT="${SWINIR_ROOT:-/root/autodl-tmp/SwinIR}"
HAT_ROOT="${HAT_ROOT:-/root/autodl-tmp/HAT}"
MODEL_REPO_ROOT="${MODEL_REPO_ROOT:-}"
SUGAR_ENV_DIR="${SUGAR_ENV_DIR:-${HBSR_ROOT}/.venvs/sugar-system-py}"
PYTHON_EXE="${PYTHON_EXE:-}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
INPUT_SUBDIR="${INPUT_SUBDIR:-images_8}"
REFERENCE_DIR="${REFERENCE_DIR:-}"
if [[ -z "${MODEL_REPO_ROOT}" ]]; then
  if [[ "${SR_BACKEND}" == "hat" ]]; then
    MODEL_REPO_ROOT="${HAT_ROOT}"
  else
    MODEL_REPO_ROOT="${SWINIR_ROOT}"
  fi
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_${SR_BACKEND}_x8to2_classical}"
TASK="${TASK:-classical_sr}"
SCALE="${SCALE:-4}"
TRAINING_PATCH_SIZE="${TRAINING_PATCH_SIZE:-64}"
MODEL_PATH="${MODEL_PATH:-}"
DEVICE="${DEVICE:-cuda}"
TILE="${TILE:-0}"
TILE_OVERLAP="${TILE_OVERLAP:-32}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.12}"
MASK_MODE="${MASK_MODE:-soft}"
DISCREPANCY_FLOOR="${DISCREPANCY_FLOOR:-0.05}"
SAVE_FUSED_PRIORS="${SAVE_FUSED_PRIORS:-1}"
SAVE_DISCREPANCY_NPZ="${SAVE_DISCREPANCY_NPZ:-0}"

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

  if [[ -n "${PYTHON_EXE}" ]]; then
    resolved="$(resolve_executable "${PYTHON_EXE}")"
    if [[ -n "${resolved}" ]]; then
      printf '%s\n' "${resolved}"
      return 0
    fi
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

PYTHON_EXE_RESOLVED="$(discover_python_for_swinir || true)"

if [[ ! -d "${MODEL_REPO_ROOT}" ]]; then
  echo "[sr-prior-x8to2] model repo not found: ${MODEL_REPO_ROOT}" >&2
  exit 1
fi

if [[ -z "${PYTHON_EXE_RESOLVED}" ]]; then
  echo "[sr-prior-x8to2] no usable python was found for SR prior generation." >&2
  echo "[sr-prior-x8to2] run 16_install_sugar_system_python.sh first, or pass a system python path via PYTHON_EXE." >&2
  exit 1
fi

INPUT_DIR="${SCENE_ROOT}/${INPUT_SUBDIR}"
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[sr-prior-x8to2] input image dir missing: ${INPUT_DIR}" >&2
  exit 1
fi

cd "${HBSR_ROOT}"

CMD=(
  "${PYTHON_EXE_RESOLVED}" "hybrid_sdfgs/tools/generate_swinir_priors.py"
  "--backend" "${SR_BACKEND}"
  "--model_repo_root" "${MODEL_REPO_ROOT}"
  "--folder_lq" "${INPUT_DIR}"
  "--output_root" "${OUTPUT_ROOT}"
  "--task" "${TASK}"
  "--scale" "${SCALE}"
  "--training_patch_size" "${TRAINING_PATCH_SIZE}"
  "--device" "${DEVICE}"
  "--tile" "${TILE}"
  "--tile_overlap" "${TILE_OVERLAP}"
  "--mask_threshold" "${MASK_THRESHOLD}"
  "--mask_mode" "${MASK_MODE}"
  "--discrepancy_floor" "${DISCREPANCY_FLOOR}"
  "--save_raw_priors"
)

if [[ -n "${MODEL_PATH}" ]]; then
  CMD+=("--model_path" "${MODEL_PATH}")
fi

if [[ -n "${REFERENCE_DIR}" ]]; then
  CMD+=("--reference_dir" "${REFERENCE_DIR}")
  if [[ "${SAVE_FUSED_PRIORS}" == "1" ]]; then
    CMD+=("--save_fused_priors")
  fi
  if [[ "${SAVE_DISCREPANCY_NPZ}" == "1" ]]; then
    CMD+=("--save_discrepancy_npz")
  fi
fi

echo "[sr-prior-x8to2] running:"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "[sr-prior-x8to2] backend    : ${SR_BACKEND}"
echo "[sr-prior-x8to2] repo root  : ${MODEL_REPO_ROOT}"
echo "[sr-prior-x8to2] python     : ${PYTHON_EXE_RESOLVED}"
echo "[sr-prior-x8to2] raw priors : ${OUTPUT_ROOT}/priors"
if [[ -n "${REFERENCE_DIR}" ]]; then
  echo "[sr-prior-x8to2] masks      : ${OUTPUT_ROOT}/usable_masks"
  echo "[sr-prior-x8to2] discrepancy: ${OUTPUT_ROOT}/discrepancy"
  echo "[sr-prior-x8to2] aligned ref: ${OUTPUT_ROOT}/aligned_references"
  echo "[sr-prior-x8to2] masked prior: ${OUTPUT_ROOT}/masked_priors"
  echo "[sr-prior-x8to2] masked ref : ${OUTPUT_ROOT}/masked_references"
  if [[ "${SAVE_FUSED_PRIORS}" == "1" ]]; then
    echo "[sr-prior-x8to2] fused priors: ${OUTPUT_ROOT}/fused_priors"
    echo "[sr-prior-x8to2] train with  : --external_prior_root ${OUTPUT_ROOT} --external_prior_subdir fused_priors --external_prior_mask_subdir usable_masks"
  else
    echo "[sr-prior-x8to2] train with  : --external_prior_root ${OUTPUT_ROOT} --external_prior_subdir priors --external_prior_mask_subdir usable_masks"
  fi
else
  echo "[sr-prior-x8to2] train with  : --external_prior_root ${OUTPUT_ROOT} --external_prior_subdir priors"
fi

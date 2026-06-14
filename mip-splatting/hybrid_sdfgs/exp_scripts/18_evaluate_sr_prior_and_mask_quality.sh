#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="${HBSR_ROOT:-/root/autodl-tmp/HBSR}"
SUGAR_ENV_DIR="${SUGAR_ENV_DIR:-${HBSR_ROOT}/.venvs/sugar-system-py}"
PYTHON_EXE="${PYTHON_EXE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/priors/kitchen_hat_x8to2_classical}"
INPUT_DIR="${INPUT_DIR:-/root/autodl-tmp/kitchen/images_8}"
REFERENCE_DIR="${REFERENCE_DIR:-}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_ROOT}/quality_eval}"
PRIOR_SUBDIR="${PRIOR_SUBDIR:-priors}"
MASK_SUBDIR="${MASK_SUBDIR:-usable_masks}"
REFERENCE_SUBDIR="${REFERENCE_SUBDIR:-aligned_references}"
FUSED_SUBDIR="${FUSED_SUBDIR:-fused_priors}"
HARD_MASK_THRESHOLD="${HARD_MASK_THRESHOLD:-0.5}"
ORACLE_MARGIN="${ORACLE_MARGIN:-0.0}"
TOP_K="${TOP_K:-12}"

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

discover_python() {
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

PYTHON_EXE_RESOLVED="$(discover_python || true)"
if [[ -z "${PYTHON_EXE_RESOLVED}" ]]; then
  echo "[prior-mask-eval] no usable python was found." >&2
  exit 1
fi

cd "${HBSR_ROOT}"

CMD=(
  "${PYTHON_EXE_RESOLVED}" "hybrid_sdfgs/tools/evaluate_prior_mask_quality.py"
  "--output_root" "${OUTPUT_ROOT}"
  "--analysis_dir" "${ANALYSIS_DIR}"
  "--input_dir" "${INPUT_DIR}"
  "--prior_subdir" "${PRIOR_SUBDIR}"
  "--mask_subdir" "${MASK_SUBDIR}"
  "--reference_subdir" "${REFERENCE_SUBDIR}"
  "--fused_subdir" "${FUSED_SUBDIR}"
  "--hard_mask_threshold" "${HARD_MASK_THRESHOLD}"
  "--oracle_margin" "${ORACLE_MARGIN}"
  "--top_k" "${TOP_K}"
)

if [[ -n "${REFERENCE_DIR}" ]]; then
  CMD+=("--reference_dir" "${REFERENCE_DIR}")
fi

echo "[prior-mask-eval] running:"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "[prior-mask-eval] analysis_dir: ${ANALYSIS_DIR}"
echo "[prior-mask-eval] summary     : ${ANALYSIS_DIR}/summary.json"
echo "[prior-mask-eval] metrics csv : ${ANALYSIS_DIR}/per_frame_metrics.csv"
echo "[prior-mask-eval] review      : ${ANALYSIS_DIR}/review_panels"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HBSR_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

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

SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_NAME="${SCENE_NAME:-$(basename "${SCENE_ROOT}")}"
INPUT_SUBDIR="${INPUT_SUBDIR:-images_8}"
GT_SUBDIR="${GT_SUBDIR:-images_2}"
REFERENCE_DIR="${REFERENCE_DIR:-${SCENE_ROOT}/${INPUT_SUBDIR}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_EXE="$(resolve_executable "${PYTHON_BIN}")"
if [[ -z "${PYTHON_EXE}" && "${PYTHON_BIN}" == "python3" ]]; then
  PYTHON_EXE="$(resolve_executable python)"
fi

CUDA_DEVICE="${CUDA_DEVICE:-0}"
OMP_THREADS="${OMP_THREADS:-4}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_IE_SRGS="${RUN_IE_SRGS:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

ITERATIONS="${ITERATIONS:-30000}"
RENDER_ITERATION="${RENDER_ITERATION:-${ITERATIONS}}"
RESOLUTION="${RESOLUTION:--1}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"
QUIET="${QUIET:-0}"

SR_BACKEND="${SR_BACKEND:-stablesr}"
PRIOR_SOURCE_MODE="${PRIOR_SOURCE_MODE:-existing}"
EXISTING_PRIOR_DIR="${EXISTING_PRIOR_DIR:-${SCENE_ROOT}/priors}"
SWINIR_ROOT="${SWINIR_ROOT:-/root/autodl-tmp/SwinIR}"
HAT_ROOT="${HAT_ROOT:-/root/autodl-tmp/HAT}"
MODEL_REPO_ROOT="${MODEL_REPO_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-}"
TASK="${TASK:-classical_sr}"
SCALE="${SCALE:-4}"
TRAINING_PATCH_SIZE="${TRAINING_PATCH_SIZE:-64}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.12}"
MASK_MODE="${MASK_MODE:-soft}"
DISCREPANCY_FLOOR="${DISCREPANCY_FLOOR:-0.05}"
FORCE_PREPARE_PRIORS="${FORCE_PREPARE_PRIORS:-0}"
TILE="${TILE:-0}"
TILE_OVERLAP="${TILE_OVERLAP:-32}"

PRIOR_L1_WEIGHT="${PRIOR_L1_WEIGHT:-0.02}"
PRIOR_HF_WEIGHT="${PRIOR_HF_WEIGHT:-0.10}"
PRIOR_MASK_FLOOR="${PRIOR_MASK_FLOOR:-0.0}"
PRIOR_SUBDIR="${PRIOR_SUBDIR:-fused_priors}"
PRIOR_MASK_SUBDIR="${PRIOR_MASK_SUBDIR:-usable_masks}"
PRIOR_CONSISTENCY_THRESHOLD="${PRIOR_CONSISTENCY_THRESHOLD:-0.08}"
PRIOR_MIN_VALID_RATIO="${PRIOR_MIN_VALID_RATIO:-0.80}"
PRIOR_DELTA_CLIP="${PRIOR_DELTA_CLIP:-0.08}"
DISABLE_PRIOR_HF_RESIDUAL="${DISABLE_PRIOR_HF_RESIDUAL:-0}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${HBSR_ROOT}/outputs/ie_srgs_repro/${SCENE_NAME}}"
PRIOR_ROOT="${PRIOR_ROOT:-${EXPERIMENT_ROOT}/priors/${SR_BACKEND}_x8to2}"
BASELINE_OUTPUT_PATH="${BASELINE_OUTPUT_PATH:-${EXPERIMENT_ROOT}/baseline_vanilla_2dgs_lr_${INPUT_SUBDIR}}"
PRIOR_OUTPUT_PATH="${PRIOR_OUTPUT_PATH:-${EXPERIMENT_ROOT}/${SR_BACKEND}_ie_srgs_style_${INPUT_SUBDIR}_to_${GT_SUBDIR}}"
COMPARE_JSON="${COMPARE_JSON:-${EXPERIMENT_ROOT}/compare_${SR_BACKEND}_ie_srgs_style.json}"
PREPARED_PRIOR_MANIFEST="${PRIOR_ROOT}/manifest.json"
PREPARED_PRIOR_MASK_DIR="${PRIOR_ROOT}/${PRIOR_MASK_SUBDIR}"
PREPARED_PRIOR_IMAGE_DIR="${PRIOR_ROOT}/${PRIOR_SUBDIR}"

if [[ ! -d "${HBSR_ROOT}" ]]; then
  echo "[ie-srgs-style] HBSR root not found: ${HBSR_ROOT}" >&2
  exit 1
fi

if [[ -z "${PYTHON_EXE}" ]]; then
  echo "[ie-srgs-style] python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[ie-srgs-style] scene root not found: ${SCENE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/${INPUT_SUBDIR}" ]]; then
  echo "[ie-srgs-style] input subdir missing: ${SCENE_ROOT}/${INPUT_SUBDIR}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/${GT_SUBDIR}" ]]; then
  echo "[ie-srgs-style] gt subdir missing: ${SCENE_ROOT}/${GT_SUBDIR}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_ROOT}/sparse/0" ]]; then
  echo "[ie-srgs-style] sparse/0 missing: ${SCENE_ROOT}/sparse/0" >&2
  exit 1
fi

mkdir -p "${EXPERIMENT_ROOT}"

validate_prepared_priors() {
  (
    cd "${HBSR_ROOT}"
    "${PYTHON_EXE}" -m hybrid_sdfgs.tools.validate_prepared_sr_priors \
      --output_root "${PRIOR_ROOT}" \
      --prior_subdir "${PRIOR_SUBDIR}" \
      --mask_subdir "${PRIOR_MASK_SUBDIR}"
  )
}

render_eval() {
  local model_path="$1"
  echo "[ie-srgs-style] eval render: ${model_path}"
  local render_cmd=(
    "${PYTHON_EXE}" render.py
    -s "${SCENE_ROOT}"
    -i "${GT_SUBDIR}"
    -m "${model_path}"
    --iteration "${RENDER_ITERATION}"
    --resolution "${RESOLUTION}"
    --skip_train
  )
  if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
    render_cmd+=(--white_background)
  fi
  if [[ "${QUIET}" == "1" ]]; then
    render_cmd+=(--quiet)
  fi
  (
    cd "${HBSR_ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" OMP_NUM_THREADS="${OMP_THREADS}" "${render_cmd[@]}"
  )

  echo "[ie-srgs-style] eval metrics: ${model_path}"
  local metrics_cmd=(
    "${PYTHON_EXE}" metrics.py
    -m "${model_path}"
    -r "${RESOLUTION}"
  )
  (
    cd "${HBSR_ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" OMP_NUM_THREADS="${OMP_THREADS}" "${metrics_cmd[@]}"
  )
}

run_mip_baseline_train() {
  local train_cmd=(
    "${PYTHON_EXE}" train.py
    -s "${SCENE_ROOT}"
    -i "${INPUT_SUBDIR}"
    -m "${BASELINE_OUTPUT_PATH}"
    -r "${RESOLUTION}"
    --iterations "${ITERATIONS}"
    --eval
  )

  if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
    train_cmd+=(--white_background)
  fi
  if [[ "${QUIET}" == "1" ]]; then
    train_cmd+=(--quiet)
  fi

  echo "[ie-srgs-style] mip baseline train:"
  printf '  CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q' "${CUDA_DEVICE}" "${OMP_THREADS}"
  printf ' %q' "${train_cmd[@]}"
  printf '\n'
  (
    cd "${HBSR_ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" OMP_NUM_THREADS="${OMP_THREADS}" "${train_cmd[@]}"
  )
}

run_hybrid_prior_train() {
  local prior_root="$1"
  local train_cmd=(
    "${PYTHON_EXE}" -m hybrid_sdfgs.train
    -s "${SCENE_ROOT}"
    -i "${INPUT_SUBDIR}"
    -m "${PRIOR_OUTPUT_PATH}"
    -r "${RESOLUTION}"
    --iterations "${ITERATIONS}"
    --eval
    --external_prior_root "${prior_root}"
    --external_prior_subdir "${PRIOR_SUBDIR}"
    --external_prior_mask_subdir "${PRIOR_MASK_SUBDIR}"
    --prior_l1_weight "${PRIOR_L1_WEIGHT}"
    --prior_hf_weight "${PRIOR_HF_WEIGHT}"
    --prior_mask_floor "${PRIOR_MASK_FLOOR}"
    --prior_consistency_threshold "${PRIOR_CONSISTENCY_THRESHOLD}"
    --prior_min_valid_ratio "${PRIOR_MIN_VALID_RATIO}"
    --prior_delta_clip "${PRIOR_DELTA_CLIP}"
  )

  if [[ "${DISABLE_PRIOR_HF_RESIDUAL}" == "1" ]]; then
    train_cmd+=(--disable_prior_hf_residual)
  fi
  if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
    train_cmd+=(--white_background)
  fi
  if [[ "${QUIET}" == "1" ]]; then
    train_cmd+=(--quiet)
  fi

  echo "[ie-srgs-style] hybrid prior train:"
  printf '  CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q' "${CUDA_DEVICE}" "${OMP_THREADS}"
  printf ' %q' "${train_cmd[@]}"
  printf '\n'
  (
    cd "${HBSR_ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" OMP_NUM_THREADS="${OMP_THREADS}" "${train_cmd[@]}"
  )
}

echo "[ie-srgs-style] scene              : ${SCENE_NAME}"
echo "[ie-srgs-style] scene root         : ${SCENE_ROOT}"
echo "[ie-srgs-style] input subdir       : ${INPUT_SUBDIR}"
echo "[ie-srgs-style] gt subdir          : ${GT_SUBDIR}"
echo "[ie-srgs-style] reference dir      : ${REFERENCE_DIR}"
echo "[ie-srgs-style] prior root         : ${PRIOR_ROOT}"
echo "[ie-srgs-style] prior source mode  : ${PRIOR_SOURCE_MODE}"
if [[ "${PRIOR_SOURCE_MODE}" == "existing" ]]; then
  echo "[ie-srgs-style] existing priors    : ${EXISTING_PRIOR_DIR}"
fi
echo "[ie-srgs-style] baseline output    : ${BASELINE_OUTPUT_PATH}"
echo "[ie-srgs-style] IE-SRGS output     : ${PRIOR_OUTPUT_PATH}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "[1/4] train mip-splatting baseline"
  run_mip_baseline_train
fi

if [[ "${RUN_IE_SRGS}" == "1" ]]; then
  echo "[2/4] train IE-SRGS-style prior run"
  if [[ "${PRIOR_SOURCE_MODE}" == "existing" ]]; then
    if [[ -z "${EXISTING_PRIOR_DIR}" ]]; then
      echo "[ie-srgs-style] EXISTING_PRIOR_DIR is required when PRIOR_SOURCE_MODE=existing" >&2
      exit 1
    fi
    if [[ ! -d "${EXISTING_PRIOR_DIR}" ]]; then
      echo "[ie-srgs-style] existing prior dir not found: ${EXISTING_PRIOR_DIR}" >&2
      exit 1
    fi
    if [[ "${FORCE_PREPARE_PRIORS}" != "1" && -f "${PREPARED_PRIOR_MANIFEST}" && -d "${PREPARED_PRIOR_MASK_DIR}" && -d "${PREPARED_PRIOR_IMAGE_DIR}" ]] && validate_prepared_priors >/dev/null; then
      echo "[ie-srgs-style] prepared priors valid, skipping: ${PRIOR_ROOT}"
    else
      if [[ "${FORCE_PREPARE_PRIORS}" != "1" && -f "${PREPARED_PRIOR_MANIFEST}" ]]; then
        echo "[ie-srgs-style] prepared priors invalid or incomplete, rebuilding: ${PRIOR_ROOT}"
      fi
      (
        cd "${HBSR_ROOT}"
        "${PYTHON_EXE}" -m hybrid_sdfgs.tools.prepare_existing_sr_priors \
          --prior_dir "${EXISTING_PRIOR_DIR}" \
          --reference_dir "${REFERENCE_DIR}" \
          --output_root "${PRIOR_ROOT}" \
          --mask_threshold "${MASK_THRESHOLD}" \
          --mask_mode "${MASK_MODE}" \
          --discrepancy_floor "${DISCREPANCY_FLOOR}" \
          --copy_raw_priors \
          --save_fused_priors
      )
    fi
    run_hybrid_prior_train "${PRIOR_ROOT}"
  else
    if [[ "${SR_BACKEND}" != "swinir" && "${SR_BACKEND}" != "hat" ]]; then
      echo "[ie-srgs-style] PRIOR_SOURCE_MODE=generate only supports SR_BACKEND=swinir or hat, got ${SR_BACKEND}" >&2
      exit 1
    fi
    HBSR_ROOT="${HBSR_ROOT}" \
    SWINIR_ROOT="${SWINIR_ROOT}" \
    HAT_ROOT="${HAT_ROOT}" \
    SR_BACKEND="${SR_BACKEND}" \
    MODEL_REPO_ROOT="${MODEL_REPO_ROOT}" \
    MODEL_PATH="${MODEL_PATH}" \
    PYTHON_EXE="${PYTHON_EXE}" \
    SCENE_ROOT="${SCENE_ROOT}" \
    INPUT_SUBDIR="${INPUT_SUBDIR}" \
    REFERENCE_DIR="${REFERENCE_DIR}" \
    OUTPUT_ROOT="${PRIOR_ROOT}" \
    TASK="${TASK}" \
    SCALE="${SCALE}" \
    TRAINING_PATCH_SIZE="${TRAINING_PATCH_SIZE}" \
    MASK_THRESHOLD="${MASK_THRESHOLD}" \
    MASK_MODE="${MASK_MODE}" \
    DISCREPANCY_FLOOR="${DISCREPANCY_FLOOR}" \
    TILE="${TILE}" \
    TILE_OVERLAP="${TILE_OVERLAP}" \
    SAVE_FUSED_PRIORS=1 \
    bash "${SCRIPT_DIR}/14_generate_swinir_priors_and_masks_x8to2.sh"
    run_hybrid_prior_train "${PRIOR_ROOT}"
  fi
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[3/4] evaluate baseline and IE-SRGS-style runs on ${GT_SUBDIR}"
  render_eval "${BASELINE_OUTPUT_PATH}"
  render_eval "${PRIOR_OUTPUT_PATH}"
fi

if [[ "${RUN_COMPARE}" == "1" ]]; then
  echo "[4/4] compare baseline vs IE-SRGS-style"
  (
    cd "${HBSR_ROOT}"
    "${PYTHON_EXE}" -m hybrid_sdfgs.tools.compare_eval_results \
      --baseline_model "${BASELINE_OUTPUT_PATH}" \
      --current_model "${PRIOR_OUTPUT_PATH}" \
      --output_json "${COMPARE_JSON}"
  )
fi

echo "[ie-srgs-style] done"
echo "  baseline : ${BASELINE_OUTPUT_PATH}"
echo "  current  : ${PRIOR_OUTPUT_PATH}"
echo "  compare  : ${COMPARE_JSON}"

#!/usr/bin/env bash
set -euo pipefail

# Supported enhancement-SR wrapper around the prior-from-scratch mainline.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MIPSPLATTING_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)/mip-splatting"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-images_8}"
REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_2}"
FALLBACK_IMAGES_SUBDIR="${FALLBACK_IMAGES_SUBDIR:-${REFERENCE_IMAGES_SUBDIR}}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-${REFERENCE_IMAGES_SUBDIR}}"
PREPARE_IMAGES8="${PREPARE_IMAGES8:-1}"
GENERATE_IMAGES8_SCALE="${GENERATE_IMAGES8_SCALE:-4}"
RESIZE_FILTER="${RESIZE_FILTER:-bicubic}"

ENHANCEMENT_BACKEND="${ENHANCEMENT_BACKEND:-swinir}"
RAW_PRIOR_SUBDIR="${RAW_PRIOR_SUBDIR:-priors_${ENHANCEMENT_BACKEND}}"
RAW_PRIOR_DIR="${RAW_PRIOR_DIR:-${SCENE_ROOT}/${RAW_PRIOR_SUBDIR}}"

PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-${ENHANCEMENT_BACKEND}_aligned_${REFERENCE_IMAGES_SUBDIR}_scratch_v0}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}}"
SR_PRIOR_SUBDIR="${SR_PRIOR_SUBDIR:-fused_priors}"
SR_PRIOR_MASK_SUBDIR="${SR_PRIOR_MASK_SUBDIR:-usable_masks}"
SR_ANCHOR_SUBDIR="${SR_ANCHOR_SUBDIR:-aligned_references}"

MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${DEFAULT_MIPSPLATTING_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"
PRIOR_ONLY_RUN_TAG="${PRIOR_ONLY_RUN_TAG:-mip30k_r1_${ENHANCEMENT_BACKEND}_prioronly_scratch_v0}"

ITERATIONS="${ITERATIONS:-30000}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
EXTERNAL_RESTORATION_ROOT="${EXTERNAL_RESTORATION_ROOT:-}"
EXTERNAL_RESTORATION_PYTHON="${EXTERNAL_RESTORATION_PYTHON:-${PYTHON_BIN}}"
EXTERNAL_RESTORATION_CONFIG="${EXTERNAL_RESTORATION_CONFIG:-}"
RESTORMER_TASK="${RESTORMER_TASK:-Single_Image_Defocus_Deblurring}"
RESTORMER_TILE="${RESTORMER_TILE:-0}"
RESTORMER_TILE_OVERLAP="${RESTORMER_TILE_OVERLAP:-32}"
FORCE_GENERATE_RAW_PRIORS="${FORCE_GENERATE_RAW_PRIORS:-0}"
FORCE_PREPARE_SR_PRIORS="${FORCE_PREPARE_SR_PRIORS:-0}"
ALLOW_PREPARED_SR_PRIOR_REBUILD="${ALLOW_PREPARED_SR_PRIOR_REBUILD:-1}"

MASK_THRESHOLD="${MASK_THRESHOLD:-0.12}"
MASK_MODE="${MASK_MODE:-soft}"
DISCREPANCY_FLOOR="${DISCREPANCY_FLOOR:-0.05}"
DISABLE_PRIOR_USABLE_MASKS="${DISABLE_PRIOR_USABLE_MASKS:-0}"
PRIOR_MATCH_POLICY="${PRIOR_MATCH_POLICY:-stem}"

RUN_NOSR_AFTER="${RUN_NOSR_AFTER:-0}"
NOSR_CLEANUP_ITERS="${NOSR_CLEANUP_ITERS:-2000}"
NOSR_LOWFREQ_ANCHOR_MODE="${NOSR_LOWFREQ_ANCHOR_MODE:-images8}"
NOSR_HF_RETENTION_PROFILE="${NOSR_HF_RETENTION_PROFILE:-preserve_v1}"

RAW_PRIOR_MANIFEST="${RAW_PRIOR_DIR}/manifest.json"
PREPARED_PRIOR_MANIFEST="${PREPARED_SR_PRIOR_ROOT}/manifest.json"
PREPARED_PRIOR_IMAGE_DIR="${PREPARED_SR_PRIOR_ROOT}/${SR_PRIOR_SUBDIR}"
PREPARED_PRIOR_MASK_DIR="${PREPARED_SR_PRIOR_ROOT}/${SR_PRIOR_MASK_SUBDIR}"
PREPARED_PRIOR_ANCHOR_DIR="${PREPARED_SR_PRIOR_ROOT}/${SR_ANCHOR_SUBDIR}"
PREPARED_REFERENCE_DIR="${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}"
PRIOR_ONLY_MODEL_DIR="${OUTPUT_ROOT}/mipsplatting_prior_only_from_scratch_v0/${SCENE_NAME}/${PRIOR_ONLY_RUN_TAG}"

MIP_PYTHONPATH="${MIPSPLATTING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=1
fi

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

validate_prepared_priors() {
  (
    cd "${MIPSPLATTING_ROOT}"
    export PYTHONPATH="${MIP_PYTHONPATH}"
    VALIDATE_ARGS=(
      "${PYTHON_BIN}" -m hybrid_sdfgs.tools.validate_prepared_sr_priors
      --output_root "${PREPARED_SR_PRIOR_ROOT}"
      --prior_subdir "${SR_PRIOR_SUBDIR}"
    )
    if [[ "${DISABLE_PRIOR_USABLE_MASKS}" != "1" ]]; then
      VALIDATE_ARGS+=(--mask_subdir "${SR_PRIOR_MASK_SUBDIR}")
    else
      VALIDATE_ARGS+=(--mask_subdir "")
    fi
    "${VALIDATE_ARGS[@]}"
  )
}

validate_prepared_prior_reference_dir() {
  (
    export PREPARED_PRIOR_MANIFEST="${PREPARED_PRIOR_MANIFEST}"
    export EXPECTED_REFERENCE_DIR="${PREPARED_REFERENCE_DIR}"
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["PREPARED_PRIOR_MANIFEST"]).resolve()
expected_reference = Path(os.environ["EXPECTED_REFERENCE_DIR"]).resolve()
if not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
reference_dir = manifest.get("reference_dir")
if not reference_dir:
    raise SystemExit(1)
if Path(reference_dir).resolve() != expected_reference:
    raise SystemExit(1)
PY
  )
}

for path in \
  "${SCENE_ROOT}" \
  "${SCENE_ROOT}/sparse/0" \
  "${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}" \
  "${SCENE_ROOT}/${FALLBACK_IMAGES_SUBDIR}" \
  "${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}" \
  "${MIPSPLATTING_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[enhance-prior-scratch-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}" ]]; then
  if [[ "${PREPARE_IMAGES8}" != "1" ]]; then
    echo "[enhance-prior-scratch-v0] missing ${SOURCE_IMAGES_SUBDIR} and PREPARE_IMAGES8=0: ${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}" >&2
    exit 1
  fi
  echo "[enhance-prior-scratch-v0] generate ${SOURCE_IMAGES_SUBDIR} from ${TARGET_IMAGES_SUBDIR}"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/generate_downsampled_images.py \
      --source_dir "${SCENE_ROOT}/${TARGET_IMAGES_SUBDIR}" \
      --output_dir "${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}" \
      --scale "${GENERATE_IMAGES8_SCALE}" \
      --resize_filter "${RESIZE_FILTER}"
  )
fi

mkdir -p "${SCENE_ASSET_ROOT}" "${OUTPUT_ROOT}"

echo "[enhance-prior-scratch-v0] scene                : ${SCENE_NAME}"
echo "[enhance-prior-scratch-v0] scene root           : ${SCENE_ROOT}"
echo "[enhance-prior-scratch-v0] enhancement backend  : ${ENHANCEMENT_BACKEND}"
echo "[enhance-prior-scratch-v0] source images        : ${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}"
echo "[enhance-prior-scratch-v0] raw prior dir        : ${RAW_PRIOR_DIR}"
echo "[enhance-prior-scratch-v0] prepared prior root  : ${PREPARED_SR_PRIOR_ROOT}"
echo "[enhance-prior-scratch-v0] external root        : ${EXTERNAL_RESTORATION_ROOT:-<backend env/default>}"
echo "[enhance-prior-scratch-v0] prior match policy  : ${PRIOR_MATCH_POLICY}"
echo "[enhance-prior-scratch-v0] scratch run tag      : ${PRIOR_ONLY_RUN_TAG}"
echo "[enhance-prior-scratch-v0] iterations           : ${ITERATIONS}"
echo "[enhance-prior-scratch-v0] run nosr after       : ${RUN_NOSR_AFTER}"

echo
echo "[1/4] generate raw enhancement SR priors"
raw_input_count=$(find "${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}" -maxdepth 1 -type f | wc -l | tr -d ' ')
raw_prior_count=0
if [[ -d "${RAW_PRIOR_DIR}" ]]; then
  raw_prior_count=$(find "${RAW_PRIOR_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
fi
if [[ "${FORCE_GENERATE_RAW_PRIORS}" == "1" || ! -f "${RAW_PRIOR_MANIFEST}" || "${raw_prior_count}" -lt "${raw_input_count}" ]]; then
  CMD=(
    "${PYTHON_BIN}" "${SOF_ROOT}/scripts/generate_enhancement_sr_priors.py"
    --input_dir "${SCENE_ROOT}/${SOURCE_IMAGES_SUBDIR}"
    --output_dir "${RAW_PRIOR_DIR}"
    --backend "${ENHANCEMENT_BACKEND}"
    --device "${DEVICE}"
    --external_python "${EXTERNAL_RESTORATION_PYTHON}"
    --restormer_task "${RESTORMER_TASK}"
    --restormer_tile "${RESTORMER_TILE}"
    --restormer_tile_overlap "${RESTORMER_TILE_OVERLAP}"
  )
  if [[ -n "${EXTERNAL_RESTORATION_ROOT}" ]]; then
    CMD+=(--external_repo_root "${EXTERNAL_RESTORATION_ROOT}")
  fi
  if [[ -n "${EXTERNAL_RESTORATION_CONFIG}" ]]; then
    CMD+=(--external_config "${EXTERNAL_RESTORATION_CONFIG}")
  fi
  if [[ "${FORCE_GENERATE_RAW_PRIORS}" == "1" ]]; then
    CMD+=(--overwrite)
  fi
  "${CMD[@]}"
else
  echo "[enhance-prior-scratch-v0] reuse raw priors: ${RAW_PRIOR_DIR}"
fi

echo
echo "[2/4] prepare aligned SR prior cache"
PREPARED_READY=0
PREPARED_INVALID_REASON=""
if [[ "${FORCE_PREPARE_SR_PRIORS}" == "1" ]]; then
  PREPARED_INVALID_REASON="forced rebuild requested"
elif [[ ! -f "${PREPARED_PRIOR_MANIFEST}" ]]; then
  PREPARED_INVALID_REASON="missing manifest"
elif [[ ! -d "${PREPARED_PRIOR_IMAGE_DIR}" ]]; then
  PREPARED_INVALID_REASON="missing fused priors"
elif [[ "${DISABLE_PRIOR_USABLE_MASKS}" != "1" && ! -d "${PREPARED_PRIOR_MASK_DIR}" ]]; then
  PREPARED_INVALID_REASON="missing usable masks"
elif [[ "${DISABLE_PRIOR_USABLE_MASKS}" != "1" && ! -d "${PREPARED_PRIOR_ANCHOR_DIR}" ]]; then
  PREPARED_INVALID_REASON="missing aligned references"
elif ! PREPARED_VALIDATE_OUTPUT="$(validate_prepared_priors 2>&1)"; then
  PREPARED_INVALID_REASON="validation failed: ${PREPARED_VALIDATE_OUTPUT}"
elif ! validate_prepared_prior_reference_dir; then
  PREPARED_INVALID_REASON="reference_dir mismatch vs ${PREPARED_REFERENCE_DIR}"
else
  PREPARED_READY=1
  echo "[enhance-prior-scratch-v0] prepared priors valid, reusing: ${PREPARED_SR_PRIOR_ROOT}"
fi

if [[ "${PREPARED_READY}" != "1" ]]; then
  echo "[enhance-prior-scratch-v0] prepared priors not ready: ${PREPARED_INVALID_REASON}" >&2
  if [[ "${ALLOW_PREPARED_SR_PRIOR_REBUILD}" != "1" ]]; then
    echo "[enhance-prior-scratch-v0] refusing to rebuild prepared priors. Set ALLOW_PREPARED_SR_PRIOR_REBUILD=1 or FORCE_PREPARE_SR_PRIORS=1." >&2
    exit 1
  fi
  (
    cd "${MIPSPLATTING_ROOT}"
    export PYTHONPATH="${MIP_PYTHONPATH}"
    PREPARE_ARGS=(
      "${PYTHON_BIN}" -m hybrid_sdfgs.tools.prepare_existing_sr_priors
      --prior_dir "${RAW_PRIOR_DIR}"
      --reference_dir "${PREPARED_REFERENCE_DIR}"
      --output_root "${PREPARED_SR_PRIOR_ROOT}"
      --mask_threshold "${MASK_THRESHOLD}"
      --mask_mode "${MASK_MODE}"
      --discrepancy_floor "${DISCREPANCY_FLOOR}"
      --match_policy "${PRIOR_MATCH_POLICY}"
      --copy_raw_priors
      --save_fused_priors
    )
    if [[ "${DISABLE_PRIOR_USABLE_MASKS}" == "1" ]]; then
      PREPARE_ARGS+=(--disable_usable_masks)
    fi
    "${PREPARE_ARGS[@]}"
  )
  validate_prepared_priors
  validate_prepared_prior_reference_dir
else
  validate_prepared_prior_reference_dir
fi

echo
echo "[3/4] train scratch prior-only mip-splatting model"
(
  cd "${SOF_ROOT}"
  PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT}" \
  PRIOR_IMAGE_SUBDIR="${SR_PRIOR_SUBDIR}" \
  REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR}" \
  FALLBACK_IMAGES_SUBDIR="${FALLBACK_IMAGES_SUBDIR}" \
  TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR}" \
  MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_TAG="${PRIOR_ONLY_RUN_TAG}" \
  ITERATIONS="${ITERATIONS}" \
  RENDER_RESOLUTION="${RENDER_RESOLUTION}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
  CUDA_HOME="${CUDA_HOME}" \
  bash scripts/run_mipsplatting_prior_only_from_scratch_v0_kitchen.sh
)

echo
if [[ "${RUN_NOSR_AFTER}" == "1" ]]; then
  echo "[4/4] run NoSR cleanup on the enhancement-prior scratch model"
  (
    cd "${SOF_ROOT}"
    PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT}" \
    PRIOR_IMAGE_SUBDIR="${SR_PRIOR_SUBDIR}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    INPUT_RUN_TAG="${PRIOR_ONLY_RUN_TAG}" \
    INPUT_MODEL_DIR="${PRIOR_ONLY_MODEL_DIR}" \
    INPUT_ITERATION="${ITERATIONS}" \
    CLEANUP_ITERS="${NOSR_CLEANUP_ITERS}" \
    LOWFREQ_ANCHOR_MODE="${NOSR_LOWFREQ_ANCHOR_MODE}" \
    HF_RETENTION_PROFILE="${NOSR_HF_RETENTION_PROFILE}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen_srprior_lrmesh.sh
  )
else
  echo "[4/4] skip NoSR cleanup (RUN_NOSR_AFTER=${RUN_NOSR_AFTER})"
fi

echo
echo "[done] raw priors      : ${RAW_PRIOR_DIR}"
echo "[done] prepared priors : ${PREPARED_SR_PRIOR_ROOT}"
echo "[done] scratch model   : ${PRIOR_ONLY_MODEL_DIR}"

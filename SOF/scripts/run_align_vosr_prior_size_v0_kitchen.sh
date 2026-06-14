#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

PRIOR_DIR="${PRIOR_DIR:-${WORK_ROOT}/test_preds_1_vosr_same/qwen_steps1_seed42_rcgm}"
REFERENCE_DIR="${REFERENCE_DIR:-${SCENE_ROOT}/images_2}"
OUTPUT_NAME="${OUTPUT_NAME:-qwen_steps1_seed42_rcgm_aligned_images2_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${OUTPUT_NAME}}"
REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-$(basename "${REFERENCE_DIR}")}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-auto}"
LLFFHOLD="${LLFFHOLD:-8}"

ANCHOR="${ANCHOR:-center}"
FILL_MODE="${FILL_MODE:-reference}"
MATCH_MODE="${MATCH_MODE:-auto}"
MAX_DELTA="${MAX_DELTA:-64}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "${PRIOR_DIR}" ]]; then
  echo "[align-vosr-prior-v0] missing PRIOR_DIR=${PRIOR_DIR}" >&2
  exit 1
fi
if [[ ! -d "${REFERENCE_DIR}" ]]; then
  echo "[align-vosr-prior-v0] missing REFERENCE_DIR=${REFERENCE_DIR}" >&2
  echo "[align-vosr-prior-v0] set REFERENCE_DIR to the image grid the priors should align to." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${ALLOW_LARGE_DELTA:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_large_delta)
fi
if [[ "${ALLOW_MISSING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_missing)
fi
if [[ "${ORDER_FALLBACK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--order_fallback)
fi
if [[ -n "${MATCH_MODE}" ]]; then
  EXTRA_ARGS+=(--match_mode "${MATCH_MODE}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi
if [[ -n "${X_OFFSET:-}" ]]; then
  EXTRA_ARGS+=(--x_offset "${X_OFFSET}")
fi
if [[ -n "${Y_OFFSET:-}" ]]; then
  EXTRA_ARGS+=(--y_offset "${Y_OFFSET}")
fi
if [[ -n "${START_PRIOR_NAME:-}" ]]; then
  EXTRA_ARGS+=(--start_prior_name "${START_PRIOR_NAME}")
fi
if [[ -n "${START_REFERENCE_NAME:-}" ]]; then
  EXTRA_ARGS+=(--start_reference_name "${START_REFERENCE_NAME}")
fi

echo "[align-vosr-prior-v0] prior dir    : ${PRIOR_DIR}"
echo "[align-vosr-prior-v0] reference    : ${REFERENCE_DIR}"
echo "[align-vosr-prior-v0] output root  : ${OUTPUT_ROOT}"
echo "[align-vosr-prior-v0] anchor/fill  : ${ANCHOR}/${FILL_MODE}"
echo "[align-vosr-prior-v0] match mode   : ${MATCH_MODE}"
echo "[align-vosr-prior-v0] ref split    : ${REFERENCE_SPLIT} (${REFERENCE_IMAGES_SUBDIR}, llffhold=${LLFFHOLD})"
if [[ -n "${START_PRIOR_NAME:-}" || -n "${START_REFERENCE_NAME:-}" ]]; then
  echo "[align-vosr-prior-v0] resume from  : prior=${START_PRIOR_NAME:-<none>} reference=${START_REFERENCE_NAME:-<none>}"
fi
echo "[align-vosr-prior-v0] max delta    : ${MAX_DELTA}"

"${PYTHON_BIN}" "${SOF_ROOT}/scripts/align_vosr_prior_size_v0.py" \
  --prior_dir "${PRIOR_DIR}" \
  --reference_dir "${REFERENCE_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --scene_root "${SCENE_ROOT}" \
  --images_subdir "${REFERENCE_IMAGES_SUBDIR}" \
  --reference_split "${REFERENCE_SPLIT}" \
  --llffhold "${LLFFHOLD}" \
  --anchor "${ANCHOR}" \
  --fill_mode "${FILL_MODE}" \
  --max_delta "${MAX_DELTA}" \
  --progress_every "${PROGRESS_EVERY}" \
  "${EXTRA_ARGS[@]}"

echo "[align-vosr-prior-v0] prepared root: ${OUTPUT_ROOT}"
echo "[align-vosr-prior-v0] use SR_PRIOR_ROOT=${OUTPUT_ROOT}, SR_PRIOR_SUBDIR=fused_priors, SR_PRIOR_MASK_SUBDIR=usable_masks"

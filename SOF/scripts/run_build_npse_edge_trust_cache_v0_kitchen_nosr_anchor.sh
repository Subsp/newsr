#!/usr/bin/env bash
set -euo pipefail

# Build N-PSE cache using a NoSR-cleaned GS model render as the low-frequency
# anchor. This tests whether NoSR's non-surface HF suppression gives cleaner
# trust/residual/continuous-region cues than the vanilla directsrc render.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output}"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${REPO_ROOT}/mip-splatting}"
PYTHON_BIN="${PYTHON_BIN:-python}"

REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_2}"
ENHANCEMENT_BACKEND="${ENHANCEMENT_BACKEND:-restormer}"

NOSR_RUN_TAG="${NOSR_RUN_TAG:-mip30k_rerun_check_directsrc_r1_v0_to32000_trainr4_nosr28_layerfreq_cleanup_v0}"
NOSR_MODEL_DIR="${NOSR_MODEL_DIR:-${OUTPUT_ROOT}/mipsplatting_nosr_layerfreq_cleanup_v0/${SCENE_NAME}/${NOSR_RUN_TAG}}"
NOSR_ITERATION="${NOSR_ITERATION:-32000}"
ANCHOR_RENDER_RESOLUTION="${ANCHOR_RENDER_RESOLUTION:-1}"
ANCHOR_DIR="${ANCHOR_DIR:-${NOSR_MODEL_DIR}/train/ours_${NOSR_ITERATION}/test_preds_${ANCHOR_RENDER_RESOLUTION}}"

RENDER_NOSR_TRAIN_IF_MISSING="${RENDER_NOSR_TRAIN_IF_MISSING:-1}"
FORCE_RENDER_NOSR_TRAIN="${FORCE_RENDER_NOSR_TRAIN:-0}"

DEPTH_PRIOR_DIR="${DEPTH_PRIOR_DIR:?Set DEPTH_PRIOR_DIR to the mesh-aligned depth prior directory.}"
OUTPUT_NAME="${OUTPUT_NAME:-render_x1_${ENHANCEMENT_BACKEND}_depthprior_npse_nosranchor_srconfirmed_v0}"

for path in "${SCENE_ROOT}" "${SCENE_ROOT}/sparse/0" "${SCENE_ROOT}/${REFERENCE_IMAGES_SUBDIR}" "${MIPSPLATTING_ROOT}" "${NOSR_MODEL_DIR}" "${DEPTH_PRIOR_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[npse-nosr-anchor-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ ! -f "${NOSR_MODEL_DIR}/point_cloud/iteration_${NOSR_ITERATION}/point_cloud.ply" ]]; then
  echo "[npse-nosr-anchor-v0] missing NoSR point cloud: ${NOSR_MODEL_DIR}/point_cloud/iteration_${NOSR_ITERATION}/point_cloud.ply" >&2
  exit 1
fi

echo "[npse-nosr-anchor-v0] scene        : ${SCENE_ROOT}"
echo "[npse-nosr-anchor-v0] NoSR model   : ${NOSR_MODEL_DIR}"
echo "[npse-nosr-anchor-v0] NoSR iter    : ${NOSR_ITERATION}"
echo "[npse-nosr-anchor-v0] anchor dir   : ${ANCHOR_DIR}"
echo "[npse-nosr-anchor-v0] depth prior  : ${DEPTH_PRIOR_DIR}"
echo "[npse-nosr-anchor-v0] output name  : ${OUTPUT_NAME}"

if [[ "${FORCE_RENDER_NOSR_TRAIN}" == "1" || ! -d "${ANCHOR_DIR}" ]]; then
  if [[ "${RENDER_NOSR_TRAIN_IF_MISSING}" != "1" && "${FORCE_RENDER_NOSR_TRAIN}" != "1" ]]; then
    echo "[npse-nosr-anchor-v0] missing anchor dir and RENDER_NOSR_TRAIN_IF_MISSING=0: ${ANCHOR_DIR}" >&2
    exit 1
  fi
  echo "[npse-nosr-anchor-v0] render NoSR train views"
  (
    cd "${MIPSPLATTING_ROOT}"
    export PYTHONPATH="${MIPSPLATTING_ROOT}:${PYTHONPATH:-}"
    "${PYTHON_BIN}" render.py \
      -m "${NOSR_MODEL_DIR}" \
      -s "${SCENE_ROOT}" \
      -i "${REFERENCE_IMAGES_SUBDIR}" \
      -r "${ANCHOR_RENDER_RESOLUTION}" \
      --iteration "${NOSR_ITERATION}" \
      --skip_test
  )
else
  echo "[npse-nosr-anchor-v0] reuse NoSR train render: ${ANCHOR_DIR}"
fi

if [[ ! -d "${ANCHOR_DIR}" ]]; then
  echo "[npse-nosr-anchor-v0] anchor render was not created: ${ANCHOR_DIR}" >&2
  exit 1
fi

ANCHOR_DIR="${ANCHOR_DIR}" \
DEPTH_PRIOR_DIR="${DEPTH_PRIOR_DIR}" \
OUTPUT_NAME="${OUTPUT_NAME}" \
bash "${SCRIPT_DIR}/run_build_npse_edge_trust_cache_v0_kitchen.sh"

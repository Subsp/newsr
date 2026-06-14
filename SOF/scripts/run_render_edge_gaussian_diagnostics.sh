#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_eval}"

MODEL_PATH="${MODEL_PATH:-}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
ITERATION="${ITERATION:--1}"
SPLIT="${SPLIT:-test}"
RENDER_INDEX="${RENDER_INDEX:-0}"
ACTION_PAYLOAD="${ACTION_PAYLOAD:-}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

VISIBILITY_DOWNSAMPLE="${VISIBILITY_DOWNSAMPLE:-8}"
VISIBILITY_TOPK="${VISIBILITY_TOPK:-4}"
VISIBILITY_MAX_VISIBLE="${VISIBILITY_MAX_VISIBLE:-30000}"
VISIBILITY_MAX_PATCH_RADIUS="${VISIBILITY_MAX_PATCH_RADIUS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -z "${CHECKPOINT_TAG}" ]]; then
    echo "[edge-diagnostics] provide MODEL_PATH or CHECKPOINT_TAG" >&2
    exit 1
  fi
  MODEL_PATH="${OUTPUT_ROOT}/${SCENE_NAME}_joint_${CHECKPOINT_TAG}"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ -n "${CHECKPOINT_TAG}" ]]; then
    OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE_NAME}_edge_diag_${CHECKPOINT_TAG}_${SPLIT}_${RENDER_INDEX}"
  else
    OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE_NAME}_edge_diag_${SPLIT}_${RENDER_INDEX}"
  fi
fi

for path in "${SCENE_ROOT}" "${MODEL_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[edge-diagnostics] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[edge-diagnostics] scene root            : ${SCENE_ROOT}"
echo "[edge-diagnostics] model path            : ${MODEL_PATH}"
echo "[edge-diagnostics] images subdir         : ${IMAGES_SUBDIR}"
echo "[edge-diagnostics] split/index           : ${SPLIT} / ${RENDER_INDEX}"
echo "[edge-diagnostics] iteration             : ${ITERATION}"
echo "[edge-diagnostics] output dir            : ${OUTPUT_DIR}"
if [[ -n "${ACTION_PAYLOAD}" ]]; then
  echo "[edge-diagnostics] action payload        : ${ACTION_PAYLOAD}"
fi
echo "[edge-diagnostics] visibility downsample : ${VISIBILITY_DOWNSAMPLE}"

CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/render_hrgs_edge_diagnostics.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --images_subdir "${IMAGES_SUBDIR}"
  --iteration "${ITERATION}"
  --split "${SPLIT}"
  --render_index "${RENDER_INDEX}"
  --output_dir "${OUTPUT_DIR}"
  --visibility_downsample "${VISIBILITY_DOWNSAMPLE}"
  --visibility_topk "${VISIBILITY_TOPK}"
  --visibility_max_visible "${VISIBILITY_MAX_VISIBLE}"
  --visibility_max_patch_radius "${VISIBILITY_MAX_PATCH_RADIUS}"
)

if [[ -n "${ACTION_PAYLOAD}" ]]; then
  CMD+=(--action_payload "${ACTION_PAYLOAD}")
fi

(
  cd "${SOF_ROOT}"
  "${CMD[@]}"
)

echo
echo "[done] diagnostics : ${OUTPUT_DIR}"

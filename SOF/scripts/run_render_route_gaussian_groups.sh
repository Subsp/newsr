#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/hrgs_eval}"

MIP_EXPERIMENT_GROUP="${MIP_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_EXPERIMENT_NAME="${MIP_EXPERIMENT_NAME:-mip30k}"
DETAIL_MODEL_PATH="${DETAIL_MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_EXPERIMENT_GROUP}/${MIP_EXPERIMENT_NAME}_sof_native_input}"

CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
ROUTE_PAYLOAD="${ROUTE_PAYLOAD:-}"
RENDER_OUTPUT_DIR="${RENDER_OUTPUT_DIR:-}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"
ITERATION="${ITERATION:-30000}"
SPLIT="${SPLIT:-test}"
ROUTE_GROUPS="${ROUTE_GROUPS:-full,attach,detail,suppress}"
GROUP_MODE="${GROUP_MODE:-threshold}"
THRESHOLD="${THRESHOLD:-0.2}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${ROUTE_PAYLOAD}" ]]; then
  if [[ -z "${CHECKPOINT_TAG}" ]]; then
    echo "[render-route-groups] provide ROUTE_PAYLOAD or CHECKPOINT_TAG" >&2
    exit 1
  fi
  ROUTE_PAYLOAD="${OUTPUT_ROOT}/${SCENE_NAME}_${CHECKPOINT_TAG}/route_payload_v0.pt"
fi

if [[ -z "${RENDER_OUTPUT_DIR}" ]]; then
  if [[ -z "${CHECKPOINT_TAG}" ]]; then
    CHECKPOINT_TAG="manual"
  fi
  RENDER_OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE_NAME}_route_groups_${CHECKPOINT_TAG}"
fi

for path in "${SCENE_ROOT}" "${DETAIL_MODEL_PATH}" "${ROUTE_PAYLOAD}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[render-route-groups] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[render-route-groups] scene root       : ${SCENE_ROOT}"
echo "[render-route-groups] detail model     : ${DETAIL_MODEL_PATH}"
echo "[render-route-groups] route payload    : ${ROUTE_PAYLOAD}"
echo "[render-route-groups] output dir       : ${RENDER_OUTPUT_DIR}"
echo "[render-route-groups] images subdir    : ${IMAGES_SUBDIR}"
echo "[render-route-groups] split            : ${SPLIT}"
echo "[render-route-groups] groups           : ${ROUTE_GROUPS}"
echo "[render-route-groups] group mode       : ${GROUP_MODE}"
echo "[render-route-groups] iteration        : ${ITERATION}"

CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/render_route_gaussian_groups.py"
  --scene_root "${SCENE_ROOT}"
  --detail_model_path "${DETAIL_MODEL_PATH}"
  --route_payload "${ROUTE_PAYLOAD}"
  --output_dir "${RENDER_OUTPUT_DIR}"
  --images_subdir "${IMAGES_SUBDIR}"
  --iteration "${ITERATION}"
  --split "${SPLIT}"
  --groups "${ROUTE_GROUPS}"
  --group_mode "${GROUP_MODE}"
  --threshold "${THRESHOLD}"
)

if [[ "${WHITE_BACKGROUND}" == "1" ]]; then
  CMD+=(--white_background)
fi

"${CMD[@]}"

echo
echo "[done] route group renders: ${RENDER_OUTPUT_DIR}"

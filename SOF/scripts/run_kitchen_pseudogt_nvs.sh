#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WORKSPACE_ROOT}/experiments/sof/kitchen_pseudogt}"

DEFAULT_SCENE_ROOT="${WORKSPACE_ROOT}/kitchen"
if [[ ! -d "${DEFAULT_SCENE_ROOT}" ]]; then
  DEFAULT_SCENE_ROOT="${HOME}/Downloads/kitchen"
fi

SCENE_ROOT="${SCENE_ROOT:-${DEFAULT_SCENE_ROOT}}"
PSEUDO_SCENE_DIR="${PSEUDO_SCENE_DIR:-${EXPERIMENT_ROOT}/alias}"
MODEL_DIR="${MODEL_DIR:-${EXPERIMENT_ROOT}/model}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREPARE_TOOL="${PREPARE_TOOL:-${SOF_ROOT}/scripts/prepare_colmap_pseudo_sr_scene.py}"
ITERATIONS="${ITERATIONS:-30000}"
SPLATTING_CONFIG="${SPLATTING_CONFIG:-configs/hierarchical.json}"

PREPARE_ONLY="${PREPARE_ONLY:-0}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_RENDER="${RUN_RENDER:-0}"
RUN_METRICS="${RUN_METRICS:-0}"

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[sof-kitchen-pseudogt] scene root not found: ${SCENE_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${PREPARE_TOOL}" ]]; then
  echo "[sof-kitchen-pseudogt] pseudo-scene tool not found: ${PREPARE_TOOL}" >&2
  exit 1
fi

mkdir -p "${PSEUDO_SCENE_DIR}"
mkdir -p "${MODEL_DIR}"

echo "[sof-kitchen-pseudogt] step 1/4: prepare pseudo scene"
"${PYTHON_BIN}" "${PREPARE_TOOL}" \
  --scene_root "${SCENE_ROOT}" \
  --scene_alias_dir "${PSEUDO_SCENE_DIR}" \
  --source_images_subdir images_8 \
  --target_images_subdir images_2 \
  --resize_filter bicubic

TRAIN_CMD=(
  python train.py
  --splatting_config "${SPLATTING_CONFIG}"
  -s "${PSEUDO_SCENE_DIR}"
  --eval
  -m "${MODEL_DIR}"
  --iterations "${ITERATIONS}"
)

RENDER_CMD=(
  python render.py
  -m "${MODEL_DIR}"
  -s "${SCENE_ROOT}"
  -i images_2
  --eval
  --skip_train
  --data_device cpu
)

METRICS_CMD=(
  python metrics.py
  -m "${MODEL_DIR}"
)

echo "[sof-kitchen-pseudogt] pseudo scene ready:"
echo "  scene_root      : ${SCENE_ROOT}"
echo "  experiment_root : ${EXPERIMENT_ROOT}"
echo "  pseudo_scene    : ${PSEUDO_SCENE_DIR}"
echo "  model_dir       : ${MODEL_DIR}"
echo "  summary         : ${PSEUDO_SCENE_DIR}/pseudo_sr_summary.json"
echo
echo "[sof-kitchen-pseudogt] next commands:"
printf '  (cd %q &&' "${SOF_ROOT}"
printf ' %q' "${TRAIN_CMD[@]}"
printf ')\n'
printf '  (cd %q &&' "${SOF_ROOT}"
printf ' %q' "${RENDER_CMD[@]}"
printf ')\n'
printf '  (cd %q &&' "${SOF_ROOT}"
printf ' %q' "${METRICS_CMD[@]}"
printf ')\n'

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  exit 0
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[sof-kitchen-pseudogt] step 2/4: train"
  (
    cd "${SOF_ROOT}"
    "${TRAIN_CMD[@]}"
  )
fi

if [[ "${RUN_RENDER}" == "1" ]]; then
  echo "[sof-kitchen-pseudogt] step 3/4: render against images_2"
  (
    cd "${SOF_ROOT}"
    "${RENDER_CMD[@]}"
  )
fi

if [[ "${RUN_METRICS}" == "1" ]]; then
  echo "[sof-kitchen-pseudogt] step 4/4: metrics"
  (
    cd "${SOF_ROOT}"
    "${METRICS_CMD[@]}"
  )
fi

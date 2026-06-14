#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"

SOF_MODEL_PATH="${SOF_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images2_v1/sof30k}"
SOF_ITERATION="${SOF_ITERATION:-30000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/oracle/formal_oracle_sof30k}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-1e-4}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"

DEPTH_DIR="${OUTPUT_ROOT}/depth"
META_PATH="${OUTPUT_ROOT}/formal_oracle_sof_meta.json"

mkdir -p "${OUTPUT_ROOT}"

for path in "${SCENE_ROOT}" "${SCENE_ROOT}/${IMAGES_SUBDIR}" "${SCENE_ROOT}/sparse/0" "${SOF_MODEL_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[prepare-sof-oracle] required path not found: ${path}" >&2
    exit 1
  fi
done

echo "[prepare-sof-oracle] scene root      : ${SCENE_ROOT}"
echo "[prepare-sof-oracle] asset root      : ${SCENE_ASSET_ROOT}"
echo "[prepare-sof-oracle] images subdir   : ${IMAGES_SUBDIR}"
echo "[prepare-sof-oracle] sof model path  : ${SOF_MODEL_PATH}"
echo "[prepare-sof-oracle] sof iteration   : ${SOF_ITERATION}"
echo "[prepare-sof-oracle] output root     : ${OUTPUT_ROOT}"
echo "[prepare-sof-oracle] alpha threshold : ${ALPHA_THRESHOLD}"

DEPTH_COUNT=0
if compgen -G "${DEPTH_DIR}/*.npy" > /dev/null; then
  DEPTH_COUNT="$(find "${DEPTH_DIR}" -maxdepth 1 -name '*.npy' | wc -l | tr -d ' ')"
fi

if [[ "${FORCE_EXPORT}" == "1" || "${DEPTH_COUNT}" == "0" ]]; then
  echo
  echo "[1/2] export SOF oracle buffers"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" scripts/export_sof_oracle_buffers.py \
      --scene_root "${SCENE_ROOT}" \
      --model_path "${SOF_MODEL_PATH}" \
      --output_root "${OUTPUT_ROOT}" \
      --images_subdir "${IMAGES_SUBDIR}" \
      --iteration "${SOF_ITERATION}" \
      --alpha_threshold "${ALPHA_THRESHOLD}" \
      --save_preview_json
  )
else
  echo
  echo "[1/2] reuse existing SOF oracle buffers (${DEPTH_COUNT} npy files)"
fi

echo
echo "[2/2] summarize oracle depth coverage"
(
  cd "${SOF_ROOT}"
  "${PYTHON_BIN}" scripts/summarize_oracle_depth_dir.py \
    --oracle_root "${OUTPUT_ROOT}" \
    --scene_root "${SCENE_ROOT}" \
    --images_subdir "${IMAGES_SUBDIR}" \
    --output_dir "${OUTPUT_ROOT}"
)

echo
echo "[done] formal oracle root : ${OUTPUT_ROOT}"
echo "[done] trainer can use     : ORACLE_ROOT=${OUTPUT_ROOT}"
echo "[done] meta               : ${META_PATH}"

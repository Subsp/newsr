#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

LR_SOF_MODEL="${LR_SOF_MODEL:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images8_v1/soflr30k}"
HR_SOF_MODEL="${HR_SOF_MODEL:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_sof_vanilla_images2_v1/sof30k}"
LR_IMAGES_SUBDIR="${LR_IMAGES_SUBDIR:-images_8}"
HR_IMAGES_SUBDIR="${HR_IMAGES_SUBDIR:-images_2}"
ITERATION="${ITERATION:-30000}"

OUT_ROOT="${OUT_ROOT:-${SOF_ROOT}/output/sof_mesh_patch_enhancer_v0/${SCENE_NAME}}"
RUN_NAME="${RUN_NAME:-soflr_to_sofhr_geom_v0}"
RUN_ROOT="${RUN_ROOT:-${OUT_ROOT}/${RUN_NAME}}"
DATASET_PATH="${DATASET_PATH:-${RUN_ROOT}/sof_mesh_patch_dataset_v0.pt}"
TRAIN_OUT="${TRAIN_OUT:-${RUN_ROOT}/train}"
APPLY_OUT="${APPLY_OUT:-${RUN_ROOT}/apply}"

MESH_NAME_LR="${MESH_NAME_LR:-lr_sof_mesh_v0}"
MESH_NAME_HR="${MESH_NAME_HR:-hr_sof_mesh_v0}"
LR_MESH_PATH="${LR_MESH_PATH:-${LR_SOF_MODEL}/test/ours_${ITERATION}/${MESH_NAME_LR}_7.ply}"
HR_MESH_PATH="${HR_MESH_PATH:-${HR_SOF_MODEL}/test/ours_${ITERATION}/${MESH_NAME_HR}_7.ply}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

REFERENCE_SAMPLES="${REFERENCE_SAMPLES:-300000}"
STRONG_RATIO="${STRONG_RATIO:-0.002}"
WEAK_RATIO="${WEAK_RATIO:-0.006}"
MAX_VERTICES="${MAX_VERTICES:-0}"
VERTEX_STRIDE="${VERTEX_STRIDE:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
DEVICE="${DEVICE:-cuda}"
CARRIERS_PER_FACE="${CARRIERS_PER_FACE:-1}"
CARRIER_MIN_CONFIDENCE="${CARRIER_MIN_CONFIDENCE:-0.05}"

mkdir -p "${RUN_ROOT}" "${TRAIN_OUT}" "${APPLY_OUT}"

echo "[sof-mesh-patch-v0] scene        : ${SCENE_ROOT}"
echo "[sof-mesh-patch-v0] LR SOF model : ${LR_SOF_MODEL}"
echo "[sof-mesh-patch-v0] HR SOF model : ${HR_SOF_MODEL}"
echo "[sof-mesh-patch-v0] run root     : ${RUN_ROOT}"

cd "${SOF_ROOT}"

if [[ "${SKIP_EXTRACT}" != "1" ]]; then
  echo "[sof-mesh-patch-v0] extract LR mesh"
  python -u extract_mesh_tets.py \
    -s "${SCENE_ROOT}" \
    -m "${LR_SOF_MODEL}" \
    -i "${LR_IMAGES_SUBDIR}" \
    --iteration "${ITERATION}" \
    --mesh_name "${MESH_NAME_LR}" \
    --eval \
    --data_device cpu

  echo "[sof-mesh-patch-v0] extract HR mesh"
  python -u extract_mesh_tets.py \
    -s "${SCENE_ROOT}" \
    -m "${HR_SOF_MODEL}" \
    -i "${HR_IMAGES_SUBDIR}" \
    --iteration "${ITERATION}" \
    --mesh_name "${MESH_NAME_HR}" \
    --eval \
    --data_device cpu
fi

if [[ ! -f "${LR_MESH_PATH}" ]]; then
  echo "[sof-mesh-patch-v0] missing LR mesh: ${LR_MESH_PATH}" >&2
  exit 1
fi
if [[ ! -f "${HR_MESH_PATH}" ]]; then
  echo "[sof-mesh-patch-v0] missing HR mesh: ${HR_MESH_PATH}" >&2
  exit 1
fi

echo "[sof-mesh-patch-v0] build dataset"
python -u build_sof_mesh_patch_dataset_v0.py \
  --lr_mesh_path "${LR_MESH_PATH}" \
  --hr_mesh_path "${HR_MESH_PATH}" \
  --output_path "${DATASET_PATH}" \
  --reference_samples "${REFERENCE_SAMPLES}" \
  --strong_ratio "${STRONG_RATIO}" \
  --weak_ratio "${WEAK_RATIO}" \
  --vertex_stride "${VERTEX_STRIDE}" \
  --max_vertices "${MAX_VERTICES}"

echo "[sof-mesh-patch-v0] train enhancer"
python -u train_sof_mesh_patch_enhancer_v0.py \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${TRAIN_OUT}" \
  --steps "${TRAIN_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}"

CKPT="${TRAIN_OUT}/sof_mesh_patch_enhancer_step_$(printf "%06d" "${TRAIN_STEPS}").pt"

echo "[sof-mesh-patch-v0] apply enhancer"
python -u apply_sof_mesh_patch_enhancer_v0.py \
  --lr_mesh_path "${LR_MESH_PATH}" \
  --checkpoint_path "${CKPT}" \
  --output_dir "${APPLY_OUT}" \
  --device "${DEVICE}" \
  --carriers_per_face "${CARRIERS_PER_FACE}" \
  --carrier_min_confidence "${CARRIER_MIN_CONFIDENCE}"

echo "[done] LR mesh         : ${LR_MESH_PATH}"
echo "[done] HR mesh         : ${HR_MESH_PATH}"
echo "[done] dataset         : ${DATASET_PATH}"
echo "[done] checkpoint      : ${CKPT}"
echo "[done] SR mesh         : ${APPLY_OUT}/sr_mesh_patch_enhanced_v0.ply"
echo "[done] carrier payload : ${APPLY_OUT}/sr_mesh_patch_carrier_payload_v0.npz"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RAW_MIP_MODEL_PATH="${RAW_MIP_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k}"
RAW_MIP_ITERATION="${RAW_MIP_ITERATION:-30000}"
PREPARE_DEBUG_MODEL_PATH="${PREPARE_DEBUG_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k_sof_native_input_init_early4ksoft_v1_debug}"
PREPARE_DEBUG_ITERATION="${PREPARE_DEBUG_ITERATION:-34000}"
DEBUG_STAGE_ROOT="${DEBUG_STAGE_ROOT:-${PREPARE_DEBUG_MODEL_PATH}/debug_prepare_stages}"

# Yesterday's diagnostic read: starburst is already visible at stage 00;
# stage 00b3 is where the large block artifacts begin after scale canonicalization.
STAGE_NAMES="${STAGE_NAMES:-debug_stage_00_after_finite_aabb debug_stage_00b3_after_scale_canonicalize}"

MESH_IMAGES_SUBDIR="${MESH_IMAGES_SUBDIR:-images_8}"
RAW_MESH_NAME="${RAW_MESH_NAME:-raw_mip_sof_export_mesh_v0}"
STAGE_MESH_NAME_PREFIX="${STAGE_MESH_NAME_PREFIX:-prepare_stage_sof_export_mesh_v0}"
RUN_RAW_MIP="${RUN_RAW_MIP:-1}"
RUN_STAGES="${RUN_STAGES:-1}"
FILTER_MESH="${FILTER_MESH:-1}"
TEXTURE_MESH="${TEXTURE_MESH:-0}"
EXTRA_EXTRACT_ARGS="${EXTRA_EXTRACT_ARGS:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}}"
COPY_MESHES_TO_OUTPUT="${COPY_MESHES_TO_OUTPUT:-1}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/sof_mesh_prepare_stage_compare_v0_summary.txt}"

mkdir -p "${OUTPUT_ROOT}"

echo "[sof-mesh-stage-compare-v0] scene              : ${SCENE_ROOT}"
echo "[sof-mesh-stage-compare-v0] raw mip model      : ${RAW_MIP_MODEL_PATH} iter=${RAW_MIP_ITERATION}"
echo "[sof-mesh-stage-compare-v0] prepare debug model: ${PREPARE_DEBUG_MODEL_PATH} iter=${PREPARE_DEBUG_ITERATION}"
echo "[sof-mesh-stage-compare-v0] debug stage root   : ${DEBUG_STAGE_ROOT}"
echo "[sof-mesh-stage-compare-v0] stage names        : ${STAGE_NAMES}"
echo "[sof-mesh-stage-compare-v0] mesh images subdir : ${MESH_IMAGES_SUBDIR}"
echo "[sof-mesh-stage-compare-v0] output root        : ${OUTPUT_ROOT}"

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[sof-mesh-stage-compare-v0] missing scene root: ${SCENE_ROOT}" >&2
  exit 1
fi

mesh_flag_args=()
if [[ "${FILTER_MESH}" == "1" ]]; then
  mesh_flag_args+=(--filter_mesh)
fi
if [[ "${TEXTURE_MESH}" == "1" ]]; then
  mesh_flag_args+=(--texture_mesh)
fi
if [[ -n "${EXTRA_EXTRACT_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_EXTRACT_ARGS} )
else
  extra_args=()
fi

: > "${SUMMARY_PATH}"

run_extract() {
  local label="$1"
  local model_path="$2"
  local iteration="$3"
  local mesh_name="$4"
  local output_copy_name="$5"

  if [[ ! -f "${model_path}/point_cloud/iteration_${iteration}/point_cloud.ply" ]]; then
    echo "[sof-mesh-stage-compare-v0] missing point cloud for ${label}: ${model_path}/point_cloud/iteration_${iteration}/point_cloud.ply" >&2
    return 1
  fi

  echo
  echo "[sof-mesh-stage-compare-v0] extract ${label}"
  (
    cd "${SOF_ROOT}"
    "${PYTHON_BIN}" -u extract_mesh_tets.py \
      -s "${SCENE_ROOT}" \
      -m "${model_path}" \
      -i "${MESH_IMAGES_SUBDIR}" \
      --iteration "${iteration}" \
      --mesh_name "${mesh_name}" \
      --eval \
      --data_device cpu \
      "${mesh_flag_args[@]}" \
      "${extra_args[@]}"
  )

  local mesh_path="${model_path}/test/ours_${iteration}/${mesh_name}_7.ply"
  if [[ ! -f "${mesh_path}" ]]; then
    echo "[sof-mesh-stage-compare-v0] expected mesh missing for ${label}: ${mesh_path}" >&2
    return 1
  fi

  echo "[mesh] ${label}: ${mesh_path}" | tee -a "${SUMMARY_PATH}"
  if [[ "${COPY_MESHES_TO_OUTPUT}" == "1" ]]; then
    cp "${mesh_path}" "${OUTPUT_ROOT}/${output_copy_name}"
    echo "[copy] ${label}: ${OUTPUT_ROOT}/${output_copy_name}" | tee -a "${SUMMARY_PATH}"
  fi
}

if [[ "${RUN_RAW_MIP}" == "1" ]]; then
  run_extract "raw_mip" "${RAW_MIP_MODEL_PATH}" "${RAW_MIP_ITERATION}" "${RAW_MESH_NAME}" "raw_mip_${RAW_MESH_NAME}_7.ply"
fi

if [[ "${RUN_STAGES}" == "1" ]]; then
  if [[ ! -d "${DEBUG_STAGE_ROOT}" ]]; then
    echo "[sof-mesh-stage-compare-v0] missing debug stage root: ${DEBUG_STAGE_ROOT}" >&2
    echo "[sof-mesh-stage-compare-v0] rerun prepare with DEBUG_DUMP_PREPARE_STAGES=1 first." >&2
    exit 1
  fi

  normalized_stage_names="${STAGE_NAMES//,/ }"
  for stage_name in ${normalized_stage_names}; do
    stage_model_path="${DEBUG_STAGE_ROOT}/${stage_name}"
    stage_mesh_name="${STAGE_MESH_NAME_PREFIX}_${stage_name}"
    run_extract "${stage_name}" "${stage_model_path}" "${PREPARE_DEBUG_ITERATION}" "${stage_mesh_name}" "${stage_name}_${stage_mesh_name}_7.ply"
  done
fi

echo
echo "[done] summary: ${SUMMARY_PATH}"
cat "${SUMMARY_PATH}"

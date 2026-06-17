#!/usr/bin/env bash
set -euo pipefail

# Align an external depth prior to the gs2mesh/COLMAP camera depth domain.
# The aligned_depth/*.npz output can be passed to N-PSE as DEPTH_PRIOR_DIR.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SOF_ROOT}/.." && pwd)"
MIPSPLATTING_ROOT="${MIPSPLATTING_ROOT:-${REPO_ROOT}/mip-splatting}"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_2}"
MIP_RENDER_EXPERIMENT_GROUP="${MIP_RENDER_EXPERIMENT_GROUP:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_RENDER_MODEL_NAME="${MIP_RENDER_MODEL_NAME:-mip30k_rerun_check_directsrc_r1_v0}"
MODEL_PATH="${MODEL_PATH:-${SCENE_ASSET_ROOT}/${MIP_RENDER_EXPERIMENT_GROUP}/${MIP_RENDER_MODEL_NAME}}"

MESH_PATH="${MESH_PATH:?Set MESH_PATH to the gs2mesh mesh .ply path.}"
DEPTH_PRIOR_DIR="${DEPTH_PRIOR_DIR:?Set DEPTH_PRIOR_DIR to the raw external depth prior directory.}"

SPLIT="${SPLIT:-train}"
PRIOR_LLFFHOLD="${PRIOR_LLFFHOLD:-8}"
LIMIT="${LIMIT:-0}"
OUTPUT_NAME="${OUTPUT_NAME:-render_x1_depthprior_${REFERENCE_IMAGES_SUBDIR}_${SPLIT}_gs2mesh_aligned_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCENE_ASSET_ROOT}/depth_prior_aligned_gs2mesh/${OUTPUT_NAME}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MESH_DEPTH_MODE="${MESH_DEPTH_MODE:-raycast}"
RAYCAST_DOWNSAMPLE="${RAYCAST_DOWNSAMPLE:-1}"
RAYCAST_CHUNK="${RAYCAST_CHUNK:-262144}"
DEPTH_SUBDIRS="${DEPTH_SUBDIRS:-auto}"
DEPTH_KEYS="${DEPTH_KEYS:-auto}"
ALIGNMENT_MODES="${ALIGNMENT_MODES:-affine,inverse_affine,negative_affine}"
ALIGN_MIN_PIXELS="${ALIGN_MIN_PIXELS:-2048}"
DEPTH_MIN="${DEPTH_MIN:-0.02}"
RECURSIVE_DEPTH_SEARCH="${RECURSIVE_DEPTH_SEARCH:-0}"
SAVE_DEBUG_IMAGES="${SAVE_DEBUG_IMAGES:-1}"
OVERWRITE="${OVERWRITE:-0}"

for path in "${SCENE_ROOT}" "${MODEL_PATH}" "${DEPTH_PRIOR_DIR}" "${MIPSPLATTING_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[mesh-align-depth-v0] required path not found: ${path}" >&2
    exit 1
  fi
done
if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[mesh-align-depth-v0] mesh file not found: ${MESH_PATH}" >&2
  exit 1
fi

echo "[mesh-align-depth-v0] scene      : ${SCENE_ROOT}"
echo "[mesh-align-depth-v0] model      : ${MODEL_PATH}"
echo "[mesh-align-depth-v0] mesh       : ${MESH_PATH}"
echo "[mesh-align-depth-v0] depth prior: ${DEPTH_PRIOR_DIR}"
echo "[mesh-align-depth-v0] output     : ${OUTPUT_ROOT}"

CMD=(
  "${PYTHON_BIN}" "${SOF_ROOT}/scripts/build_mesh_aligned_depth_prior_cache_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --mesh_path "${MESH_PATH}"
  --depth_prior_dir "${DEPTH_PRIOR_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --images_subdir "${REFERENCE_IMAGES_SUBDIR}"
  --split "${SPLIT}"
  --llffhold "${PRIOR_LLFFHOLD}"
  --limit "${LIMIT}"
  --mesh_depth_mode "${MESH_DEPTH_MODE}"
  --raycast_downsample "${RAYCAST_DOWNSAMPLE}"
  --raycast_chunk "${RAYCAST_CHUNK}"
  --depth_subdirs "${DEPTH_SUBDIRS}"
  --depth_keys "${DEPTH_KEYS}"
  --alignment_modes "${ALIGNMENT_MODES}"
  --align_min_pixels "${ALIGN_MIN_PIXELS}"
  --depth_min "${DEPTH_MIN}"
)

if [[ "${RECURSIVE_DEPTH_SEARCH}" == "1" ]]; then
  CMD+=(--recursive_depth_search)
fi
if [[ "${SAVE_DEBUG_IMAGES}" == "1" ]]; then
  CMD+=(--save_debug_images)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

PYTHONPATH="${MIPSPLATTING_ROOT}:${PYTHONPATH:-}" "${CMD[@]}"

echo "[mesh-align-depth-v0] aligned depth dir:"
echo "${OUTPUT_ROOT}/aligned_depth"
echo "[mesh-align-depth-v0] next N-PSE command can set:"
echo "DEPTH_PRIOR_DIR=${OUTPUT_ROOT}/aligned_depth"

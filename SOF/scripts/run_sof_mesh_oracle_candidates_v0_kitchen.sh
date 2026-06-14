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
ITERATION="${ITERATION:-30000}"

LR_MESH_PATH="${LR_MESH_PATH:-${LR_SOF_MODEL}/test/ours_${ITERATION}/lr_sof_mesh_v0_7.ply}"
HR_MESH_PATH="${HR_MESH_PATH:-${HR_SOF_MODEL}/test/ours_${ITERATION}/hr_sof_mesh_v0_7.ply}"

OUT_ROOT="${OUT_ROOT:-${SOF_ROOT}/output/sof_mesh_oracle_candidates_v0/${SCENE_NAME}}"
RUN_NAME="${RUN_NAME:-lr_shell_hr_oracle_v0}"
RUN_ROOT="${RUN_ROOT:-${OUT_ROOT}/${RUN_NAME}}"

ANCHOR_COUNT="${ANCHOR_COUNT:-250000}"
HR_REFERENCE_SAMPLES="${HR_REFERENCE_SAMPLES:-500000}"
OFFSET_LAYERS="${OFFSET_LAYERS:-9}"
OFFSET_RADIUS_RATIO="${OFFSET_RADIUS_RATIO:-0.003}"
SELECT_DISTANCE_RATIO="${SELECT_DISTANCE_RATIO:-0.0015}"
MIN_NORMAL_ALIGNMENT="${MIN_NORMAL_ALIGNMENT:--0.25}"
SELECTION_MODE="${SELECTION_MODE:-best_per_anchor}"
ATTRACT_STEPS="${ATTRACT_STEPS:-3}"
ATTRACT_ALPHA="${ATTRACT_ALPHA:-0.85}"
MAX_SELECTED="${MAX_SELECTED:-200000}"
CARRIER_SPACING_RATIO="${CARRIER_SPACING_RATIO:-0.001}"

mkdir -p "${RUN_ROOT}"

echo "[sof-oracle-candidates-v0] LR mesh : ${LR_MESH_PATH}"
echo "[sof-oracle-candidates-v0] HR mesh : ${HR_MESH_PATH}"
echo "[sof-oracle-candidates-v0] out     : ${RUN_ROOT}"

if [[ ! -f "${LR_MESH_PATH}" ]]; then
  echo "[sof-oracle-candidates-v0] missing LR mesh: ${LR_MESH_PATH}" >&2
  exit 1
fi
if [[ ! -f "${HR_MESH_PATH}" ]]; then
  echo "[sof-oracle-candidates-v0] missing HR mesh: ${HR_MESH_PATH}" >&2
  exit 1
fi

cd "${SOF_ROOT}"

python -u build_sof_mesh_oracle_candidates_v0.py \
  --lr_mesh_path "${LR_MESH_PATH}" \
  --hr_mesh_path "${HR_MESH_PATH}" \
  --output_dir "${RUN_ROOT}" \
  --anchor_count "${ANCHOR_COUNT}" \
  --hr_reference_samples "${HR_REFERENCE_SAMPLES}" \
  --offset_layers "${OFFSET_LAYERS}" \
  --offset_radius_ratio "${OFFSET_RADIUS_RATIO}" \
  --select_distance_ratio "${SELECT_DISTANCE_RATIO}" \
  --min_normal_alignment "${MIN_NORMAL_ALIGNMENT}" \
  --selection_mode "${SELECTION_MODE}" \
  --attract_steps "${ATTRACT_STEPS}" \
  --attract_alpha "${ATTRACT_ALPHA}" \
  --max_selected "${MAX_SELECTED}" \
  --carrier_spacing_ratio "${CARRIER_SPACING_RATIO}"

python -u build_sof_mesh_oracle_candidate_mesh_preview_v0.py \
  --lr_mesh_path "${LR_MESH_PATH}" \
  --candidate_records "${RUN_ROOT}/oracle_hr_supported_candidate_records_v0.npz" \
  --output_dir "${RUN_ROOT}"

echo "[done] selected point cloud : ${RUN_ROOT}/selected_refined_hr_supported_candidates_v0.ply"
echo "[done] all candidates preview: ${RUN_ROOT}/all_lr_shell_candidates_preview_v0.ply"
echo "[done] selected LR faces   : ${RUN_ROOT}/selected_source_faces_lrmesh_v0.ply"
echo "[done] oracle moved faces  : ${RUN_ROOT}/selected_source_faces_oracle_moved_v0.ply"
echo "[done] carrier payload      : ${RUN_ROOT}/oracle_hr_supported_candidate_carrier_payload_v0.npz"
echo "[done] summary              : ${RUN_ROOT}/build_sof_mesh_oracle_candidates_v0_summary.json"

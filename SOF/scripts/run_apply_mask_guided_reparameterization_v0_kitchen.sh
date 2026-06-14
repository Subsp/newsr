#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-srtest}"

PREPARE_DEBUG_MODEL_PATH="${PREPARE_DEBUG_MODEL_PATH:-${SCENE_ASSET_ROOT}/${SCENE_NAME}_mip_vanilla_images8_v1/mip30k_sof_native_input_init_early4ksoft_v1_debug}"
PREPARE_DEBUG_ITERATION="${PREPARE_DEBUG_ITERATION:-34000}"
STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
MODEL_PATH="${MODEL_PATH:-${PREPARE_DEBUG_MODEL_PATH}/debug_prepare_stages/${STAGE_NAME}}"
ITERATION="${ITERATION:-${PREPARE_DEBUG_ITERATION}}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"

MESH_COMPARE_ROOT="${MESH_COMPARE_ROOT:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}}"
GEOMETRY_MESH_PATH="${GEOMETRY_MESH_PATH:-${MESH_COMPARE_ROOT}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"

GEOMETRY_RUN_NAME="${GEOMETRY_RUN_NAME:-single_mesh_offsurface_farthest_small_more_brightq55_${STAGE_NAME}_v0}"
GEOMETRY_PAYLOAD_ROOT="${GEOMETRY_PAYLOAD_ROOT:-${SOF_ROOT}/output/single_mesh_tangent_gaussian_probe_v0/${SCENE_NAME}/${GEOMETRY_RUN_NAME}}"
GEOMETRY_PAYLOAD_PATH="${GEOMETRY_PAYLOAD_PATH-${GEOMETRY_PAYLOAD_ROOT}/mesh_delta_star_gaussian_candidates_v0.pt}"
GEOMETRY_MASK_KEY="${GEOMETRY_MASK_KEY:-geometry_candidate_mask}"
GEOMETRY_SCORE_KEY="${GEOMETRY_SCORE_KEY:-candidate_score}"
GEOMETRY_NEAREST_SURFACE_KEY="${GEOMETRY_NEAREST_SURFACE_KEY:-nearest_surface_index}"

HIGHLIGHT_RUN_NAME="${HIGHLIGHT_RUN_NAME:-${GEOMETRY_RUN_NAME}}"
HIGHLIGHT_PAYLOAD_ROOT="${HIGHLIGHT_PAYLOAD_ROOT:-${GEOMETRY_PAYLOAD_ROOT}}"
HIGHLIGHT_PAYLOAD_PATH="${HIGHLIGHT_PAYLOAD_PATH-${HIGHLIGHT_PAYLOAD_ROOT}/mesh_delta_star_gaussian_candidates_v0.pt}"
HIGHLIGHT_MASK_KEY="${HIGHLIGHT_MASK_KEY:-brightness_mask}"
HIGHLIGHT_SCORE_KEY="${HIGHLIGHT_SCORE_KEY:-dc_luma}"
HIGHLIGHT_EXCLUDE_SELECTED_KEY="${HIGHLIGHT_EXCLUDE_SELECTED_KEY:-geometry_selected_output_mask}"

OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${STAGE_NAME}_mask_reparam_v0}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${SOF_ROOT}/output/mask_guided_reparameterization_v0/${SCENE_NAME}/${OUTPUT_RUN_NAME}}"

SAVE_INTERMEDIATE_MODELS="${SAVE_INTERMEDIATE_MODELS:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
SPLIT="${SPLIT:-test}"
MAX_VIEWS="${MAX_VIEWS:-12}"

GEOMETRY_APPLY_TO_CHILDREN="${GEOMETRY_APPLY_TO_CHILDREN:-0}"
GEOMETRY_MAX_FRACTION="${GEOMETRY_MAX_FRACTION:-0.015}"
GEOMETRY_MAX_COUNT="${GEOMETRY_MAX_COUNT:-24000}"
GEOMETRY_SPLIT_COUNT="${GEOMETRY_SPLIT_COUNT:-4}"
GEOMETRY_MAX_SPLIT_COUNT="${GEOMETRY_MAX_SPLIT_COUNT:-10}"
GEOMETRY_CHILD_LAYOUT="${GEOMETRY_CHILD_LAYOUT:-major_axis_adaptive_chunk}"
GEOMETRY_CHUNK_ASPECT_TARGET="${GEOMETRY_CHUNK_ASPECT_TARGET:-1.8}"
GEOMETRY_OFFSET_SCALE="${GEOMETRY_OFFSET_SCALE:-0.55}"
GEOMETRY_PARENT_TAU_KEEP="${GEOMETRY_PARENT_TAU_KEEP:-0.85}"
GEOMETRY_CHILD_TAU_RATIO="${GEOMETRY_CHILD_TAU_RATIO:-0.35}"
GEOMETRY_MASS_CAP_EPS="${GEOMETRY_MASS_CAP_EPS:-0.10}"
GEOMETRY_PARENT_DC_SCALE="${GEOMETRY_PARENT_DC_SCALE:-1.0}"
GEOMETRY_PARENT_REST_SCALE="${GEOMETRY_PARENT_REST_SCALE:-1.0}"
GEOMETRY_CHILD_MAJOR_SCALE_MULTIPLIER="${GEOMETRY_CHILD_MAJOR_SCALE_MULTIPLIER:-0.55}"
GEOMETRY_CHILD_MINOR_SCALE_MULTIPLIER="${GEOMETRY_CHILD_MINOR_SCALE_MULTIPLIER:-0.72}"
GEOMETRY_CHILD_NORMAL_SCALE_MULTIPLIER="${GEOMETRY_CHILD_NORMAL_SCALE_MULTIPLIER:-0.72}"
GEOMETRY_CHILD_DC_SCALE="${GEOMETRY_CHILD_DC_SCALE:-1.0}"
GEOMETRY_CHILD_REST_SCALE="${GEOMETRY_CHILD_REST_SCALE:-0.0}"
GEOMETRY_CHILD_FILTER_SCALE="${GEOMETRY_CHILD_FILTER_SCALE:-0.35}"
GEOMETRY_FILTER_CAP_RATIO="${GEOMETRY_FILTER_CAP_RATIO:-0.0015}"
GEOMETRY_ENERGY_CONSERVE_MODE="${GEOMETRY_ENERGY_CONSERVE_MODE:-area}"
GEOMETRY_MESH_PULL_LAMBDA="${GEOMETRY_MESH_PULL_LAMBDA:-0.15}"

HIGHLIGHT_APPLY_TO_CHILDREN="${HIGHLIGHT_APPLY_TO_CHILDREN:-0}"
HIGHLIGHT_MAX_FRACTION="${HIGHLIGHT_MAX_FRACTION:-0.020}"
HIGHLIGHT_MAX_COUNT="${HIGHLIGHT_MAX_COUNT:-32000}"
HIGHLIGHT_SPLIT_COUNT="${HIGHLIGHT_SPLIT_COUNT:-2}"
HIGHLIGHT_MAX_SPLIT_COUNT="${HIGHLIGHT_MAX_SPLIT_COUNT:-2}"
HIGHLIGHT_CHILD_LAYOUT="${HIGHLIGHT_CHILD_LAYOUT:-major_axis}"
HIGHLIGHT_CHUNK_ASPECT_TARGET="${HIGHLIGHT_CHUNK_ASPECT_TARGET:-1.8}"
HIGHLIGHT_OFFSET_SCALE="${HIGHLIGHT_OFFSET_SCALE:-0.28}"
HIGHLIGHT_PARENT_TAU_KEEP="${HIGHLIGHT_PARENT_TAU_KEEP:-0.95}"
HIGHLIGHT_CHILD_TAU_RATIO="${HIGHLIGHT_CHILD_TAU_RATIO:-0.20}"
HIGHLIGHT_MASS_CAP_EPS="${HIGHLIGHT_MASS_CAP_EPS:-0.08}"
HIGHLIGHT_PARENT_DC_SCALE="${HIGHLIGHT_PARENT_DC_SCALE:-1.0}"
HIGHLIGHT_PARENT_REST_SCALE="${HIGHLIGHT_PARENT_REST_SCALE:-0.50}"
HIGHLIGHT_CHILD_MAJOR_SCALE_MULTIPLIER="${HIGHLIGHT_CHILD_MAJOR_SCALE_MULTIPLIER:-0.80}"
HIGHLIGHT_CHILD_MINOR_SCALE_MULTIPLIER="${HIGHLIGHT_CHILD_MINOR_SCALE_MULTIPLIER:-0.88}"
HIGHLIGHT_CHILD_NORMAL_SCALE_MULTIPLIER="${HIGHLIGHT_CHILD_NORMAL_SCALE_MULTIPLIER:-0.92}"
HIGHLIGHT_CHILD_DC_SCALE="${HIGHLIGHT_CHILD_DC_SCALE:-1.0}"
HIGHLIGHT_CHILD_REST_SCALE="${HIGHLIGHT_CHILD_REST_SCALE:-0.0}"
HIGHLIGHT_CHILD_FILTER_SCALE="${HIGHLIGHT_CHILD_FILTER_SCALE:-0.45}"
HIGHLIGHT_FILTER_CAP_RATIO="${HIGHLIGHT_FILTER_CAP_RATIO:-0.0015}"
HIGHLIGHT_ENERGY_CONSERVE_MODE="${HIGHLIGHT_ENERGY_CONSERVE_MODE:-area}"

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

echo "[mask-reparam-v0] scene           : ${SCENE_ROOT}"
echo "[mask-reparam-v0] model           : ${MODEL_PATH} iter=${ITERATION}"
echo "[mask-reparam-v0] geometry payload: ${GEOMETRY_PAYLOAD_PATH} key=${GEOMETRY_MASK_KEY}"
echo "[mask-reparam-v0] highlight payload: ${HIGHLIGHT_PAYLOAD_PATH} key=${HIGHLIGHT_MASK_KEY}"
echo "[mask-reparam-v0] output model    : ${OUTPUT_MODEL_PATH}"

for path in "${SCENE_ROOT}" "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply"; do
  if [[ ! -e "${path}" ]]; then
    echo "[mask-reparam-v0] required path not found: ${path}" >&2
    exit 1
  fi
done

if [[ -n "${GEOMETRY_PAYLOAD_PATH}" && ! -f "${GEOMETRY_PAYLOAD_PATH}" ]]; then
  echo "[mask-reparam-v0] geometry payload not found: ${GEOMETRY_PAYLOAD_PATH}" >&2
  exit 1
fi
if [[ -n "${HIGHLIGHT_PAYLOAD_PATH}" && ! -f "${HIGHLIGHT_PAYLOAD_PATH}" ]]; then
  echo "[mask-reparam-v0] highlight payload not found: ${HIGHLIGHT_PAYLOAD_PATH}" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/apply_mask_guided_reparameterization_v0.py"
  --scene_root "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --output_model_path "${OUTPUT_MODEL_PATH}"
  --images_subdir "${IMAGES_SUBDIR}"
  --iteration "${ITERATION}"
  --geometry_payload_path "${GEOMETRY_PAYLOAD_PATH}"
  --geometry_mask_key "${GEOMETRY_MASK_KEY}"
  --geometry_score_key "${GEOMETRY_SCORE_KEY}"
  --geometry_nearest_surface_key "${GEOMETRY_NEAREST_SURFACE_KEY}"
  --geometry_mesh_path "${GEOMETRY_MESH_PATH}"
  --geometry_max_fraction "${GEOMETRY_MAX_FRACTION}"
  --geometry_max_count "${GEOMETRY_MAX_COUNT}"
  --geometry_split_count "${GEOMETRY_SPLIT_COUNT}"
  --geometry_max_split_count "${GEOMETRY_MAX_SPLIT_COUNT}"
  --geometry_child_layout "${GEOMETRY_CHILD_LAYOUT}"
  --geometry_chunk_aspect_target "${GEOMETRY_CHUNK_ASPECT_TARGET}"
  --geometry_offset_scale "${GEOMETRY_OFFSET_SCALE}"
  --geometry_parent_tau_keep "${GEOMETRY_PARENT_TAU_KEEP}"
  --geometry_child_tau_ratio "${GEOMETRY_CHILD_TAU_RATIO}"
  --geometry_mass_cap_eps "${GEOMETRY_MASS_CAP_EPS}"
  --geometry_parent_dc_scale "${GEOMETRY_PARENT_DC_SCALE}"
  --geometry_parent_rest_scale "${GEOMETRY_PARENT_REST_SCALE}"
  --geometry_child_major_scale_multiplier "${GEOMETRY_CHILD_MAJOR_SCALE_MULTIPLIER}"
  --geometry_child_minor_scale_multiplier "${GEOMETRY_CHILD_MINOR_SCALE_MULTIPLIER}"
  --geometry_child_normal_scale_multiplier "${GEOMETRY_CHILD_NORMAL_SCALE_MULTIPLIER}"
  --geometry_child_dc_scale "${GEOMETRY_CHILD_DC_SCALE}"
  --geometry_child_rest_scale "${GEOMETRY_CHILD_REST_SCALE}"
  --geometry_child_filter_scale "${GEOMETRY_CHILD_FILTER_SCALE}"
  --geometry_filter_cap_ratio "${GEOMETRY_FILTER_CAP_RATIO}"
  --geometry_energy_conserve_mode "${GEOMETRY_ENERGY_CONSERVE_MODE}"
  --geometry_mesh_pull_lambda "${GEOMETRY_MESH_PULL_LAMBDA}"
  --highlight_payload_path "${HIGHLIGHT_PAYLOAD_PATH}"
  --highlight_mask_key "${HIGHLIGHT_MASK_KEY}"
  --highlight_score_key "${HIGHLIGHT_SCORE_KEY}"
  --highlight_exclude_selected_key "${HIGHLIGHT_EXCLUDE_SELECTED_KEY}"
  --highlight_max_fraction "${HIGHLIGHT_MAX_FRACTION}"
  --highlight_max_count "${HIGHLIGHT_MAX_COUNT}"
  --highlight_split_count "${HIGHLIGHT_SPLIT_COUNT}"
  --highlight_max_split_count "${HIGHLIGHT_MAX_SPLIT_COUNT}"
  --highlight_child_layout "${HIGHLIGHT_CHILD_LAYOUT}"
  --highlight_chunk_aspect_target "${HIGHLIGHT_CHUNK_ASPECT_TARGET}"
  --highlight_offset_scale "${HIGHLIGHT_OFFSET_SCALE}"
  --highlight_parent_tau_keep "${HIGHLIGHT_PARENT_TAU_KEEP}"
  --highlight_child_tau_ratio "${HIGHLIGHT_CHILD_TAU_RATIO}"
  --highlight_mass_cap_eps "${HIGHLIGHT_MASS_CAP_EPS}"
  --highlight_parent_dc_scale "${HIGHLIGHT_PARENT_DC_SCALE}"
  --highlight_parent_rest_scale "${HIGHLIGHT_PARENT_REST_SCALE}"
  --highlight_child_major_scale_multiplier "${HIGHLIGHT_CHILD_MAJOR_SCALE_MULTIPLIER}"
  --highlight_child_minor_scale_multiplier "${HIGHLIGHT_CHILD_MINOR_SCALE_MULTIPLIER}"
  --highlight_child_normal_scale_multiplier "${HIGHLIGHT_CHILD_NORMAL_SCALE_MULTIPLIER}"
  --highlight_child_dc_scale "${HIGHLIGHT_CHILD_DC_SCALE}"
  --highlight_child_rest_scale "${HIGHLIGHT_CHILD_REST_SCALE}"
  --highlight_child_filter_scale "${HIGHLIGHT_CHILD_FILTER_SCALE}"
  --highlight_filter_cap_ratio "${HIGHLIGHT_FILTER_CAP_RATIO}"
  --highlight_energy_conserve_mode "${HIGHLIGHT_ENERGY_CONSERVE_MODE}"
)

if [[ "${GEOMETRY_APPLY_TO_CHILDREN}" == "1" ]]; then
  CMD+=(--geometry_apply_to_children)
fi
if [[ "${HIGHLIGHT_APPLY_TO_CHILDREN}" == "1" ]]; then
  CMD+=(--highlight_apply_to_children)
fi
if [[ "${SAVE_INTERMEDIATE_MODELS}" == "1" ]]; then
  CMD+=(--save_intermediate_models)
fi

"${CMD[@]}"

echo "[done] output model : ${OUTPUT_MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
echo "[done] summary      : ${OUTPUT_MODEL_PATH}/mask_guided_reparameterization_summary.json"

if [[ "${RUN_RENDER}" == "1" ]]; then
  "${PYTHON_BIN}" -u "${SOF_ROOT}/scripts/render_model_no_gt.py" \
    --scene_root "${SCENE_ROOT}" \
    --model_path "${OUTPUT_MODEL_PATH}" \
    --images_subdir "${IMAGES_SUBDIR}" \
    --iteration "${ITERATION}" \
    --split "${SPLIT}" \
    --max_views "${MAX_VIEWS}"
  echo "[done] preview renders: ${OUTPUT_MODEL_PATH}/${SPLIT}/ours_${ITERATION}/renders"
fi

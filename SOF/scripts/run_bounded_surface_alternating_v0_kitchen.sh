#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

START_RUN_NAME="${START_RUN_NAME:-debug_stage_00b3_after_scale_canonicalize_geometry_only_v0}"
START_MODEL_PATH="${START_MODEL_PATH:-${SOF_ROOT}/output/mask_guided_reparameterization_v0/${SCENE_NAME}/${START_RUN_NAME}}"
START_ITERATION="${START_ITERATION:-34000}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
MESH_PATH="${MESH_PATH:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"

PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-quality_fuse_v1_maskreparam_geometry_only_safe_hf_v1}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}}"
SR_PRIOR_ROOT="${SR_PRIOR_ROOT:-${PREPARED_SR_PRIOR_ROOT}/fused_priors}"
SR_ANCHOR_ROOT="${SR_ANCHOR_ROOT:-${PREPARED_SR_PRIOR_ROOT}/aligned_references}"
SR_PRIOR_MASK_DIR="${SR_PRIOR_MASK_DIR:-${PREPARED_SR_PRIOR_ROOT}/usable_masks}"

RUN_NAME="${RUN_NAME:-${START_RUN_NAME}_bounded_surface_alt_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOF_ROOT}/output/bounded_surface_alternating_v0/${SCENE_NAME}}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${OUTPUT_ROOT}/${RUN_NAME}}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_2}"

PHASE_MODE="${PHASE_MODE:-alternating}"
CYCLES="${CYCLES:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-0}"
APPEARANCE_STEPS="${APPEARANCE_STEPS:-200}"
GEOMETRY_STEPS="${GEOMETRY_STEPS:-200}"
SAVE_EVERY_CYCLES="${SAVE_EVERY_CYCLES:-1}"
SAVE_INITIAL_SURFACE_MAP="${SAVE_INITIAL_SURFACE_MAP:-1}"
SAVE_CYCLE_SURFACE_MAPS="${SAVE_CYCLE_SURFACE_MAPS:-1}"
SAVE_CYCLE_SURFACE_TARGETS="${SAVE_CYCLE_SURFACE_TARGETS:-1}"
SAVE_CYCLE_MEMORY="${SAVE_CYCLE_MEMORY:-1}"
SAVE_FINAL_SURFACE_MAP="${SAVE_FINAL_SURFACE_MAP:-1}"
SAVE_FINAL_MEMORY="${SAVE_FINAL_MEMORY:-1}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:--1}"

MAX_VIEWS="${MAX_VIEWS:-0}"
FACE_K="${FACE_K:-8}"
BIND_CHUNK_SIZE="${BIND_CHUNK_SIZE:-16384}"
TAU_FLOOR="${TAU_FLOOR:-0.002}"
TAU_EDGE_SCALE="${TAU_EDGE_SCALE:-0.4}"
MIN_SUPPORT_VIEWS="${MIN_SUPPORT_VIEWS:-1}"
MIN_SAMPLE_WEIGHT="${MIN_SAMPLE_WEIGHT:-0.05}"
AGREEMENT_SIGMA="${AGREEMENT_SIGMA:-0.07}"
BASE_SIGMA="${BASE_SIGMA:-0.08}"
BASE_SUPPORT_MODE="${BASE_SUPPORT_MODE:-multiply}"
BASE_SUPPORT_FLOOR="${BASE_SUPPORT_FLOOR:-0.0}"
AGGREGATION_RESIDUAL_SAMPLE_CLIP="${AGGREGATION_RESIDUAL_SAMPLE_CLIP:-0.0}"
AGGREGATION_RESIDUAL_BAND="${AGGREGATION_RESIDUAL_BAND:-raw}"
AGGREGATION_RESIDUAL_LOWPASS_KERNEL="${AGGREGATION_RESIDUAL_LOWPASS_KERNEL:-5}"
AGGREGATION_RESIDUAL_MIN_L1="${AGGREGATION_RESIDUAL_MIN_L1:-0.0}"
AGGREGATION_RESIDUAL_MAX_L1="${AGGREGATION_RESIDUAL_MAX_L1:-0.0}"
DISAGREEMENT_SIGMA="${DISAGREEMENT_SIGMA:-0.10}"
AGGREGATION_MODE="${AGGREGATION_MODE:-trimmed_mean}"
ROBUST_TRIM_SIGMA="${ROBUST_TRIM_SIGMA:-0.12}"
ROBUST_TRIM_DISAGREEMENT_SCALE="${ROBUST_TRIM_DISAGREEMENT_SCALE:-2.5}"

ENABLE_SURFACE_MEMORY="${ENABLE_SURFACE_MEMORY:-1}"
MEMORY_BETA="${MEMORY_BETA:-0.20}"
MEMORY_MIN_CONFIDENCE="${MEMORY_MIN_CONFIDENCE:-0.05}"
MEMORY_MAX_DISAGREEMENT="${MEMORY_MAX_DISAGREEMENT:-0.16}"
MEMORY_STABLE_UPDATES="${MEMORY_STABLE_UPDATES:-2}"
MEMORY_STABLE_MIN_CONFIDENCE="${MEMORY_STABLE_MIN_CONFIDENCE:-0.08}"
MEMORY_STABLE_MAX_DISAGREEMENT="${MEMORY_STABLE_MAX_DISAGREEMENT:-0.12}"
MEMORY_ELIGIBILITY="${MEMORY_ELIGIBILITY:-selected}"
MEMORY_LOOSE_MIN_VISIBLE_VIEWS="${MEMORY_LOOSE_MIN_VISIBLE_VIEWS:-2}"
MEMORY_LOOSE_MIN_TOUCH_VIEWS="${MEMORY_LOOSE_MIN_TOUCH_VIEWS:-1}"
MEMORY_LOOSE_MIN_TOUCH_RATIO="${MEMORY_LOOSE_MIN_TOUCH_RATIO:-0.0}"

APPEARANCE_LR="${APPEARANCE_LR:-5e-4}"
LAMBDA_DC_ANCHOR="${LAMBDA_DC_ANCHOR:-0.05}"
LAMBDA_BASE_GUARD="${LAMBDA_BASE_GUARD:-0.10}"
APPEARANCE_TARGET_MODE="${APPEARANCE_TARGET_MODE:-absolute}"
APPEARANCE_RESIDUAL_CLIP="${APPEARANCE_RESIDUAL_CLIP:-0.0}"
APPEARANCE_RESIDUAL_SCALE="${APPEARANCE_RESIDUAL_SCALE:-1.0}"

GEOMETRY_LR="${GEOMETRY_LR:-5e-4}"
GEOMETRY_PRIOR_MASK_THRESHOLD="${GEOMETRY_PRIOR_MASK_THRESHOLD:-0.05}"
LAMBDA_UV="${LAMBDA_UV:-0.05}"
LAMBDA_DELTA="${LAMBDA_DELTA:-0.05}"
TRUSTED_TANGENT_SCALE="${TRUSTED_TANGENT_SCALE:-2.5}"
LOOSE_TANGENT_SCALE="${LOOSE_TANGENT_SCALE:-1.5}"
TRUSTED_NORMAL_SCALE="${TRUSTED_NORMAL_SCALE:-1.5}"
LOOSE_NORMAL_SCALE="${LOOSE_NORMAL_SCALE:-0.75}"

PRIOR_PREFILTER_VIEW_LIMIT="${PRIOR_PREFILTER_VIEW_LIMIT:-0}"
PRIOR_PREFILTER_MIN_TOUCH_VIEWS="${PRIOR_PREFILTER_MIN_TOUCH_VIEWS:-1}"
PRIOR_PREFILTER_MIN_VISIBLE_VIEWS="${PRIOR_PREFILTER_MIN_VISIBLE_VIEWS:-1}"
PRIOR_PREFILTER_MIN_TOUCH_RATIO="${PRIOR_PREFILTER_MIN_TOUCH_RATIO:-0.0}"
PRIOR_PREFILTER_MIN_CANDIDATE_OPACITY="${PRIOR_PREFILTER_MIN_CANDIDATE_OPACITY:-0.0}"
PRIOR_PREFILTER_RADIUS_SCALE="${PRIOR_PREFILTER_RADIUS_SCALE:-0.5}"
PRIOR_PREFILTER_MIN_TOUCH_RADIUS_PX="${PRIOR_PREFILTER_MIN_TOUCH_RADIUS_PX:-2.0}"
PRIOR_PREFILTER_MAX_TOUCH_RADIUS_PX="${PRIOR_PREFILTER_MAX_TOUCH_RADIUS_PX:-16.0}"
DUMP_MASKED_PRIOR_INPUTS="${DUMP_MASKED_PRIOR_INPUTS:-1}"
DUMP_MASKED_PRIOR_MAX_VIEWS="${DUMP_MASKED_PRIOR_MAX_VIEWS:-16}"

PRUNE_AFTER_CYCLE="${PRUNE_AFTER_CYCLE:-0}"
PRUNE_BAD_STREAK="${PRUNE_BAD_STREAK:-2}"
PRUNE_CONFIDENCE_THRESHOLD="${PRUNE_CONFIDENCE_THRESHOLD:-0.02}"
PRUNE_DISAGREEMENT_THRESHOLD="${PRUNE_DISAGREEMENT_THRESHOLD:-0.18}"
PRUNE_COLOR_ERROR_THRESHOLD="${PRUNE_COLOR_ERROR_THRESHOLD:-0.12}"
PRUNE_MIN_VIEWS="${PRUNE_MIN_VIEWS:-1}"
PROTECT_CONFIDENCE_THRESHOLD="${PROTECT_CONFIDENCE_THRESHOLD:-0.08}"

SR_PRIOR_CONSISTENCY_THRESHOLD="${SR_PRIOR_CONSISTENCY_THRESHOLD:-0.08}"
SR_PRIOR_MASK_FLOOR="${SR_PRIOR_MASK_FLOOR:-0.0}"
SR_PRIOR_MASK_SUFFIX="${SR_PRIOR_MASK_SUFFIX:-}"
DEPTH_MIN="${DEPTH_MIN:-0.02}"
CHARBONNIER_EPS="${CHARBONNIER_EPS:-1e-3}"
SEED="${SEED:-0}"

if [[ ! -d "${START_MODEL_PATH}" ]]; then
  echo "[bounded-surface-alternating-v0] missing start model: ${START_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MESH_PATH}" ]]; then
  echo "[bounded-surface-alternating-v0] missing mesh: ${MESH_PATH}" >&2
  exit 1
fi
if [[ ! -d "${SR_PRIOR_ROOT}" ]]; then
  echo "[bounded-surface-alternating-v0] missing SR prior root: ${SR_PRIOR_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${SR_ANCHOR_ROOT}" ]]; then
  echo "[bounded-surface-alternating-v0] missing SR anchor root: ${SR_ANCHOR_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${SR_PRIOR_MASK_DIR}" ]]; then
  SR_PRIOR_MASK_DIR=""
fi

mkdir -p "${OUTPUT_MODEL_PATH}"

echo "[bounded-surface-alternating-v0] scene        : ${SCENE_ROOT}"
echo "[bounded-surface-alternating-v0] start model  : ${START_MODEL_PATH}"
echo "[bounded-surface-alternating-v0] mesh         : ${MESH_PATH}"
echo "[bounded-surface-alternating-v0] prior dir    : ${SR_PRIOR_ROOT}"
echo "[bounded-surface-alternating-v0] anchor dir   : ${SR_ANCHOR_ROOT}"
echo "[bounded-surface-alternating-v0] output       : ${OUTPUT_MODEL_PATH}"
echo "[bounded-surface-alternating-v0] images       : ${IMAGES_SUBDIR}"
echo "[bounded-surface-alternating-v0] phase        : ${PHASE_MODE}"
echo "[bounded-surface-alternating-v0] schedule     : cycles=${CYCLES} total_steps=${TOTAL_STEPS} A=${APPEARANCE_STEPS} G=${GEOMETRY_STEPS}"
echo "[bounded-surface-alternating-v0] save         : every=${SAVE_EVERY_CYCLES} cycle_maps=${SAVE_CYCLE_SURFACE_MAPS} cycle_targets=${SAVE_CYCLE_SURFACE_TARGETS} cycle_memory=${SAVE_CYCLE_MEMORY} final_map=${SAVE_FINAL_SURFACE_MAP}"
echo "[bounded-surface-alternating-v0] aggregation : ${AGGREGATION_MODE} trim_sigma=${ROBUST_TRIM_SIGMA} base_support=${BASE_SUPPORT_MODE} floor=${BASE_SUPPORT_FLOOR} sample_res_clip=${AGGREGATION_RESIDUAL_SAMPLE_CLIP} residual_band=${AGGREGATION_RESIDUAL_BAND} lowpass=${AGGREGATION_RESIDUAL_LOWPASS_KERNEL} residual_l1=[${AGGREGATION_RESIDUAL_MIN_L1},${AGGREGATION_RESIDUAL_MAX_L1}]"
echo "[bounded-surface-alternating-v0] memory      : enabled=${ENABLE_SURFACE_MEMORY} beta=${MEMORY_BETA} stable_updates=${MEMORY_STABLE_UPDATES} eligibility=${MEMORY_ELIGIBILITY}"
echo "[bounded-surface-alternating-v0] A-target    : ${APPEARANCE_TARGET_MODE} clip=${APPEARANCE_RESIDUAL_CLIP} scale=${APPEARANCE_RESIDUAL_SCALE}"

CMD=(
  python -u "${SOF_ROOT}/train_bounded_surface_alternating_v0.py"
  -s "${SCENE_ROOT}"
  --start_model_path "${START_MODEL_PATH}"
  --start_iteration "${START_ITERATION}"
  --mesh_path "${MESH_PATH}"
  --output_model_path "${OUTPUT_MODEL_PATH}"
  --images_subdir "${IMAGES_SUBDIR}"
  --split train
  --max_views "${MAX_VIEWS}"
  --sr_prior_root "${SR_PRIOR_ROOT}"
  --sr_anchor_root "${SR_ANCHOR_ROOT}"
  --phase_mode "${PHASE_MODE}"
  --cycles "${CYCLES}"
  --total_steps "${TOTAL_STEPS}"
  --appearance_steps "${APPEARANCE_STEPS}"
  --geometry_steps "${GEOMETRY_STEPS}"
  --save_every_cycles "${SAVE_EVERY_CYCLES}"
  --save_initial_surface_map "${SAVE_INITIAL_SURFACE_MAP}"
  --save_cycle_surface_maps "${SAVE_CYCLE_SURFACE_MAPS}"
  --save_cycle_surface_targets "${SAVE_CYCLE_SURFACE_TARGETS}"
  --save_cycle_memory "${SAVE_CYCLE_MEMORY}"
  --save_final_surface_map "${SAVE_FINAL_SURFACE_MAP}"
  --save_final_memory "${SAVE_FINAL_MEMORY}"
  --output_iteration "${OUTPUT_ITERATION}"
  --face_k "${FACE_K}"
  --bind_chunk_size "${BIND_CHUNK_SIZE}"
  --tau_floor "${TAU_FLOOR}"
  --tau_edge_scale "${TAU_EDGE_SCALE}"
  --min_support_views "${MIN_SUPPORT_VIEWS}"
  --min_sample_weight "${MIN_SAMPLE_WEIGHT}"
  --agreement_sigma "${AGREEMENT_SIGMA}"
  --base_sigma "${BASE_SIGMA}"
  --base_support_mode "${BASE_SUPPORT_MODE}"
  --base_support_floor "${BASE_SUPPORT_FLOOR}"
  --aggregation_residual_sample_clip "${AGGREGATION_RESIDUAL_SAMPLE_CLIP}"
  --aggregation_residual_band "${AGGREGATION_RESIDUAL_BAND}"
  --aggregation_residual_lowpass_kernel "${AGGREGATION_RESIDUAL_LOWPASS_KERNEL}"
  --aggregation_residual_min_l1 "${AGGREGATION_RESIDUAL_MIN_L1}"
  --aggregation_residual_max_l1 "${AGGREGATION_RESIDUAL_MAX_L1}"
  --disagreement_sigma "${DISAGREEMENT_SIGMA}"
  --aggregation_mode "${AGGREGATION_MODE}"
  --robust_trim_sigma "${ROBUST_TRIM_SIGMA}"
  --robust_trim_disagreement_scale "${ROBUST_TRIM_DISAGREEMENT_SCALE}"
  --enable_surface_memory "${ENABLE_SURFACE_MEMORY}"
  --memory_beta "${MEMORY_BETA}"
  --memory_min_confidence "${MEMORY_MIN_CONFIDENCE}"
  --memory_max_disagreement "${MEMORY_MAX_DISAGREEMENT}"
  --memory_stable_updates "${MEMORY_STABLE_UPDATES}"
  --memory_stable_min_confidence "${MEMORY_STABLE_MIN_CONFIDENCE}"
  --memory_stable_max_disagreement "${MEMORY_STABLE_MAX_DISAGREEMENT}"
  --memory_eligibility "${MEMORY_ELIGIBILITY}"
  --memory_loose_min_visible_views "${MEMORY_LOOSE_MIN_VISIBLE_VIEWS}"
  --memory_loose_min_touch_views "${MEMORY_LOOSE_MIN_TOUCH_VIEWS}"
  --memory_loose_min_touch_ratio "${MEMORY_LOOSE_MIN_TOUCH_RATIO}"
  --appearance_lr "${APPEARANCE_LR}"
  --lambda_dc_anchor "${LAMBDA_DC_ANCHOR}"
  --lambda_base_guard "${LAMBDA_BASE_GUARD}"
  --appearance_target_mode "${APPEARANCE_TARGET_MODE}"
  --appearance_residual_clip "${APPEARANCE_RESIDUAL_CLIP}"
  --appearance_residual_scale "${APPEARANCE_RESIDUAL_SCALE}"
  --geometry_lr "${GEOMETRY_LR}"
  --geometry_prior_mask_threshold "${GEOMETRY_PRIOR_MASK_THRESHOLD}"
  --lambda_uv "${LAMBDA_UV}"
  --lambda_delta "${LAMBDA_DELTA}"
  --trusted_tangent_scale "${TRUSTED_TANGENT_SCALE}"
  --loose_tangent_scale "${LOOSE_TANGENT_SCALE}"
  --trusted_normal_scale "${TRUSTED_NORMAL_SCALE}"
  --loose_normal_scale "${LOOSE_NORMAL_SCALE}"
  --prior_prefilter_view_limit "${PRIOR_PREFILTER_VIEW_LIMIT}"
  --prior_prefilter_min_touch_views "${PRIOR_PREFILTER_MIN_TOUCH_VIEWS}"
  --prior_prefilter_min_visible_views "${PRIOR_PREFILTER_MIN_VISIBLE_VIEWS}"
  --prior_prefilter_min_touch_ratio "${PRIOR_PREFILTER_MIN_TOUCH_RATIO}"
  --prior_prefilter_min_candidate_opacity "${PRIOR_PREFILTER_MIN_CANDIDATE_OPACITY}"
  --prior_prefilter_radius_scale "${PRIOR_PREFILTER_RADIUS_SCALE}"
  --prior_prefilter_min_touch_radius_px "${PRIOR_PREFILTER_MIN_TOUCH_RADIUS_PX}"
  --prior_prefilter_max_touch_radius_px "${PRIOR_PREFILTER_MAX_TOUCH_RADIUS_PX}"
  --dump_masked_prior_max_views "${DUMP_MASKED_PRIOR_MAX_VIEWS}"
  --prune_bad_streak "${PRUNE_BAD_STREAK}"
  --prune_confidence_threshold "${PRUNE_CONFIDENCE_THRESHOLD}"
  --prune_disagreement_threshold "${PRUNE_DISAGREEMENT_THRESHOLD}"
  --prune_color_error_threshold "${PRUNE_COLOR_ERROR_THRESHOLD}"
  --prune_min_views "${PRUNE_MIN_VIEWS}"
  --protect_confidence_threshold "${PROTECT_CONFIDENCE_THRESHOLD}"
  --sr_prior_consistency_threshold "${SR_PRIOR_CONSISTENCY_THRESHOLD}"
  --sr_prior_mask_floor "${SR_PRIOR_MASK_FLOOR}"
  --sr_prior_mask_suffix "${SR_PRIOR_MASK_SUFFIX}"
  --depth_min "${DEPTH_MIN}"
  --charbonnier_eps "${CHARBONNIER_EPS}"
  --seed "${SEED}"
)

if [[ -n "${SR_PRIOR_MASK_DIR}" ]]; then
  CMD+=(--sr_prior_mask_dir "${SR_PRIOR_MASK_DIR}")
fi
if [[ "${PRUNE_AFTER_CYCLE}" == "1" ]]; then
  CMD+=(--prune_after_cycle)
fi
if [[ "${DUMP_MASKED_PRIOR_INPUTS}" == "1" ]]; then
  CMD+=(--dump_masked_prior_inputs)
fi

"${CMD[@]}"

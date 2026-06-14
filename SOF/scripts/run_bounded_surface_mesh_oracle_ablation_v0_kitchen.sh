#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

START_RUN_NAME="${START_RUN_NAME:-debug_stage_00b3_after_scale_canonicalize_geometry_only_v0}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${SOF_ROOT}/output/mask_guided_reparameterization_v0/${SCENE_NAME}/${START_RUN_NAME}}"
BASE_ITERATION="${BASE_ITERATION:-34000}"

LR_SMOOTH_MODEL_PATH="${LR_SMOOTH_MODEL_PATH:-${SOF_ROOT}/output/surface_smooth_filter_v0/${SCENE_NAME}/${START_RUN_NAME}_delta_sigma_e3_delta_sigma_bodyaware_stronger_v1}"
LR_SMOOTH_ITERATION="${LR_SMOOTH_ITERATION:-34000}"

STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
LR_MESH_PATH="${LR_MESH_PATH:-${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}/${STAGE_NAME}_prepare_stage_sof_export_mesh_v0_${STAGE_NAME}_7.ply}"
GT_MESH_PATH="${GT_MESH_PATH:-}"

PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-sof_surface_v0_images_8_to_images_2_mask0.12_soft}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${PREPARED_SR_PRIOR_NAME}}"
SR_PRIOR_ROOT="${SR_PRIOR_ROOT:-${PREPARED_SR_PRIOR_ROOT}/priors}"
SR_ANCHOR_ROOT="${SR_ANCHOR_ROOT:-${SR_PRIOR_ROOT}}"
SR_PRIOR_MASK_DIR="${SR_PRIOR_MASK_DIR:-}"

ORACLE_MODE="${ORACLE_MODE:-gt_binding}"
RUN_AUDIT="${RUN_AUDIT:-1}"
AUDIT_MAX_VIEWS="${AUDIT_MAX_VIEWS:-24}"
TRAIN_MAX_VIEWS="${TRAIN_MAX_VIEWS:-0}"

# Keep all oracle arms on the current best clean readout setting unless overridden.
PHASE_MODE="${PHASE_MODE:-appearance_only}"
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
APPEARANCE_STEPS="${APPEARANCE_STEPS:-500}"
GEOMETRY_STEPS="${GEOMETRY_STEPS:-250}"
OUTPUT_ITERATION="${OUTPUT_ITERATION:-35000}"

BASE_SUPPORT_MODE="${BASE_SUPPORT_MODE:-floor}"
BASE_SUPPORT_FLOOR="${BASE_SUPPORT_FLOOR:-0.25}"
AGGREGATION_RESIDUAL_SAMPLE_CLIP="${AGGREGATION_RESIDUAL_SAMPLE_CLIP:-0.12}"
AGGREGATION_RESIDUAL_BAND="${AGGREGATION_RESIDUAL_BAND:-raw}"
AGGREGATION_RESIDUAL_LOWPASS_KERNEL="${AGGREGATION_RESIDUAL_LOWPASS_KERNEL:-5}"
AGGREGATION_RESIDUAL_MIN_L1="${AGGREGATION_RESIDUAL_MIN_L1:-0.0}"
AGGREGATION_RESIDUAL_MAX_L1="${AGGREGATION_RESIDUAL_MAX_L1:-0.0}"
APPEARANCE_TARGET_MODE="${APPEARANCE_TARGET_MODE:-residual_clipped}"
APPEARANCE_RESIDUAL_CLIP="${APPEARANCE_RESIDUAL_CLIP:-0.05}"
APPEARANCE_RESIDUAL_SCALE="${APPEARANCE_RESIDUAL_SCALE:-1.0}"
MEMORY_ELIGIBILITY="${MEMORY_ELIGIBILITY:-trust_surface}"
SAVE_FINAL_MEMORY="${SAVE_FINAL_MEMORY:-1}"
DUMP_MASKED_PRIOR_INPUTS="${DUMP_MASKED_PRIOR_INPUTS:-0}"

require_gt_mesh() {
  if [[ -z "${GT_MESH_PATH}" ]]; then
    echo "[mesh-oracle-ablation-v0] GT_MESH_PATH is required for ORACLE_MODE=${ORACLE_MODE}" >&2
    echo "[mesh-oracle-ablation-v0] example: GT_MESH_PATH=/root/.../gt_reconstruction_mesh.ply ORACLE_MODE=${ORACLE_MODE} bash ${BASH_SOURCE[0]}" >&2
    exit 1
  fi
  if [[ ! -f "${GT_MESH_PATH}" ]]; then
    echo "[mesh-oracle-ablation-v0] missing GT mesh: ${GT_MESH_PATH}" >&2
    exit 1
  fi
}

BOUNDED_RUN_SUFFIX=""
BOUNDED_MESH_PATH=""
BOUNDED_START_MODEL_PATH=""
BOUNDED_START_ITERATION=""
AUDIT_BEFORE_MODEL_PATH=""
AUDIT_BEFORE_ITERATION=""

case "${ORACLE_MODE}" in
  lr_binding)
    BOUNDED_RUN_SUFFIX="oracle_lrmesh_binding_residual_v0"
    BOUNDED_MESH_PATH="${LR_MESH_PATH}"
    BOUNDED_START_MODEL_PATH="${LR_SMOOTH_MODEL_PATH}"
    BOUNDED_START_ITERATION="${LR_SMOOTH_ITERATION}"
    AUDIT_BEFORE_MODEL_PATH="${LR_SMOOTH_MODEL_PATH}"
    AUDIT_BEFORE_ITERATION="${LR_SMOOTH_ITERATION}"
    ;;
  gt_binding)
    require_gt_mesh
    BOUNDED_RUN_SUFFIX="oracle_gtmesh_binding_residual_v0"
    BOUNDED_MESH_PATH="${GT_MESH_PATH}"
    BOUNDED_START_MODEL_PATH="${LR_SMOOTH_MODEL_PATH}"
    BOUNDED_START_ITERATION="${LR_SMOOTH_ITERATION}"
    AUDIT_BEFORE_MODEL_PATH="${LR_SMOOTH_MODEL_PATH}"
    AUDIT_BEFORE_ITERATION="${LR_SMOOTH_ITERATION}"
    ;;
  gt_smooth)
    require_gt_mesh
    GT_SMOOTH_RUN_TAG="${GT_SMOOTH_RUN_TAG:-gtmesh_bodyaware_oracle_v0}"
    GT_SMOOTH_OUTPUT_PATH="${GT_SMOOTH_OUTPUT_PATH:-${SOF_ROOT}/output/surface_smooth_filter_v0/${SCENE_NAME}/${START_RUN_NAME}_delta_sigma_${GT_SMOOTH_RUN_TAG}}"
    if [[ "${SKIP_GT_SMOOTH:-0}" != "1" || ! -d "${GT_SMOOTH_OUTPUT_PATH}" ]]; then
      echo "[mesh-oracle-ablation-v0] running GT static smoothing -> ${GT_SMOOTH_OUTPUT_PATH}"
      MODEL_PATH="${BASE_MODEL_PATH}" \
      ITERATION="${BASE_ITERATION}" \
      MESH_PATH="${GT_MESH_PATH}" \
      MODE="delta_sigma" \
      RUN_TAG="${GT_SMOOTH_RUN_TAG}" \
      OUTPUT_PATH="${GT_SMOOTH_OUTPUT_PATH}" \
      bash "${SOF_ROOT}/scripts/run_surface_smooth_filter_v0_kitchen.sh"
    else
      echo "[mesh-oracle-ablation-v0] reusing existing GT smooth output: ${GT_SMOOTH_OUTPUT_PATH}"
    fi
    BOUNDED_RUN_SUFFIX="oracle_gtmesh_smooth_residual_v0"
    BOUNDED_MESH_PATH="${GT_MESH_PATH}"
    BOUNDED_START_MODEL_PATH="${GT_SMOOTH_OUTPUT_PATH}"
    BOUNDED_START_ITERATION="${BASE_ITERATION}"
    AUDIT_BEFORE_MODEL_PATH="${GT_SMOOTH_OUTPUT_PATH}"
    AUDIT_BEFORE_ITERATION="${BASE_ITERATION}"
    ;;
  lr_conf)
    BOUNDED_RUN_SUFFIX="oracle_lrmesh_conf_residual_v0"
    BOUNDED_MESH_PATH="${LR_MESH_PATH}"
    BOUNDED_START_MODEL_PATH="${LR_SMOOTH_MODEL_PATH}"
    BOUNDED_START_ITERATION="${LR_SMOOTH_ITERATION}"
    AUDIT_BEFORE_MODEL_PATH="${LR_SMOOTH_MODEL_PATH}"
    AUDIT_BEFORE_ITERATION="${LR_SMOOTH_ITERATION}"
    MEMORY_MIN_CONFIDENCE="${MEMORY_MIN_CONFIDENCE:-0.12}"
    MEMORY_STABLE_MIN_CONFIDENCE="${MEMORY_STABLE_MIN_CONFIDENCE:-0.20}"
    MEMORY_MAX_DISAGREEMENT="${MEMORY_MAX_DISAGREEMENT:-0.10}"
    MEMORY_STABLE_MAX_DISAGREEMENT="${MEMORY_STABLE_MAX_DISAGREEMENT:-0.08}"
    MIN_SUPPORT_VIEWS="${MIN_SUPPORT_VIEWS:-4}"
    ;;
  *)
    echo "[mesh-oracle-ablation-v0] unknown ORACLE_MODE=${ORACLE_MODE}" >&2
    echo "[mesh-oracle-ablation-v0] supported: lr_binding | gt_binding | gt_smooth | lr_conf" >&2
    exit 1
    ;;
esac

RUN_NAME="${RUN_NAME:-${START_RUN_NAME}_bounded_aonly_${BOUNDED_RUN_SUFFIX}}"
AFTER_MODEL_PATH="${SOF_ROOT}/output/bounded_surface_alternating_v0/${SCENE_NAME}/${RUN_NAME}"

if [[ ! -d "${BOUNDED_START_MODEL_PATH}" ]]; then
  echo "[mesh-oracle-ablation-v0] missing start model: ${BOUNDED_START_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${BOUNDED_MESH_PATH}" ]]; then
  echo "[mesh-oracle-ablation-v0] missing mesh: ${BOUNDED_MESH_PATH}" >&2
  exit 1
fi

echo "[mesh-oracle-ablation-v0] mode        : ${ORACLE_MODE}"
echo "[mesh-oracle-ablation-v0] start model : ${BOUNDED_START_MODEL_PATH}"
echo "[mesh-oracle-ablation-v0] mesh        : ${BOUNDED_MESH_PATH}"
echo "[mesh-oracle-ablation-v0] run name    : ${RUN_NAME}"
echo "[mesh-oracle-ablation-v0] prior dir   : ${SR_PRIOR_ROOT}"
echo "[mesh-oracle-ablation-v0] target      : ${APPEARANCE_TARGET_MODE} clip=${APPEARANCE_RESIDUAL_CLIP} band=${AGGREGATION_RESIDUAL_BAND}"

export START_MODEL_PATH="${BOUNDED_START_MODEL_PATH}"
export START_ITERATION="${BOUNDED_START_ITERATION}"
export MESH_PATH="${BOUNDED_MESH_PATH}"
export PREPARED_SR_PRIOR_NAME
export SR_PRIOR_ROOT
export SR_ANCHOR_ROOT
export SR_PRIOR_MASK_DIR
export PHASE_MODE
export TOTAL_STEPS
export APPEARANCE_STEPS
export GEOMETRY_STEPS
export OUTPUT_ITERATION
export BASE_SUPPORT_MODE
export BASE_SUPPORT_FLOOR
export AGGREGATION_RESIDUAL_SAMPLE_CLIP
export AGGREGATION_RESIDUAL_BAND
export AGGREGATION_RESIDUAL_LOWPASS_KERNEL
export AGGREGATION_RESIDUAL_MIN_L1
export AGGREGATION_RESIDUAL_MAX_L1
export APPEARANCE_TARGET_MODE
export APPEARANCE_RESIDUAL_CLIP
export APPEARANCE_RESIDUAL_SCALE
export MEMORY_ELIGIBILITY
export SAVE_FINAL_MEMORY
export DUMP_MASKED_PRIOR_INPUTS
export RUN_NAME
export MAX_VIEWS="${TRAIN_MAX_VIEWS}"

if [[ "${ORACLE_MODE}" == "lr_conf" ]]; then
  export MEMORY_MIN_CONFIDENCE
  export MEMORY_STABLE_MIN_CONFIDENCE
  export MEMORY_MAX_DISAGREEMENT
  export MEMORY_STABLE_MAX_DISAGREEMENT
  export MIN_SUPPORT_VIEWS
fi

bash "${SOF_ROOT}/scripts/run_bounded_surface_alternating_mainline5k_safe_v1_kitchen.sh"

if [[ "${RUN_AUDIT}" == "1" ]]; then
  echo "[mesh-oracle-ablation-v0] running readout audit for ${RUN_NAME}"
  BEFORE_MODEL_PATH="${AUDIT_BEFORE_MODEL_PATH}" \
  BEFORE_ITERATION="${AUDIT_BEFORE_ITERATION}" \
  AFTER_RUN_NAME="${RUN_NAME}" \
  AFTER_ITERATION="${OUTPUT_ITERATION}" \
  PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME}" \
  SR_PRIOR_ROOT="${SR_PRIOR_ROOT}" \
  SR_ANCHOR_ROOT="${SR_ANCHOR_ROOT}" \
  SR_PRIOR_MASK_DIR="${SR_PRIOR_MASK_DIR}" \
  APPEARANCE_TARGET_MODE="${APPEARANCE_TARGET_MODE}" \
  APPEARANCE_RESIDUAL_CLIP="${APPEARANCE_RESIDUAL_CLIP}" \
  APPEARANCE_RESIDUAL_SCALE="${APPEARANCE_RESIDUAL_SCALE}" \
  MAX_VIEWS="${AUDIT_MAX_VIEWS}" \
  bash "${SOF_ROOT}/scripts/run_audit_surface_memory_readout_v0_kitchen.sh"
fi

cat <<EOF
[mesh-oracle-ablation-v0] done
  after_model_path: ${AFTER_MODEL_PATH}
  summary_json     : ${AFTER_MODEL_PATH}/surface_bounded_alternating_summary.json
  audit_json       : ${AFTER_MODEL_PATH}/audit_surface_memory_readout_v0/surface_memory_readout_audit_v0_summary.json
EOF

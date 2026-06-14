#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="${SOF_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

DELETE="${DELETE:-0}"
CONFIRM="${CONFIRM:-}"
ALLOW_MISSING_REQUIRED="${ALLOW_MISSING_REQUIRED:-0}"

MIP_GROUP_NAME="${MIP_GROUP_NAME:-${SCENE_NAME}_mip_vanilla_images8_v1}"
MIP_GROUP_ROOT="${SCENE_ASSET_ROOT}/${MIP_GROUP_NAME}"
MIP_PARENT_NAME="${MIP_PARENT_NAME:-mip30k_sof_native_input_init_repair_v0}"
MIP_RAW_NAME="${MIP_RAW_NAME:-mip30k}"
STAGE_NAME="${STAGE_NAME:-debug_stage_00b3_after_scale_canonicalize}"
GEOMETRY_PROBE_RUN_NAME="${GEOMETRY_PROBE_RUN_NAME:-single_mesh_offsurface_farthest_small_more_brightq55_${STAGE_NAME}_v0}"
LAYER_STACK_RUN_NAME="${LAYER_STACK_RUN_NAME:-${GEOMETRY_PROBE_RUN_NAME}_layer_stack}"
MASK_REPARAM_RUN_NAME="${MASK_REPARAM_RUN_NAME:-${STAGE_NAME}_mask_reparam_v0}"

STABLE_PRIOR_NAME="${STABLE_PRIOR_NAME:-sof_surface_v0_images_8_to_images_2_mask0.12_soft}"
QUALITY_PRIOR_NAME="${QUALITY_PRIOR_NAME:-quality_fuse_v0_k27_base_mip30ksr_quick20}"
QUALITY_PRIOR_SAFE_NAME="${QUALITY_PRIOR_SAFE_NAME:-quality_fuse_v1_k27_base_mip30ksr_safe_hf_v1}"

CLEANUP_PROFILE="${CLEANUP_PROFILE:-working_set_v0}"
case "${CLEANUP_PROFILE}" in
  working_set_v0)
    DEFAULT_KEEP_PREPARED_PRIORS="${STABLE_PRIOR_NAME} ${QUALITY_PRIOR_NAME} ${QUALITY_PRIOR_SAFE_NAME}"
    DEFAULT_KEEP_MIP_GROUP_DIRS="${MIP_RAW_NAME} ${MIP_PARENT_NAME}"
    DEFAULT_KEEP_CLEANUP_RUNS="view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1"
    DEFAULT_KEEP_RECOVER_RUNS="k27_base k27_quality_fuse_v0 k27_quality_fuse_directprior_warm_v0 k27_base_quality_fuse_safe_hf_v1 k27_base_starq_release_gentle_safe_hf_v1 k27_base_hf_error_correct_rest_v0"
    DEFAULT_KEEP_STAR_QUARANTINE_RUNS="k27_base_starq_gentle_safe_hf_v1"
    DEFAULT_KEEP_MIP_TO_SOF_RUNS="k22_reg"
    DEFAULT_KEEP_DIAG_RUNS=""
    DEFAULT_KEEP_MIPSPLATTING_PRIOR_RUNS="stablesr_mipsplatting_hrprior_finetune_stronger_34k_v1 stablesr_mipsplatting_soffull_merged_smoke_v6"
    ;;
  minimal_hf_rest_v0)
    DEFAULT_KEEP_PREPARED_PRIORS="${STABLE_PRIOR_NAME} ${QUALITY_PRIOR_SAFE_NAME}"
    DEFAULT_KEEP_MIP_GROUP_DIRS="${MIP_RAW_NAME} ${MIP_PARENT_NAME}"
    DEFAULT_KEEP_CLEANUP_RUNS="view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1"
    DEFAULT_KEEP_RECOVER_RUNS="k27_base k27_base_hf_error_correct_rest_v0"
    DEFAULT_KEEP_STAR_QUARANTINE_RUNS="__none__"
    DEFAULT_KEEP_MIP_TO_SOF_RUNS="k22_reg"
    DEFAULT_KEEP_DIAG_RUNS="__none__"
    DEFAULT_KEEP_MIPSPLATTING_PRIOR_RUNS="__none__"
    ;;
  *)
    echo "[cleanup-quality-fuse-v0] unknown CLEANUP_PROFILE=${CLEANUP_PROFILE}" >&2
    echo "[cleanup-quality-fuse-v0] supported profiles: working_set_v0, minimal_hf_rest_v0" >&2
    exit 2
    ;;
esac

KEEP_PREPARED_PRIORS="${KEEP_PREPARED_PRIORS:-${DEFAULT_KEEP_PREPARED_PRIORS}}"
KEEP_MIP_GROUP_DIRS="${KEEP_MIP_GROUP_DIRS:-${DEFAULT_KEEP_MIP_GROUP_DIRS}}"
KEEP_CLEANUP_RUNS="${KEEP_CLEANUP_RUNS:-${DEFAULT_KEEP_CLEANUP_RUNS}}"
KEEP_RECOVER_RUNS="${KEEP_RECOVER_RUNS:-${DEFAULT_KEEP_RECOVER_RUNS}}"
KEEP_STAR_QUARANTINE_RUNS="${KEEP_STAR_QUARANTINE_RUNS:-${DEFAULT_KEEP_STAR_QUARANTINE_RUNS}}"
KEEP_MIP_TO_SOF_RUNS="${KEEP_MIP_TO_SOF_RUNS:-${DEFAULT_KEEP_MIP_TO_SOF_RUNS}}"
KEEP_DIAG_RUNS="${KEEP_DIAG_RUNS:-${DEFAULT_KEEP_DIAG_RUNS}}"
KEEP_MIPSPLATTING_PRIOR_RUNS="${KEEP_MIPSPLATTING_PRIOR_RUNS:-${DEFAULT_KEEP_MIPSPLATTING_PRIOR_RUNS}}"

CLEAN_DIAG="${CLEAN_DIAG:-1}"
CLEAN_DEBUG="${CLEAN_DEBUG:-1}"
CLEAN_RENDER_DUMPS="${CLEAN_RENDER_DUMPS:-0}"
CLEAN_UNKNOWN_TOPLEVEL_OUTPUT="${CLEAN_UNKNOWN_TOPLEVEL_OUTPUT:-0}"

if [[ "${DELETE}" == "1" && "${CONFIRM}" != "delete" ]]; then
  echo "[cleanup-quality-fuse-v0] refusing to delete without CONFIRM=delete" >&2
  exit 2
fi

case "${SOF_ROOT}" in
  /root/autodl-tmp/SOFSR|*/SOFSR) ;;
  *)
    if [[ "${DELETE}" == "1" ]]; then
      echo "[cleanup-quality-fuse-v0] refusing delete for unexpected SOF_ROOT=${SOF_ROOT}" >&2
      echo "[cleanup-quality-fuse-v0] set SOF_ROOT=/root/autodl-tmp/SOFSR on the server" >&2
      exit 2
    fi
    ;;
esac

du_size() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    du -sh -- "${path}" 2>/dev/null | awk '{print $1}'
  else
    printf '%s' '-'
  fi
}

is_name_kept() {
  local name="$1"
  local keep_list="$2"
  local keep
  for keep in ${keep_list}; do
    if [[ "${name}" == "${keep}" ]]; then
      return 0
    fi
  done
  return 1
}

print_path() {
  local tag="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    echo "${tag} $(du_size "${path}") ${path}"
  else
    echo "${tag} MISSING ${path}"
  fi
}

delete_candidate() {
  local path="$1"
  local reason="$2"
  if [[ ! -e "${path}" ]]; then
    return 0
  fi
  if [[ "${DELETE}" == "1" ]]; then
    echo "[delete] $(du_size "${path}") ${path}  # ${reason}"
    rm -rf -- "${path}"
  else
    echo "[dry-delete] $(du_size "${path}") ${path}  # ${reason}"
  fi
}

cleanup_named_children() {
  local base="$1"
  local keep_names="$2"
  local label="$3"
  if [[ ! -d "${base}" ]]; then
    echo "[skip] missing ${label}: ${base}"
    return 0
  fi

  echo
  echo "[scan] ${label}: ${base}"
  while IFS= read -r -d '' child; do
    local name
    name="$(basename -- "${child}")"
    if is_name_kept "${name}" "${keep_names}"; then
      echo "[keep] $(du_size "${child}") ${child}"
    else
      delete_candidate "${child}" "${label}: not in keep list"
    fi
  done < <(find "${base}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
}

cleanup_optional_path() {
  local path="$1"
  local reason="$2"
  local enabled="$3"
  if [[ "${enabled}" == "1" ]]; then
    delete_candidate "${path}" "${reason}"
  elif [[ -e "${path}" ]]; then
    echo "[keep-optional] $(du_size "${path}") ${path}  # ${reason}; set corresponding CLEAN_* flag to delete"
  fi
}

echo "[cleanup-quality-fuse-v0] mode        : $([[ "${DELETE}" == "1" ]] && echo delete || echo dry-run)"
echo "[cleanup-quality-fuse-v0] sof root    : ${SOF_ROOT}"
echo "[cleanup-quality-fuse-v0] scene root  : ${SCENE_ROOT}"
echo "[cleanup-quality-fuse-v0] assets root : ${SCENE_ASSET_ROOT}"
echo "[cleanup-quality-fuse-v0] scene       : ${SCENE_NAME}"
echo "[cleanup-quality-fuse-v0] profile     : ${CLEANUP_PROFILE}"

required_paths=(
  "${SCENE_ROOT}/images_2"
  "${SCENE_ROOT}/images_8"
  "${MIP_GROUP_ROOT}/${MIP_PARENT_NAME}/point_cloud/iteration_30000/point_cloud.ply"
  "${SCENE_ASSET_ROOT}/prepared_sr_priors/${STABLE_PRIOR_NAME}/fused_priors"
  "${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/k27_base/recovered_mip_model_lr_miphr_v1/point_cloud/iteration_31600/point_cloud.ply"
)

optional_key_paths=(
  "${MIP_GROUP_ROOT}/${MIP_RAW_NAME}"
  "${SCENE_ASSET_ROOT}/prepared_sr_priors/${QUALITY_PRIOR_NAME}/fused_priors"
  "${SCENE_ASSET_ROOT}/prepared_sr_priors/${QUALITY_PRIOR_SAFE_NAME}/fused_priors"
  "${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/k27_quality_fuse_directprior_warm_v0"
  "${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/k22_reg/pulled_mip_model/point_cloud/iteration_32000/point_cloud.ply"
  "${SOF_ROOT}/output/kitchen_mipsplatting_prior_repro/stablesr_mipsplatting_hrprior_finetune_stronger_34k_v1"
  "${SOF_ROOT}/output/sof_mesh_prepare_stage_compare_v0/${SCENE_NAME}"
  "${SOF_ROOT}/output/single_mesh_tangent_gaussian_probe_v0/${SCENE_NAME}/${GEOMETRY_PROBE_RUN_NAME}"
  "${SOF_ROOT}/output/candidate_layer_stack_v0/${SCENE_NAME}/${LAYER_STACK_RUN_NAME}"
  "${SOF_ROOT}/output/mask_guided_reparameterization_v0/${SCENE_NAME}/${MASK_REPARAM_RUN_NAME}"
)

echo
echo "[required inputs]"
missing_required=0
for path in "${required_paths[@]}"; do
  print_path "[required]" "${path}"
  if [[ ! -e "${path}" ]]; then
    missing_required=$((missing_required + 1))
  fi
done

echo
echo "[optional key assets]"
for path in "${optional_key_paths[@]}"; do
  print_path "[optional]" "${path}"
done

if [[ "${missing_required}" -gt 0 && "${ALLOW_MISSING_REQUIRED}" != "1" ]]; then
  echo
  echo "[cleanup-quality-fuse-v0] ${missing_required} required path(s) missing; aborting cleanup." >&2
  echo "[cleanup-quality-fuse-v0] Set ALLOW_MISSING_REQUIRED=1 only if you intentionally changed the pipeline inputs." >&2
  exit 1
fi

cleanup_named_children "${MIP_GROUP_ROOT}" "${KEEP_MIP_GROUP_DIRS}" "mip parent/input variants"
cleanup_named_children "${SCENE_ASSET_ROOT}/prepared_sr_priors" "${KEEP_PREPARED_PRIORS}" "prepared SR prior roots"
cleanup_named_children "${SOF_ROOT}/output/cleanup_mip_view_aligned_volume_artifacts_v0/${SCENE_NAME}" "${KEEP_CLEANUP_RUNS}" "cleanup start-model runs"
cleanup_named_children "${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}" "${KEEP_RECOVER_RUNS}" "recovery runs"
cleanup_named_children "${SOF_ROOT}/output/star_quarantine_v0/${SCENE_NAME}" "${KEEP_STAR_QUARANTINE_RUNS}" "star quarantine runs"
cleanup_named_children "${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}" "${KEEP_MIP_TO_SOF_RUNS}" "mip-to-sof regulation runs"
cleanup_named_children "${SOF_ROOT}/output/kitchen_mipsplatting_prior_repro" "${KEEP_MIPSPLATTING_PRIOR_RUNS}" "mipsplatting prior repro runs"

if [[ "${CLEAN_DIAG}" == "1" ]]; then
  cleanup_named_children "${SOF_ROOT}/output/diagnose_mip_closure_v0/${SCENE_NAME}" "${KEEP_DIAG_RUNS}" "diagnostic closure runs"
else
  echo
  echo "[skip] diagnostic closure runs; set CLEAN_DIAG=1 to delete non-kept diagnostics"
fi

echo
echo "[optional nested cleanup]"
cleanup_optional_path "${SCENE_ASSET_ROOT}/prepared_sr_priors/${QUALITY_PRIOR_NAME}/debug" "quality-fuse debug visualizations" "${CLEAN_DEBUG}"
cleanup_optional_path "${SCENE_ASSET_ROOT}/prepared_sr_priors/${QUALITY_PRIOR_SAFE_NAME}/debug" "safe quality-fuse debug visualizations" "${CLEAN_DEBUG}"

for run in ${KEEP_RECOVER_RUNS}; do
  cleanup_optional_path "${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${run}/recovered_mip_renders_no_gt_hr_v0" "recover render dump for kept run ${run}" "${CLEAN_RENDER_DUMPS}"
  cleanup_optional_path "${SOF_ROOT}/output/recover_cleaned_mip_lr_v0/${SCENE_NAME}/${run}/recovered_mip_renders_no_gt_v0" "recover render dump for kept run ${run}" "${CLEAN_RENDER_DUMPS}"
done

for run in ${KEEP_MIP_TO_SOF_RUNS}; do
  cleanup_optional_path "${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${run}/pulled_mip_renders_no_gt" "mip-to-sof pulled render dump for kept run ${run}" "${CLEAN_RENDER_DUMPS}"
  cleanup_optional_path "${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${run}/base_mip_renders_no_gt" "mip-to-sof base render dump for kept run ${run}" "${CLEAN_RENDER_DUMPS}"
  cleanup_optional_path "${SOF_ROOT}/output/mip_to_sof_surface_v0/${SCENE_NAME}/${run}/render_diff_vs_base" "mip-to-sof render diff dump for kept run ${run}" "${CLEAN_RENDER_DUMPS}"
done

if [[ "${CLEAN_UNKNOWN_TOPLEVEL_OUTPUT}" == "1" ]]; then
  echo
  echo "[cleanup-quality-fuse-v0] CLEAN_UNKNOWN_TOPLEVEL_OUTPUT=1 is intentionally not implemented."
  echo "[cleanup-quality-fuse-v0] Add explicit keep/delete rules before touching arbitrary output roots."
fi

echo
if [[ "${DELETE}" == "1" ]]; then
  echo "[cleanup-quality-fuse-v0] delete pass complete."
else
  echo "[cleanup-quality-fuse-v0] dry-run complete. Re-run with DELETE=1 CONFIRM=delete to remove listed candidates."
fi

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-/root/autodl-tmp/kitchen}"

RECOVER_RUN_NAME="${RECOVER_RUN_NAME:-view_aligned_volume_delete_v1_init_energy_curve_refit_v0_prune_more_v1_mip_hr_anchor_v0_miphr_v1}"
STARCURVE_RUN_NAME="${STARCURVE_RUN_NAME:-${RECOVER_RUN_NAME}_starcurve_v0}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-${STARCURVE_RUN_NAME}_surface_patch_v0}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-srtest}"

SCENE_NAME="${SCENE_NAME}" \
SCENE_ROOT="${SCENE_ROOT}" \
RECOVER_RUN_NAME="${RECOVER_RUN_NAME}" \
OUTPUT_RUN_NAME="${STARCURVE_RUN_NAME}" \
RUN_RENDER=0 \
CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
bash "${SCRIPT_DIR}/run_apply_starburst_curve_repair_v0_kitchen.sh"

SCENE_NAME="${SCENE_NAME}" \
SCENE_ROOT="${SCENE_ROOT}" \
SOURCE_RUN_NAME="${RECOVER_RUN_NAME}" \
MODEL_PATH="${SOF_ROOT}/output/post_cleanup_starburst_curve_repair_v0/${SCENE_NAME}/${STARCURVE_RUN_NAME}" \
RUN_NAME="${FINAL_RUN_NAME}" \
RUN_RENDER=1 \
CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
bash "${SCRIPT_DIR}/run_inject_mip_surface_patch_carriers_v0_kitchen.sh"

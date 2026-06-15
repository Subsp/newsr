#!/usr/bin/env bash
set -euo pipefail

# Same-size render-restoration prior wrapper.
# Use this when the source images are LR-trained 3DGS renders already at the
# target/reference resolution, so the prior model should restore x1 details
# instead of doing x4 image super-resolution.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SCENE_NAME="${SCENE_NAME:-kitchen}"
ENHANCEMENT_BACKEND="${ENHANCEMENT_BACKEND:-nafnet}"

SOURCE_IMAGES_SUBDIR="${SOURCE_IMAGES_SUBDIR:-${RENDER_RESTORATION_SOURCE_SUBDIR:-renders_lr_same_size}}"
REFERENCE_IMAGES_SUBDIR="${REFERENCE_IMAGES_SUBDIR:-images_2}"
FALLBACK_IMAGES_SUBDIR="${FALLBACK_IMAGES_SUBDIR:-${REFERENCE_IMAGES_SUBDIR}}"
TARGET_IMAGES_SUBDIR="${TARGET_IMAGES_SUBDIR:-${REFERENCE_IMAGES_SUBDIR}}"
PREPARE_IMAGES8="${PREPARE_IMAGES8:-0}"

RAW_PRIOR_SUBDIR="${RAW_PRIOR_SUBDIR:-render_x1_priors_${ENHANCEMENT_BACKEND}}"
PREPARED_SR_PRIOR_NAME="${PREPARED_SR_PRIOR_NAME:-render_x1_${ENHANCEMENT_BACKEND}_aligned_${REFERENCE_IMAGES_SUBDIR}_scratch_v0}"
PRIOR_ONLY_RUN_TAG="${PRIOR_ONLY_RUN_TAG:-mip30k_r1_renderx1_${ENHANCEMENT_BACKEND}_prioronly_scratch_v0}"
DISABLE_PRIOR_USABLE_MASKS="${DISABLE_PRIOR_USABLE_MASKS:-1}"

export SCENE_NAME
export ENHANCEMENT_BACKEND
export SOURCE_IMAGES_SUBDIR
export REFERENCE_IMAGES_SUBDIR
export FALLBACK_IMAGES_SUBDIR
export TARGET_IMAGES_SUBDIR
export PREPARE_IMAGES8
export RAW_PRIOR_SUBDIR
export PREPARED_SR_PRIOR_NAME
export PRIOR_ONLY_RUN_TAG
export DISABLE_PRIOR_USABLE_MASKS

bash "${SCRIPT_DIR}/run_mipsplatting_enhancement_prior_scratch_v0_kitchen.sh"

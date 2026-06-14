#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_TAG="${RUN_TAG:-mip30k_r1_qwen_hfseed_viewmajor_v11_hfsmall_hfshape}"

# v11 moves artifact prevention to birth time: prior HF seeds are initialized
# as view/hf-oriented flattened ellipsoids instead of isotropic balls. The long
# tangent axis follows the local prior-guidance high-frequency structure; the
# depth/view axis is thin, so dot-like residuals become flat surfels, not bubbles.
export PRIOR_BUBBLE_CLEANUP_ENABLE="${PRIOR_BUBBLE_CLEANUP_ENABLE:-0}"

export PRIOR_HF_SEED_SHAPE_MODE="${PRIOR_HF_SEED_SHAPE_MODE:-hf_oriented}"
export PRIOR_HF_SEED_SHAPE_LONG_RATIO="${PRIOR_HF_SEED_SHAPE_LONG_RATIO:-2.20}"
export PRIOR_HF_SEED_SHAPE_SHORT_RATIO="${PRIOR_HF_SEED_SHAPE_SHORT_RATIO:-0.55}"
export PRIOR_HF_SEED_SHAPE_NORMAL_RATIO="${PRIOR_HF_SEED_SHAPE_NORMAL_RATIO:-0.12}"
export PRIOR_HF_SEED_SHAPE_CONFIDENCE_POWER="${PRIOR_HF_SEED_SHAPE_CONFIDENCE_POWER:-0.65}"

export PRIOR_HF_SEED_SCALE_MODE="${PRIOR_HF_SEED_SCALE_MODE:-pixel}"
export PRIOR_HF_SEED_SCALE_MULTIPLIER="${PRIOR_HF_SEED_SCALE_MULTIPLIER:-0.42}"
export PRIOR_HF_SEED_MIN_PIXEL_RADIUS="${PRIOR_HF_SEED_MIN_PIXEL_RADIUS:-0.14}"
export PRIOR_HF_SEED_MAX_PIXEL_RADIUS="${PRIOR_HF_SEED_MAX_PIXEL_RADIUS:-0.54}"
export PRIOR_HF_SEED_MAX_PROVIDER_RATIO="${PRIOR_HF_SEED_MAX_PROVIDER_RATIO:-0.08}"
export PRIOR_HF_SEED_SCALE_CLAMP_ENABLE="${PRIOR_HF_SEED_SCALE_CLAMP_ENABLE:-1}"
export PRIOR_HF_SEED_SCALE_CLAMP_MAX_AXIS="${PRIOR_HF_SEED_SCALE_CLAMP_MAX_AXIS:-0.016}"
export PRIOR_HF_SEED_SCALE_CLAMP_MAX_ANISOTROPY="${PRIOR_HF_SEED_SCALE_CLAMP_MAX_ANISOTROPY:-18.0}"
export PRIOR_HF_SEED_DIAG_LARGE_SCALE="${PRIOR_HF_SEED_DIAG_LARGE_SCALE:-0.016}"

export PRIOR_HF_SEED_OPACITY="${PRIOR_HF_SEED_OPACITY:-0.26}"
export PRIOR_HF_SEED_JITTER_SCALE="${PRIOR_HF_SEED_JITTER_SCALE:-0.005}"
export PRIOR_HF_SEED_COLOR_RESIDUAL_GAIN="${PRIOR_HF_SEED_COLOR_RESIDUAL_GAIN:-3.0}"

bash "${SCRIPT_DIR}/run_mipsplatting_hf_seed_prior_v0_kitchen_viewmajor_hfsmall_carry.sh"

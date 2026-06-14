#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_TAG="${RUN_TAG:-mip30k_r1_qwen_hfseed_viewmajor_v5_hfsmall}"

# Same high-frequency-only supervision as v4_hfband, but force HF seeds to be
# born at screen-pixel scale instead of inheriting the nearest parent Gaussian's
# world scale. This tests whether prior texture can be carried by genuinely
# small Gaussian bodies rather than broad parent-scale splats.
export PRIOR_HF_SEED_SCALE_MODE="${PRIOR_HF_SEED_SCALE_MODE:-pixel}"
export PRIOR_HF_SEED_SCALE_MULTIPLIER="${PRIOR_HF_SEED_SCALE_MULTIPLIER:-0.65}"
export PRIOR_HF_SEED_MIN_PIXEL_RADIUS="${PRIOR_HF_SEED_MIN_PIXEL_RADIUS:-0.30}"
export PRIOR_HF_SEED_MAX_PIXEL_RADIUS="${PRIOR_HF_SEED_MAX_PIXEL_RADIUS:-0.90}"
export PRIOR_HF_SEED_MAX_PROVIDER_RATIO="${PRIOR_HF_SEED_MAX_PROVIDER_RATIO:-0.16}"
export PRIOR_HF_SEED_DIAG_LARGE_SCALE="${PRIOR_HF_SEED_DIAG_LARGE_SCALE:-0.020}"
export PRIOR_HF_SEED_JITTER_SCALE="${PRIOR_HF_SEED_JITTER_SCALE:-0.035}"
export PRIOR_HF_SEED_OPACITY="${PRIOR_HF_SEED_OPACITY:-0.20}"
export PRIOR_HF_SEED_MAX_PER_ITER="${PRIOR_HF_SEED_MAX_PER_ITER:-256}"
export PRIOR_HF_SEED_MAX_TOTAL="${PRIOR_HF_SEED_MAX_TOTAL:-196608}"
export PRIOR_HF_SEED_GUIDANCE_THRESHOLD="${PRIOR_HF_SEED_GUIDANCE_THRESHOLD:-0.055}"
export PRIOR_HF_SEED_PRUNE_PROTECT_ITERS="${PRIOR_HF_SEED_PRUNE_PROTECT_ITERS:-1000}"

bash "${SCRIPT_DIR}/run_mipsplatting_hf_seed_prior_v0_kitchen_viewmajor_hfband.sh"

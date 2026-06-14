#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_TAG="${RUN_TAG:-mip30k_r1_qwen_hfseed_prioronly_nomask_strong_v0}"

# Keep the residual GT-free and anchor-only, but make the injected prior branch visible.
export PRIOR_RESIDUAL_USE_MASKS="${PRIOR_RESIDUAL_USE_MASKS:-0}"
export PRIOR_RESIDUAL_HF_WEIGHT="${PRIOR_RESIDUAL_HF_WEIGHT:-3.0}"
export PRIOR_RESIDUAL_DELTA_CLIP="${PRIOR_RESIDUAL_DELTA_CLIP:-0.50}"
export PRIOR_EDGE_UPDATE_SCALE="${PRIOR_EDGE_UPDATE_SCALE:-3.0}"

# Seed much earlier and denser than the default probe run.
export PRIOR_HF_SEED_FROM_ITER="${PRIOR_HF_SEED_FROM_ITER:-30020}"
export PRIOR_HF_SEED_UNTIL_ITER="${PRIOR_HF_SEED_UNTIL_ITER:-31800}"
export PRIOR_HF_SEED_INTERVAL="${PRIOR_HF_SEED_INTERVAL:-20}"
export PRIOR_HF_SEED_MAX_PER_ITER="${PRIOR_HF_SEED_MAX_PER_ITER:-4096}"
export PRIOR_HF_SEED_MAX_TOTAL="${PRIOR_HF_SEED_MAX_TOTAL:-60000}"
export PRIOR_HF_SEED_SCALE_MULTIPLIER="${PRIOR_HF_SEED_SCALE_MULTIPLIER:-0.65}"
export PRIOR_HF_SEED_OPACITY="${PRIOR_HF_SEED_OPACITY:-0.18}"
export PRIOR_HF_SEED_JITTER_SCALE="${PRIOR_HF_SEED_JITTER_SCALE:-0.30}"

# Let the prior branch grow aggressively after seeds appear.
export DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-25}"
export DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.00002}"

bash "${SCRIPT_DIR}/run_mipsplatting_hf_seed_prior_v0_kitchen.sh"

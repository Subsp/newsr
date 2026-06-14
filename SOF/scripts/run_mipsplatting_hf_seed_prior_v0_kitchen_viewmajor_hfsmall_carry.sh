#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_TAG="${RUN_TAG:-mip30k_r1_qwen_hfseed_viewmajor_v9_hfsmall_carry}"

# v9 keeps v8's dense pixel-scale carriers, but compensates for alpha blending
# at birth: the seed DC color is initialized as anchor + gain * prior_residual.
# If prior HF uptake rises, the bottleneck was carrier amplitude rather than
# view pairing or loss weight.
export PRIOR_HF_SEED_COLOR_RESIDUAL_GAIN="${PRIOR_HF_SEED_COLOR_RESIDUAL_GAIN:-3.0}"

bash "${SCRIPT_DIR}/run_mipsplatting_hf_seed_prior_v0_kitchen_viewmajor_hfsmall_dense.sh"

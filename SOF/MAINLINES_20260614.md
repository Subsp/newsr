# Mainlines as of 2026-06-14

This note narrows the actively supported work down to two pipelines only.
Nothing else is deleted here, but anything not listed below should be treated
as legacy or exploratory code rather than a maintained mainline.

## 1. Canonical NoSR mainline

Canonical entrypoint:

- `scripts/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh`

Why this is the mainline:

- It explicitly encodes the original fixed-topology NoSR cleanup protocol that
  produced the stable `~28.57` PSNR result.
- The key reproduction setting is documented inline in the script:
  `TRAIN_IMAGES_SUBDIR=images_2`, `TRAIN_RESOLUTION=4`,
  `TARGET_IMAGES_SUBDIR=images_2`, `RENDER_RESOLUTION=1`.
- The supporting observation memo
  `20260606_kitchen_layer_frequency_observations_v0.md` records
  `psnr_nosr_cleanup_mean = 28.572666229446238` and shows this cleanup is the
  reliable gain we want to preserve.

What the script owns:

- Reuse or rebuild the surface-state payload from the input checkpoint and
  mesh via `scripts/run_classify_mip_surface_state_v0_kitchen.sh`.
- Run the fixed-topology layer-frequency NoSR cleanup in
  `mip-splatting/hybrid_sdfgs.train`.
- Render and summarize the post-cleanup metrics.

How to think about related scripts:

- `scripts/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh` is the
  maintained NoSR mainline.
- Scripts such as
  `run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen_srprior_lrmesh.sh`,
  `run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen_after_prioronly_scratch.sh`,
  and other wrappers are derived routes, not the canonical definition.

## 2. Canonical prior-from-scratch mainline

Canonical entrypoint:

- `scripts/run_mipsplatting_prior_only_from_scratch_v0_kitchen.sh`

What the script owns:

- Build a COLMAP-compatible alias scene using real camera geometry plus prior
  images through `scripts/prepare_colmap_prior_supervision_scene_v0.py`.
- Train mip-splatting from scratch on prior images.
- Render and summarize metrics against the real target views.

This is the mainline to preserve when we say "train a new model from priors
from scratch."

## 3. Enhancement-SR prior wrapper

Current supported wrapper for enhancement-style SR priors:

- `scripts/run_mipsplatting_enhancement_prior_scratch_v0_kitchen.sh`

What it adds on top of the prior-from-scratch mainline:

- Generate raw enhancement SR priors, currently with the `swinir` backend.
- Prepare the aligned SR prior cache.
- Call `run_mipsplatting_prior_only_from_scratch_v0_kitchen.sh`.
- Optionally continue into NoSR cleanup after the scratch prior model is
  trained.

In other words:

- `run_mipsplatting_prior_only_from_scratch_v0_kitchen.sh` defines the scratch
  training mainline.
- `run_mipsplatting_enhancement_prior_scratch_v0_kitchen.sh` is the supported
  way to feed enhancement SR priors into that mainline.

## 4. Practical keep set

If we extract a clean repo around the maintained flows, the conceptual keep set
is:

- NoSR cleanup mainline.
- Prior-from-scratch mainline.
- Enhancement-SR prior generation and alignment needed by the new mainline.
- The mip-splatting training code these scripts invoke.

Everything else can remain in the original repo as historical context, but it
is not part of the supported mainline definition.

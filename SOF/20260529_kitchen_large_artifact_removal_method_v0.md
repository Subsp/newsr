# 20260529 Kitchen Large-Artifact Removal Method v0

## Scope

This document records the current accepted method for removing **large volumetric / view-aligned artifacts** from the kitchen `mip30k` Gaussian field before later surface-aware processing.

This is **not** the final solution for small patchy noise. The current method is specifically for:

- large non-surface volumetric compensation Gaussians
- view-aligned volume artifacts
- oversized / stressed Gaussians that break later surface annealing

The current method should be treated as:

`hard cleanup for large artifacts -> pure-mip recovery -> conservative surface covariance anneal`

## Current accepted principle

The current working rule is:

1. Remove the worst large artifacts first with the old `volume_stress` cleanup baseline.
2. Recover render quality with a **pure mip-only** recovery stage.
3. Only then run conservative covariance surface annealing.
4. Do **not** mix SOF-derived SR priors into this baseline unless explicitly testing that variant.

The most important practical lesson so far is:

- large-artifact cleanup and later surface anneal are useful
- but if recovery mixes in SOF-style priors, the output can take on a misleading "SOF-like" appearance
- therefore the baseline documented here is explicitly the **pure-mip** version

## Main code paths

### Large-artifact cleanup

- [cleanup_mip_view_aligned_volume_artifacts_v0.py](/Users/ltl/Desktop/VGGTSR/SOF/cleanup_mip_view_aligned_volume_artifacts_v0.py)
- [scripts/run_cleanup_mip_view_aligned_volume_artifacts_v0_kitchen.sh](/Users/ltl/Desktop/VGGTSR/SOF/scripts/run_cleanup_mip_view_aligned_volume_artifacts_v0_kitchen.sh)

Important logic:

- `collect_view_aligned_stats(...)`
- `build_view_aligned_prune_mask(...)`

The prune mask is built from a combination of:

- view alignment
- anisotropy / filter inflation
- footprint size
- volume radius / effective scale
- short-axis / stress statistics
- optional mesh surface distance

The cleanup payload is exported as:

- `view_aligned_volume_cleanup_payload.pt`

### Recovery after cleanup

- [recover_cleaned_mip_lr_v0.py](/Users/ltl/Desktop/VGGTSR/SOF/recover_cleaned_mip_lr_v0.py)
- [scripts/run_recover_cleaned_mip_lr_v0_kitchen.sh](/Users/ltl/Desktop/VGGTSR/SOF/scripts/run_recover_cleaned_mip_lr_v0_kitchen.sh)

This stage restores image quality after hard delete.

Important constraint for the baseline in this document:

- `USE_SR_PRIOR=0`

Why this matters:

- the recovery runner normally defaults to `USE_SR_PRIOR=1`
- when enabled, it points to a prepared prior cache rooted at `prepared_sr_priors/sof_surface_v0_...`
- this can make the recovered result visually drift toward SOF-like behavior

For the current large-artifact baseline, that is considered contamination and should be disabled.

### Conservative surface anneal

- [anneal_mip_covariance_to_surface_v0.py](/Users/ltl/Desktop/VGGTSR/SOF/anneal_mip_covariance_to_surface_v0.py)

This stage does **not** attempt to solve all artifact problems. It only:

- thins mesh-supported Gaussians
- aligns covariance more closely to the mesh tangent plane
- keeps render perturbation small with guards

By design, it leaves these unchanged:

- centers
- opacity
- DC / SH color
- unsupported residual Gaussians

This means it is expected to help with surface alignment, but not fully remove small compensation noise.

### One-shot pipeline runner

- [scripts/run_cleanup_recover_surface_anneal_v0_kitchen.sh](/Users/ltl/Desktop/VGGTSR/SOF/scripts/run_cleanup_recover_surface_anneal_v0_kitchen.sh)

This script now serves as the main reproduction entry for the current baseline.

Important defaults in the current version:

- `RECOVER_PROFILE=conservative_v0`
- `RECOVER_USE_SR_PRIOR=0`
- `ANNEAL_MODEL_ITERATION=-1`

These defaults were added to avoid two failure modes:

- accidentally mixing SOF-derived priors into recovery
- accidentally annealing the wrong checkpoint iteration instead of the latest recovered model

## Detailed method

### Stage 1: hard cleanup of large volumetric artifacts

Use the `volume_stress` cleanup path as the baseline.

Current intent:

- delete the worst large view-aligned / volumetric Gaussians
- avoid treating small patchy noise as the main target

Current representative settings are around:

- `CANDIDATE_MODE=volume_stress`
- `DELETE_QUANTILE=0.930`
- `MAX_PRUNE_FRACTION=0.120`

The core deletion score is computed inside:

- [cleanup_mip_view_aligned_volume_artifacts_v0.py](/Users/ltl/Desktop/VGGTSR/SOF/cleanup_mip_view_aligned_volume_artifacts_v0.py)

and assembled in:

- `build_view_aligned_prune_mask(...)`

Conceptually, this stage targets Gaussians that are:

- too large
- too view-aligned
- too anisotropic or filter-inflated
- too stress-sensitive under the cleanup heuristics

This is currently the main accepted baseline for removing **large** artifacts.

### Stage 2: pure-mip recovery

After hard cleanup, run a recovery stage to restore render quality without reintroducing the deleted large artifact cloud.

For the baseline documented here:

- use `REFINE_PROFILE=conservative_v0`
- keep `USE_SR_PRIOR=0`

This means recovery is anchored by the original mip model, but does not inject SOF-derived SR priors.

The purpose of this stage is:

- recover low-frequency render quality
- reduce over-cleaning damage
- keep the cleaned model visually plausible before anneal

This is the stage that prevents the cleanup baseline from being judged only by how clean the deleted subset looks.

### Stage 3: conservative covariance anneal

Run surface covariance anneal on the recovered model.

Current role of this stage:

- make mesh-supported Gaussians thinner and more surface-like
- preserve render quality with self-guarding
- avoid large center / color / opacity changes

Important internal guards include:

- volume-artifact guard
- optional counterfactual veto support
- render-delta self-guard

The important interpretation is:

- this stage improves the geometry style of supported Gaussians
- it does **not** remove all remaining residual compensation structure

So if the final output still has small patchy noise, that is not automatically a failure of anneal; it often means the remaining problem lies in residual Gaussians that were intentionally left untouched.

## Current reproduction command

This is the current pure-mip baseline command:

```bash
cd ~/autodl-tmp/SOFSR
git fetch origin
git checkout codex/prior-fusion-v0
git pull origin codex/prior-fusion-v0

SCENE_ROOT=/root/autodl-tmp/kitchen \
MIP_MODEL_PATH=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/kitchen_mip_vanilla_images8_v1/mip30k \
MESH_PATH=/root/autodl-tmp/kitchen_MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_15_cleaned_mesh.ply \
RECOVER_USE_SR_PRIOR=0 \
RECOVER_PROFILE=conservative_v0 \
CLEANUP_RUN_NAME=mip30k_volume_stress_iterclean_v0 \
RECOVER_RUN_NAME=mip30k_volume_stress_iterclean_v0_conservative_v0_puremip \
ANNEAL_RUN_NAME=mip30k_volume_stress_iterclean_v0_conservative_v0_puremip_surfaceanneal_v0 \
ANNEAL_MODEL_ITERATION=-1 \
bash scripts/run_cleanup_recover_surface_anneal_v0_kitchen.sh
```

## Current output naming

Representative outputs for the current baseline are:

- cleanup model:
  - `/root/autodl-tmp/SOFSR/output/cleanup_mip_view_aligned_volume_artifacts_v0/kitchen/mip30k_volume_stress_iterclean_v0/cleaned_mip_model_view_volume_v1`
- recovered model:
  - `/root/autodl-tmp/SOFSR/output/recover_cleaned_mip_lr_v0/kitchen/mip30k_volume_stress_iterclean_v0_conservative_v0_puremip/recovered_mip_model_lr_v0`
- annealed model:
  - `/root/autodl-tmp/SOFSR/output/anneal_mip_covariance_to_surface_v0/kitchen/mip30k_volume_stress_iterclean_v0_conservative_v0_puremip_surfaceanneal_v0`

## What this method solves well

This method is currently best at:

- clearing obvious large-block volumetric artifacts
- making the field safer to use for later surface-oriented processing
- avoiding the earlier failure mode where surface anneal was applied directly to a badly contaminated mip field

## What this method does not solve yet

This method does **not** fully solve:

- small patchy noise
- residual compensation speckles
- all local over/under-cleaning tradeoffs

Those remaining issues likely need a later stage based on:

- soft suppression
- LR/SR-gated residual cleanup
- short recovery after suppression

Relevant future code paths for that next step are:

- [cleanup_mip_blur_artifacts_v0.py](/Users/ltl/Desktop/VGGTSR/SOF/cleanup_mip_blur_artifacts_v0.py)
- [global_refine_sof_v0.py](/Users/ltl/Desktop/VGGTSR/SOF/global_refine_sof_v0.py)

They are not part of the current large-artifact baseline, but are the natural next direction once the large volumetric cloud has already been reduced.

## Summary

The current accepted method for removing large artifacts is:

`volume_stress hard cleanup -> pure-mip conservative recovery -> conservative surface covariance anneal`

The key operational rule is:

- keep this baseline pure-mip
- do not silently mix SOF-derived priors into recovery
- do not expect anneal alone to remove the remaining small residual noise

For now, this is the correct documented baseline for the "remove large artifacts first" stage.

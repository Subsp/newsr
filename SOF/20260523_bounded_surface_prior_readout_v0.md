# 20260523 Bounded Surface Prior Readout v0

## Purpose

This record tracks the current SOFGS prior-injection line after the shift from direct SR-prior loss to a bounded surface memory / readout formulation.

Current main question:

```text
Can SR prior information enter SOFGS through a safe color-only residual target,
before opening opacity, geometry, SH, or HF channels?
```

Current answer:

```text
Yes, but only weakly so far.

trust_surface residual A-only gives a small but consistent positive render response
against SR prior. Larger residual clip, low/mid residual, and a simple residual
strength gate did not improve over raw residual clip 0.05.

The next most useful validation is now an oracle surface-domain ablation:
run the same residual surface-memory readout on LR mesh, GT reconstruction mesh,
and LR mesh with stricter confidence gating.
```

## Current Branch And Commits

Branch:

```text
codex/prior-fusion-v0
```

Relevant recent commits:

```text
54dad14 Add residual-clipped surface memory targets
45af8a0 Audit residual surface memory targets
9a87e6f Document bounded surface prior readout status
ea4c3de Add lowmid surface residual aggregation
1235ff6 Add residual strength gate for surface memory
```

## Main Code Paths

Training / bounded surface memory:

```text
train_bounded_surface_alternating_v0.py
scripts/run_bounded_surface_alternating_v0_kitchen.sh
scripts/run_bounded_surface_alternating_mainline5k_safe_v1_kitchen.sh
scripts/run_bounded_surface_mesh_oracle_ablation_v0_kitchen.sh
```

Audit / readout diagnostics:

```text
scripts/audit_surface_memory_readout_v0.py
scripts/run_audit_surface_memory_readout_v0_kitchen.sh
```

Static surface smoothing pre-pass:

```text
scripts/smooth_surface_bound_gaussians_v0.py
scripts/run_surface_smooth_filter_v0_kitchen.sh
```

Render check:

```text
render.py
```

## Important Inputs

Server repo:

```text
/root/autodl-tmp/SOFSR
```

Conda environment:

```text
srtest
```

Scene root:

```text
/root/autodl-tmp/kitchen
```

Image scale used for this line:

```text
images_2
```

Base 34k checkpoint before smoothing:

```text
/root/autodl-tmp/SOFSR/output/mask_guided_reparameterization_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0
iteration: 34000
```

Current preferred start checkpoint after body-aware surface smoothing:

```text
/root/autodl-tmp/SOFSR/output/surface_smooth_filter_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_delta_sigma_e3_delta_sigma_bodyaware_stronger_v1
iteration: 34000
```

SOF mesh:

```text
/root/autodl-tmp/SOFSR/output/sof_mesh_prepare_stage_compare_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_prepare_stage_sof_export_mesh_v0_debug_stage_00b3_after_scale_canonicalize_7.ply
```

Current SR prior root:

```text
/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors
```

Current SR anchor root:

```text
/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors
```

Important note:

```text
The current prior root has prior_mask_mean = 1.0 in audit.
So it behaves like unmasked/all-pixel prior for the readout audit.
This likely dilutes useful residual signal.
```

## Output Paths Used So Far

Static body-aware surface smoothing:

```text
/root/autodl-tmp/SOFSR/output/surface_smooth_filter_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_delta_sigma_e3_delta_sigma_bodyaware_stronger_v1
```

Trust-surface absolute A-only baseline:

```text
/root/autodl-tmp/SOFSR/output/bounded_surface_alternating_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_v0
```

Trust-surface residual A-only, clip 0.05:

```text
/root/autodl-tmp/SOFSR/output/bounded_surface_alternating_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_v0
```

Trust-surface residual A-only, clip 0.08:

```text
/root/autodl-tmp/SOFSR/output/bounded_surface_alternating_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_clip008_v0
```

High-confidence gate long run:

```text
/root/autodl-tmp/SOFSR/output/bounded_surface_alternating_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_confhigh_v0
```

Audit output location pattern:

```text
<after_model_path>/audit_surface_memory_readout_v0/surface_memory_readout_audit_v0_summary.json
```

Model output checkpoint pattern:

```text
<after_model_path>/point_cloud/iteration_35000/point_cloud.ply
```

## Implemented Mechanism

The current bounded surface memory path does the following:

```text
1. Build bounded surface coordinates for existing SOFGS.
2. Select memory-eligible GS, currently trust_surface only.
3. Project selected GS to all SR-prior views.
4. Aggregate a robust surface memory target.
5. A-step updates features_dc only.
6. Geometry, opacity, scale, rotation, rest-SH are frozen.
7. Audit checks whether DC changes are visible in final render.
```

Key current training options:

```text
MEMORY_ELIGIBILITY=trust_surface
BASE_SUPPORT_MODE=floor
BASE_SUPPORT_FLOOR=0.25
AGGREGATION_RESIDUAL_SAMPLE_CLIP=0.12
APPEARANCE_TARGET_MODE=residual_clipped
APPEARANCE_RESIDUAL_CLIP=0.05
APPEARANCE_RESIDUAL_SCALE=1.0
PHASE_MODE=appearance_only
```

Residual target semantics:

```text
memory_target = robust aggregate of clipped prior/base samples
residual      = memory_target - memory_base
A target      = DC_init + clip(residual, +/- APPEARANCE_RESIDUAL_CLIP)
```

This is intentionally different from the older absolute target:

```text
old: DC -> memory_target
new: DC -> DC_init + clipped(memory_target - memory_base)
```

## Results So Far

### 1. Broad selected A-only was not clean enough

Earlier selected set:

```text
selected_count = 889665
trust_surface  = 256323
trust_loose    = 235395
trust_outlier  = 537840
```

Conclusion:

```text
selected was too broad for surface memory.
Large outlier/loose population made surface-memory semantics ambiguous.
```

### 2. trust_surface absolute A-only proved A-step works, but target was wrong

Representative audit:

```text
active_visible_ratio                 = 0.872
target_minus_base_l1_active mean      = 0.0130
target_minus_before_dc_l1_active mean = 0.0952
dc_update_l1_active mean              = 0.0244
prior_l1_improvement_masked mean      = -0.00051
```

Conclusion:

```text
A-step can change DC and active GS are mostly visible.
But absolute memory target was too base-like and did not improve prior L1.
Do not open opacity or geometry based on this result.
```

### 3. trust_surface residual A-only, clip 0.05 is the current best baseline

Run:

```text
debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_v0
```

Important audit:

```text
appearance_target_minus_before_dc_l1_active mean = 0.01125
after_dc_minus_appearance_target_l1_active mean  = 0.00145
dc_update_l1_active mean                         = 0.00980
active_visible_ratio                             = 0.873
render_delta_l1 mean                             = 0.00163
prior_l1_before_masked mean                      = 0.04375
prior_l1_after_masked mean                       = 0.04331
prior_l1_improvement_masked mean                 = +0.000446
```

Interpretation:

```text
This is the first clean positive readout.
The gain is small, but all audited views improved against SR prior.
This supports residual target over absolute target.
```

### 4. clip 0.08 did not improve over clip 0.05

Run:

```text
debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_clip008_v0
```

Important audit:

```text
appearance_residual_clip                         = 0.08
appearance_target_minus_before_dc_l1_active mean = 0.00960
after_dc_minus_appearance_target_l1_active mean  = 0.000091
dc_update_l1_active mean                         = 0.00951
active_visible_ratio                             = 0.903
render_delta_l1 mean                             = 0.00164
prior_l1_improvement_masked mean                 = +0.000404
haze_positive_lowfreq mean                       = 0.000570
```

Interpretation:

```text
clip 0.08 is not better than clip 0.05.
It did not obviously haze out, but the prior-L1 gain is slightly lower.
So current bottleneck is probably not residual clip strength.
```

### 5. High-confidence long run is not directly comparable

Run:

```text
debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_confhigh_v0
```

Observed in training summary:

```text
cycle20
appearance before/after approximately 8.4e-5 -> 8.3e-5
appearance_targets valid = 111274
geometry steps = 0
color_stable = 65085
```

Interpretation:

```text
This run went to 20 cycles / 5000 A-only steps.
It is too long for direct comparison with the 1000-step residual tests.
It mainly shows the target can be fit to saturation.
Future comparison runs should fix TOTAL_STEPS=1000 and APPEARANCE_STEPS=500.
```

### 6. Low/mid residual did not beat raw residual

Run:

```text
debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_lowmid_v0
```

Important audit:

```text
appearance_target_minus_before_dc_l1_active mean = 0.00853
dc_update_l1_active mean                         = 0.00758
active_visible_ratio                             = 0.905
render_delta_l1 mean                             = 0.00149
prior_l1_improvement_masked mean                 = +0.000386
```

Interpretation:

```text
Low/mid residual is safer/smaller, but it did not improve prior-L1 over raw clip 0.05.
So the next bottleneck is not simply high-frequency noise in the DC residual.
The likely issue is that prior_mask_mean = 1.0 makes the residual source too broad.
```

### 7. Residual strength gate did not beat raw residual

Run:

```text
debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_gate001_v0
```

Important audit:

```text
AGGREGATION_RESIDUAL_MIN_L1                     = 0.01
dc_update_l1_active mean                         = 0.01015
active_visible_ratio                             = 0.928
render_delta_l1 mean                             = 0.00182
prior_l1_improvement_masked mean                 = +0.000405
```

Interpretation:

```text
The gate increased active visibility and render movement, but did not improve
SR-prior L1 over the raw residual clip 0.05 baseline.
So residual magnitude alone is not the right confidence mask.
```

## Current Conclusions

Use these as the current working assumptions:

```text
1. trust_surface gating is necessary and useful.
2. A-step/DC optimization path is working.
3. Active memory GS are visible enough for readout; visibility is not the first bottleneck.
4. Absolute surface memory target is the wrong target for this stage.
5. Residual-clipped target is directionally correct and gives small positive render readout.
6. Increasing clip from 0.05 to 0.08 does not improve prior-L1 response.
7. Low/mid residual does not improve over raw residual.
8. Residual strength gating does not improve over raw residual.
9. The next bottleneck is likely either surface-domain quality or prior/readout quality.
```

Do not do yet:

```text
do not open opacity readout
do not re-enable G-step
do not open rest-SH
do not add HF prior
do not increase total training steps as the next move
```

## Surface Mesh Oracle Next Step

Goal:

```text
Run the same surface-memory residual A-only readout with different surface domains:

A. LR SOFmesh
B. GT reconstruction mesh
C. LR SOFmesh + stricter confidence gate
```

Interpretation:

```text
If GT mesh improves prior_l1_improvement/readout clearly:
  main bottleneck is surface-domain quality.

If GT mesh does not improve:
  main bottleneck is prior multi-view consistency, prior masks, or readout/coverage.

If LR confidence gate improves:
  LR surface is usable, but memory eligibility/confidence is still too broad.
```

Implementation entry:

```text
scripts/run_bounded_surface_mesh_oracle_ablation_v0_kitchen.sh
```

Supported modes:

```text
ORACLE_MODE=lr_binding  # current LR mesh baseline through the common wrapper
ORACLE_MODE=gt_binding  # same LR-smoothed start, but GT mesh for bounded binding/memory
ORACLE_MODE=gt_smooth   # GT mesh for static smoothing and bounded binding/memory
ORACLE_MODE=lr_conf     # LR mesh with stricter confidence/disagreement gate
```

Notes:

```text
GT_MESH_PATH is intentionally not hardcoded.
Set it on the server to the reconstruction mesh path before running GT modes.

The wrapper uses the current best readout setting by default:
  raw residual
  APPEARANCE_RESIDUAL_CLIP=0.05
  BASE_SUPPORT_MODE=floor
  BASE_SUPPORT_FLOOR=0.25
  MEMORY_ELIGIBILITY=trust_surface
  PHASE_MODE=appearance_only
  TOTAL_STEPS=1000
```

## Low/Mid Residual Option

Low/mid residual target is the next experiment after the raw residual tests:

```text
r = lowmid(prior) - lowmid(base)
```

instead of raw RGB residual:

```text
r = prior - base
```

Reason:

```text
DC is a low-frequency appearance channel.
Raw SR residual contains unstable high-frequency residuals that can cancel across views or become noise.
The current positive but tiny readout suggests the residual direction is correct, but the residual source is not clean enough.
```

Implementation entry:

```text
AGGREGATION_RESIDUAL_BAND=raw|lowmid
AGGREGATION_RESIDUAL_LOWPASS_KERNEL=5
```

When `lowmid` is enabled, aggregation samples low-passed prior/base/anchor images before forming:

```text
memory_target - memory_base
```

This keeps the existing A-step target form:

```text
DC_init + clipped(memory_target - memory_base)
```

Raw residual remains available for comparison:

```text
AGGREGATION_RESIDUAL_BAND=raw
```

First low/mid test should keep:

```text
MEMORY_ELIGIBILITY=trust_surface
APPEARANCE_RESIDUAL_CLIP=0.05
BASE_SUPPORT_MODE=floor
BASE_SUPPORT_FLOOR=0.25
TOTAL_STEPS=1000
APPEARANCE_STEPS=500
PHASE_MODE=appearance_only
AGGREGATION_RESIDUAL_BAND=lowmid
AGGREGATION_RESIDUAL_LOWPASS_KERNEL=5
```

## Repro Commands

### Pull latest code

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest
git pull --ff-only origin codex/prior-fusion-v0
```

### Mesh oracle ablation

Use the same residual A-only readout for all arms. The wrapper runs the readout
audit automatically by default.

LR mesh baseline through the common oracle wrapper:

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest

ORACLE_MODE=lr_binding \
bash /root/autodl-tmp/SOFSR/scripts/run_bounded_surface_mesh_oracle_ablation_v0_kitchen.sh
```

GT mesh binding-only oracle:

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest

GT_MESH_PATH=/root/path/to/gt_reconstruction_mesh.ply \
ORACLE_MODE=gt_binding \
bash /root/autodl-tmp/SOFSR/scripts/run_bounded_surface_mesh_oracle_ablation_v0_kitchen.sh
```

GT mesh full surface-domain oracle:

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest

GT_MESH_PATH=/root/path/to/gt_reconstruction_mesh.ply \
ORACLE_MODE=gt_smooth \
bash /root/autodl-tmp/SOFSR/scripts/run_bounded_surface_mesh_oracle_ablation_v0_kitchen.sh
```

LR mesh + stricter confidence gate:

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest

ORACLE_MODE=lr_conf \
bash /root/autodl-tmp/SOFSR/scripts/run_bounded_surface_mesh_oracle_ablation_v0_kitchen.sh
```

Expected output run names:

```text
debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_oracle_lrmesh_binding_residual_v0
debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_oracle_gtmesh_binding_residual_v0
debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_oracle_gtmesh_smooth_residual_v0
debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_oracle_lrmesh_conf_residual_v0
```

### Current best residual A-only baseline

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest

PREPARED_SR_PRIOR_NAME=sof_surface_v0_images_8_to_images_2_mask0.12_soft \
SR_PRIOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors \
SR_ANCHOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors \
SR_PRIOR_MASK_DIR= \
SR_PRIOR_CONSISTENCY_THRESHOLD=0.0 \
RUN_NAME=debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_v0 \
PHASE_MODE=appearance_only \
TOTAL_STEPS=1000 \
APPEARANCE_STEPS=500 \
MEMORY_ELIGIBILITY=trust_surface \
BASE_SUPPORT_MODE=floor \
BASE_SUPPORT_FLOOR=0.25 \
AGGREGATION_RESIDUAL_SAMPLE_CLIP=0.12 \
APPEARANCE_TARGET_MODE=residual_clipped \
APPEARANCE_RESIDUAL_CLIP=0.05 \
APPEARANCE_RESIDUAL_SCALE=1.0 \
START_MODEL_PATH=/root/autodl-tmp/SOFSR/output/surface_smooth_filter_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_delta_sigma_e3_delta_sigma_bodyaware_stronger_v1 \
START_ITERATION=34000 \
OUTPUT_ITERATION=35000 \
MAX_VIEWS=0 \
SAVE_FINAL_MEMORY=1 \
bash /root/autodl-tmp/SOFSR/scripts/run_bounded_surface_alternating_mainline5k_safe_v1_kitchen.sh
```

### Low/mid residual A-only test

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest

PREPARED_SR_PRIOR_NAME=sof_surface_v0_images_8_to_images_2_mask0.12_soft \
SR_PRIOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors \
SR_ANCHOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors \
SR_PRIOR_MASK_DIR= \
SR_PRIOR_CONSISTENCY_THRESHOLD=0.0 \
RUN_NAME=debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_lowmid_v0 \
PHASE_MODE=appearance_only \
TOTAL_STEPS=1000 \
APPEARANCE_STEPS=500 \
MEMORY_ELIGIBILITY=trust_surface \
BASE_SUPPORT_MODE=floor \
BASE_SUPPORT_FLOOR=0.25 \
AGGREGATION_RESIDUAL_BAND=lowmid \
AGGREGATION_RESIDUAL_LOWPASS_KERNEL=5 \
AGGREGATION_RESIDUAL_SAMPLE_CLIP=0.12 \
APPEARANCE_TARGET_MODE=residual_clipped \
APPEARANCE_RESIDUAL_CLIP=0.05 \
APPEARANCE_RESIDUAL_SCALE=1.0 \
START_MODEL_PATH=/root/autodl-tmp/SOFSR/output/surface_smooth_filter_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_delta_sigma_e3_delta_sigma_bodyaware_stronger_v1 \
START_ITERATION=34000 \
OUTPUT_ITERATION=35000 \
MAX_VIEWS=0 \
SAVE_FINAL_MEMORY=1 \
bash /root/autodl-tmp/SOFSR/scripts/run_bounded_surface_alternating_mainline5k_safe_v1_kitchen.sh
```

### Raw residual support-gated A-only test

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest

PREPARED_SR_PRIOR_NAME=sof_surface_v0_images_8_to_images_2_mask0.12_soft \
SR_PRIOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors \
SR_ANCHOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors \
SR_PRIOR_MASK_DIR= \
SR_PRIOR_CONSISTENCY_THRESHOLD=0.0 \
RUN_NAME=debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_gate001_v0 \
PHASE_MODE=appearance_only \
TOTAL_STEPS=1000 \
APPEARANCE_STEPS=500 \
MEMORY_ELIGIBILITY=trust_surface \
BASE_SUPPORT_MODE=floor \
BASE_SUPPORT_FLOOR=0.25 \
AGGREGATION_RESIDUAL_BAND=raw \
AGGREGATION_RESIDUAL_MIN_L1=0.01 \
AGGREGATION_RESIDUAL_MAX_L1=0.0 \
AGGREGATION_RESIDUAL_SAMPLE_CLIP=0.12 \
APPEARANCE_TARGET_MODE=residual_clipped \
APPEARANCE_RESIDUAL_CLIP=0.05 \
APPEARANCE_RESIDUAL_SCALE=1.0 \
START_MODEL_PATH=/root/autodl-tmp/SOFSR/output/surface_smooth_filter_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_delta_sigma_e3_delta_sigma_bodyaware_stronger_v1 \
START_ITERATION=34000 \
OUTPUT_ITERATION=35000 \
MAX_VIEWS=0 \
SAVE_FINAL_MEMORY=1 \
bash /root/autodl-tmp/SOFSR/scripts/run_bounded_surface_alternating_mainline5k_safe_v1_kitchen.sh
```

### Audit a run

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest

AFTER_RUN_NAME=debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_v0 \
AFTER_ITERATION=35000 \
PREPARED_SR_PRIOR_NAME=sof_surface_v0_images_8_to_images_2_mask0.12_soft \
SR_PRIOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors \
SR_ANCHOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/sof_surface_v0_images_8_to_images_2_mask0.12_soft/priors \
bash /root/autodl-tmp/SOFSR/scripts/run_audit_surface_memory_readout_v0_kitchen.sh
```

### Render a run

```bash
cd /root/autodl-tmp/SOFSR
conda activate srtest

python render.py \
  -m /root/autodl-tmp/SOFSR/output/bounded_surface_alternating_v0/kitchen/debug_stage_00b3_after_scale_canonicalize_geometry_only_v0_bounded_aonly_trust_surface_residual_v0 \
  -s /root/autodl-tmp/kitchen \
  -i images_2 \
  --init_type sfm \
  --iteration 35000 \
  --eval \
  --skip_train \
  --data_device cpu
```

## Metrics To Watch

Primary:

```text
prior_l1_improvement_masked
render_delta_l1
appearance_target_minus_before_dc_l1_active
after_dc_minus_appearance_target_l1_active
```

Safety:

```text
haze_positive_lowfreq
haze_abs_lowfreq
visual render comparison
```

Do not over-interpret:

```text
alpha_delta_l1
```

Reason:

```text
opacity/tau tensors are unchanged in A-only runs, but render_simple alpha audit still reports nonzero alpha_delta.
Use opacity_delta_all and tau_delta_all as the source of truth for whether opacity changed.
```

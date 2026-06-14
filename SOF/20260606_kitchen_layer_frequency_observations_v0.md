# 20260606 Kitchen Layer-Frequency Observations v0

## Scope

This note records a useful intermediate kitchen result from the mip-splatting StableSR prior experiments:

```text
qwen244_layerfreq_nonsurface_hfbottleneck_v0
```

This run is worth preserving because the separated surface layer became visually meaningful on its own: the `surface_carrier` layer render is now close to the full scene appearance, instead of looking like an empty or purely noisy shell.

## Repository State

Training-side support for this run came from:

- mip-splatting commit `f3b40fe`: `Add layer-frequency surface closure regularizer`
- SOF wrapper commit `bf1c522`: `Expose layer-frequency prior controls`

Diagnostic rendering fixes added after the run:

- SOF commit `71e22ef`: `Render surface-state layers in mip-compatible mode`
- SOF commit `0743889`: `Flatten surface-state layer debug renders`

Follow-up code after this observation, not part of this recorded run:

- mip-splatting commit `83ba195`: `Add non-surface RGB energy bottleneck`
- SOF commit `1ce6b5b`: `Expose non-surface RGB energy control`

## Code Implementation Logic

This section records the exact intended logic of the code version that produced the useful intermediate result.

### Training Entry Points

SOF wrapper:

```text
/root/autodl-tmp/SOFSR/scripts/run_mipsplatting_stablesr_prior_scene.sh
/Users/ltl/Desktop/VGGTSR/SOF/scripts/run_mipsplatting_stablesr_prior_scene.sh
```

Underlying training script:

```text
/root/autodl-tmp/mip-splatting/hybrid_sdfgs/train.py
/Users/ltl/Desktop/VGGTSR/mip-splatting/hybrid_sdfgs/train.py
```

The wrapper exposes layer-frequency controls through environment variables and forwards them to `hybrid_sdfgs.train`.

The important variables for this recorded run were:

```text
LAYER_FREQUENCY_MASK_PAYLOAD
LAYER_FREQUENCY_NON_SURFACE_KEY=non_surface_active
LAYER_FREQUENCY_SURFACE_KEY=surface_carrier
LAMBDA_NON_SURFACE_HF=0.03
LAMBDA_SURFACE_HF_CLOSURE=0.02
SURFACE_HF_UPDATE_SCALE=0.5
SURFACE_NORMAL_LOCK=1
DENSIFY_UNTIL_ITER=0
```

### Gaussian Subset Rendering

The training implementation adds `GaussianSubsetView` in:

```text
mip-splatting/hybrid_sdfgs/train.py
```

Purpose:

- create a lightweight view over a selected subset of the current Gaussian model
- keep tensors differentiable with respect to the original Gaussian parameters
- allow rendering only `non_surface_active` or other selected masks without cloning/detaching the model

Important exposed properties/methods:

```text
get_xyz
get_features
get_scaling
get_scaling_with_3D_filter
get_opacity
get_opacity_with_3D_filter
get_rotation
filter_3D
get_covariance(...)
get_view2gaussian(...)
```

This is important because the mip renderer needs the prefiltered scale/opacity path and may need `filter_3D` or view-to-Gaussian data depending on the rasterizer path.

### Layer-Frequency Regularizer

The main class is:

```text
LayerFrequencyRegularizer
```

It receives two masks from the surface-state payload:

```text
non_surface_mask = payload[LAYER_FREQUENCY_NON_SURFACE_KEY]
surface_mask = payload[LAYER_FREQUENCY_SURFACE_KEY]
```

For this run:

```text
non_surface_mask = non_surface_active
surface_mask = surface_carrier
```

The regularizer has two distinct pieces.

#### Non-Surface High-Frequency Bottleneck

The selected non-surface subset is rendered on black background:

```text
I_ns = render(non_surface_subset, black_background)
```

The loss is:

```text
L_ns_hf = mean(abs(Laplacian(I_ns)))
```

and the weighted term is added directly into the main training loss:

```text
L_total += lambda_non_surface_hf * L_ns_hf
```

For the recorded run:

```text
lambda_non_surface_hf = 0.03
```

Interpretation:

- this discourages `non_surface_active` from carrying sharp image detail
- it does not prevent `non_surface_active` from carrying low-frequency or mid-frequency appearance
- this explains why `near_surface_uncertain_raw_black.png` can still contain substantial scene information

Recorded-version limitation:

- the mip base renderer path primarily returns RGB render output
- alpha / aux outputs are not guaranteed in the plain mip path
- therefore the recorded run should be understood as an RGB high-frequency bottleneck plus surface HF closure, not as an alpha/mass bottleneck
- this is why a later follow-up added `LAMBDA_NON_SURFACE_RGB_ENERGY`

#### Surface High-Frequency Closure

The full render is compared against GT in Laplacian space:

```text
L_surf_hf = mean(abs(Laplacian(I_full) - Laplacian(I_gt)))
```

This term is **not** added directly to the full optimizer loss. Instead, its gradients are routed only to the surface mask:

```text
accumulate_masked_gaussian_loss_grads(
    gaussians,
    L_surf_hf * lambda_surface_hf_closure,
    update_mask=surface_carrier,
    update_scale=SURFACE_HF_UPDATE_SCALE,
)
```

For the recorded run:

```text
lambda_surface_hf_closure = 0.02
SURFACE_HF_UPDATE_SCALE = 0.5
```

Interpretation:

- full image high-frequency error still closes the loop against GT
- but the extra high-frequency correction gradient is only assigned to `surface_carrier`
- this is why the surface layer can become visually meaningful on its own
- it avoids asking the non-surface layer to be the main high-frequency compensator

### Masked Gradient Mechanics

The implementation uses:

```text
accumulate_masked_gaussian_loss_grads(...)
```

This function computes gradients for Gaussian parameters and zeroes all rows outside the update mask before adding them to the existing optimizer gradients.

The masked parameters include row-aligned Gaussian tensors such as:

```text
_xyz
_features_dc
_features_rest
_opacity
_scaling
_rotation
```

When both prior masked losses and layer-frequency surface closure are active, the code retains the autograd graph for the first masked gradient call so the second masked gradient call can still backpropagate.

### Surface Normal Lock

The recorded run also used:

```text
SURFACE_NORMAL_LOCK=1
```

Implementation:

```text
SurfaceNormalLock
```

Mask selection:

- normally uses the prior-loss mask or the intersection with any external update mask
- in this run, the practical target is `surface_carrier`

Normal source:

- if the payload contains aligned `anchor_normal`, use it
- otherwise infer the normal from the Gaussian's thinnest scale axis

Anchor source:

- if the payload contains aligned `anchor_xyz`, use it
- otherwise use the initial Gaussian center at lock creation time

Two operations are applied during training:

```text
project_xyz_gradient_to_tangent(...)
project_xyz_to_locked_normal_coord(...)
```

Effect:

- before optimizer step, the selected centers' XYZ gradients are projected onto the tangent plane
- after optimizer step, selected centers are projected back to their initial normal coordinate
- tangential movement is allowed
- normal drift is removed

This is important for this run because it lets surface carriers absorb image correction while remaining attached to the surface neighborhood.

### Densification Constraint

The run used:

```text
DENSIFY_UNTIL_ITER=0
```

Reason:

- all layer masks are row-aligned to the checkpoint Gaussian count
- densify/prune changes the Gaussian count
- changing count would invalidate `surface_carrier`, `non_surface_active`, and related masks

## Diagnostic Renderer Implementation

The diagnostic layer images should be interpreted using the fixed renderer code, not the older layer debug path.

Relevant SOF code:

```text
SOF/gaussian_renderer/__init__.py
SOF/scripts/render_surface_state_class_groups_v0.py
```

Important fixes after the recorded run:

- `render_simple(...)` accepts `kernel_size`
- `render_simple(...)` accepts `vanilla_mip_mode`
- `render_surface_state_class_groups_v0.py` can call `render_simple(..., vanilla_mip_mode=True)`
- `non_surface_active` is a first-class composite group
- flat output can be written with `--flat_output_root`
- `*_premul.png` is no longer accidentally alpha-multiplied twice
- `*_composite_white.png` gives a white-background view of the isolated layer

Preferred diagnostic command pattern:

```bash
python -u scripts/render_surface_state_class_groups_v0.py \
  --scene_root "$SCENE" \
  --model_path "$MODEL" \
  --iteration 32000 \
  --surface_state_payload "$STATE" \
  --output_root "$OUT/_tree" \
  --flat_output_root "$OUT" \
  --images_subdir images_2 \
  --split test \
  --max_views 1 \
  --groups full,surface_carrier,surface_candidate,non_surface_active,no_mesh_neutral,near_surface_uncertain,off_surface_near_mesh,low_opacity_neutral \
  --skip_empty \
  --vanilla_mip_mode \
  --save_alpha \
  --save_premul \
  --save_composite
```

The diagnostic images mentioned by the user came from the pre-flat folder:

```text
/root/autodl-tmp/SOFSR/output/debug_qwen244_layerfreq_hfbottleneck_layers_fixed_00000/surface_carrier_raw_black.png
/root/autodl-tmp/SOFSR/output/debug_qwen244_layerfreq_hfbottleneck_layers_fixed_00000/near_surface_uncertain_raw_black.png
```

If regenerating diagnostics, prefer a flat folder and compare:

```text
test_00000_surface_carrier_raw_black.png
test_00000_surface_candidate_raw_black.png
test_00000_near_surface_uncertain_raw_black.png
test_00000_non_surface_active_raw_black.png
test_00000_full_raw_black.png
summary.json
```

## Main Server Paths

Baseline mip model:

```text
/root/autodl-tmp/kitchen/_hrgsrefiner_assets/kitchen_mip_vanilla_images8_v1/mip30k_rerun_v0
```

Recorded layer-frequency model:

```text
/root/autodl-tmp/SOFSR/output/mipsplatting_prior_repro/kitchen/qwen244_layerfreq_nonsurface_hfbottleneck_v0
```

Prepared Qwen/VOSR priors:

```text
/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/qwen_steps1_seed42_rcgm_aligned_images2_train244_v0
```

Surface-state payload:

```text
/root/autodl-tmp/SOFSR/output/gaussian_surface_sort/kitchen/mip30k_rerun_gs2mesh_surface_state_v0_relaxed_carrier_v1/gaussian_surface_state_v0.pt
```

Important diagnostic render folder observed by the user:

```text
/root/autodl-tmp/SOFSR/output/debug_qwen244_layerfreq_hfbottleneck_layers_fixed_00000
```

Key diagnostic images:

```text
/root/autodl-tmp/SOFSR/output/debug_qwen244_layerfreq_hfbottleneck_layers_fixed_00000/surface_carrier_raw_black.png
/root/autodl-tmp/SOFSR/output/debug_qwen244_layerfreq_hfbottleneck_layers_fixed_00000/near_surface_uncertain_raw_black.png
```

Result files under the recorded model:

```text
/root/autodl-tmp/SOFSR/output/mipsplatting_prior_repro/kitchen/qwen244_layerfreq_nonsurface_hfbottleneck_v0/chkpnt32000.pth
/root/autodl-tmp/SOFSR/output/mipsplatting_prior_repro/kitchen/qwen244_layerfreq_nonsurface_hfbottleneck_v0/point_cloud/iteration_32000/point_cloud.ply
/root/autodl-tmp/SOFSR/output/mipsplatting_prior_repro/kitchen/qwen244_layerfreq_nonsurface_hfbottleneck_v0/results_psnr_ssim.json
/root/autodl-tmp/SOFSR/output/mipsplatting_prior_repro/kitchen/baseline_compare.json
```

## Training Configuration

Representative command settings for the recorded run:

```bash
MIPSPLATTING_ROOT=/root/autodl-tmp/mip-splatting
BASELINE_MODEL_DIR=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/kitchen_mip_vanilla_images8_v1/mip30k_rerun_v0
RUN_ROOT=/root/autodl-tmp/SOFSR/output/mipsplatting_prior_repro/kitchen
EXPERIMENT_NAME=qwen244_layerfreq_nonsurface_hfbottleneck_v0
RUN_BASELINE_IF_MISSING=0
PREPARED_PRIOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/qwen_steps1_seed42_rcgm_aligned_images2_train244_v0
PRIOR_SUBDIR=fused_priors
PRIOR_MASK_SUBDIR=usable_masks
LR_REFERENCE_IMAGES_SUBDIR=images_8
TRAIN_IMAGES_SUBDIR=images_2
TARGET_IMAGES_SUBDIR=images_2
PRIOR_SUPERVISION_IMAGES_SUBDIR=images_2
PRIOR_LOSS_GAUSSIAN_MASK_PAYLOAD=/root/autodl-tmp/SOFSR/output/gaussian_surface_sort/kitchen/mip30k_rerun_gs2mesh_surface_state_v0_relaxed_carrier_v1/gaussian_surface_state_v0.pt
PRIOR_LOSS_GAUSSIAN_MASK_KEY=surface_carrier
SURFACE_NORMAL_LOCK=1
LAYER_FREQUENCY_MASK_PAYLOAD=/root/autodl-tmp/SOFSR/output/gaussian_surface_sort/kitchen/mip30k_rerun_gs2mesh_surface_state_v0_relaxed_carrier_v1/gaussian_surface_state_v0.pt
LAYER_FREQUENCY_NON_SURFACE_KEY=non_surface_active
LAYER_FREQUENCY_SURFACE_KEY=surface_carrier
LAMBDA_NON_SURFACE_HF=0.03
LAMBDA_SURFACE_HF_CLOSURE=0.02
SURFACE_HF_UPDATE_SCALE=0.5
LAYER_FREQUENCY_FROM_ITER=30000
SURFACE_FILTER_3D=0
DENSIFY_UNTIL_ITER=0
```

Important interpretation:

- The run did **not** use the earlier surface mip-filter idea as the main mechanism.
- The run used full-image photometric/prior closure plus an added layer-frequency regularizer.
- The non-surface term primarily suppressed Laplacian high-frequency in `non_surface_active`.
- The surface closure term routed full-image high-frequency error gradients back to `surface_carrier`.
- `SURFACE_NORMAL_LOCK=1` kept prior-guided surface carriers from drifting along the normal direction.
- Densification was disabled because the mask payload is aligned to a fixed Gaussian count.

## Observations

Positive result:

- `surface_carrier_raw_black.png` is visually close to the real/full render.
- This is a good sign: the surface carrier layer is no longer just a sparse/noisy shell.
- The separated surface layer appears to carry real scene appearance and has physical/explanatory value.

Remaining issue:

- `near_surface_uncertain_raw_black.png` still contains substantial information-carrying Gaussian content.
- This means the current split is not yet cleanly disentangled.
- The high-frequency bottleneck is not enough by itself, because `near_surface_uncertain` can still carry low-frequency and mid-frequency appearance.
- In other words, this run makes the surface layer meaningful, but still leaves a second near-surface compensating layer.

Working interpretation:

```text
full render ~= surface_carrier main appearance + near_surface_uncertain residual/compensation
```

This is better than earlier states where surface-only looked broken, but it is not yet the desired physical decomposition. The target is:

```text
surface_carrier: main physically meaningful surface appearance
near_surface_uncertain / non_surface_active: minimal residual, no strong independent texture/image content
```

## Why This Run Is Valuable

This version should be kept as a reference because it demonstrates that:

- surface carrier can become self-explanatory under normal lock plus surface HF closure
- full render quality can remain acceptable while inspecting meaningful layers
- the key remaining problem is not simply "surface layer bad", but "near-surface uncertain layer still has too much expressive capacity"

This narrows the next step: suppress the RGB contribution / expression capacity of `near_surface_uncertain`, rather than only suppressing high-frequency.

## Follow-Up Direction

The follow-up change added after this observation is:

```text
LAMBDA_NON_SURFACE_RGB_ENERGY
```

Recommended first follow-up target:

```text
LAYER_FREQUENCY_NON_SURFACE_KEY=near_surface_uncertain
LAMBDA_NON_SURFACE_RGB_ENERGY=0.02
LAMBDA_NON_SURFACE_HF=0.02
LAMBDA_SURFACE_HF_CLOSURE=0.03
```

Reason:

- `near_surface_uncertain` is the layer visibly carrying extra information.
- RGB energy bottleneck directly penalizes black-background raw contribution.
- It should reduce the layer's ability to explain the image through low/mid-frequency appearance.
- Full-image closure remains active, so necessary appearance should move back to `surface_carrier` rather than simply disappear.

If the near-surface layer remains too bright/informative, raise:

```text
LAMBDA_NON_SURFACE_RGB_ENERGY=0.05
```

while keeping the rest fixed for the next comparison.

## Diagnostic Rendering Notes

Use mip-compatible layer rendering for judging these layers. The older `render_simple` path could mislead because it was not fully aligned with mip-splatting's full render path.

Preferred diagnostic output should include:

- `surface_carrier`
- `surface_candidate`
- `near_surface_uncertain`
- `non_surface_active`
- `full`

For visual interpretation:

- `*_raw_black.png`: black-background contribution; useful for seeing actual layer contribution.
- `*_composite_white.png`: white-composited appearance; useful for judging how a single layer would look as an image.
- `*_alpha.png`: coverage / opacity support.

The best recorded qualitative signal so far is:

```text
surface_carrier_raw_black is close to the real render,
but near_surface_uncertain_raw_black still carries too much scene information.
```

## 2026-06-07 Baseline Correction And SR-Prior Value Check

The 2026-06-06 HF-difference analysis used a bad mip30k baseline checkpoint. Its `cfg_args` showed:

```text
images='images_8', resolution=4
```

In mip-splatting, `-r` is applied on top of the image directory selected by `-i`, so this effectively trained from `images_8` downsampled by another 4x, close to `images_32`. That explains the abnormally blurry render and low baseline score.

Correct rerun:

```text
/root/autodl-tmp/kitchen/_hrgsrefiner_assets/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0
```

Correct low-resolution training / high-resolution evaluation semantics:

```text
train:  -i images_8 -r 1
eval:   -i images_2 -r 1
```

Correct baseline metric:

```json
{
  "PSNR": 27.9777929,
  "SSIM": 0.8099241
}
```

Because the Gaussian count changed from the bad checkpoint, all Gaussian-indexed payloads also had to be regenerated. The new surface-state payload is:

```text
/root/autodl-tmp/SOFSR/output/gaussian_surface_sort/kitchen/mip30k_rerun_check_directsrc_r1_gs2mesh_surface_state_v0_relaxed_carrier_v1/gaussian_surface_state_v0.pt
```

Two clean 30000 -> 32000 runs were then made from the corrected baseline:

```text
no-SR cleanup:
/root/autodl-tmp/SOFSR/output/mipsplatting_prior_repro/kitchen/mip30k_to32000_surfacecleanup_nosrprior_from_r1baseline_v0

SR prior + same cleanup:
/root/autodl-tmp/SOFSR/output/mipsplatting_prior_repro/kitchen/fullprior_244_vosr_qwen_masked_nlock_surfacecleanup_from_r1baseline_v0
```

The corrected HF analysis output is:

```text
/root/autodl-tmp/SOFSR/output/hf_prior_injection_analysis_from_r1baseline_v0/hf_prior_injection_analysis_v0.json
```

Summary:

```json
{
  "psnr_mip30k_mean": 27.977792724177522,
  "psnr_nosr_cleanup_mean": 28.572666229446238,
  "psnr_with_sr_prior_mean": 28.56375395544886,
  "cleanup_hf_l2_mean": 0.00509398862985628,
  "prior_extra_hf_l2_mean": 0.002789455060181873,
  "prior_extra_vs_cleanup_ratio_mean": 0.5490724033993246,
  "cleanup_align_gt_residual_mean": 0.17682898598057883,
  "prior_extra_align_gt_residual_mean": 0.06824160175664085
}
```

Interpretation:

- The corrected no-SR cleanup improves the baseline by about `+0.595 dB`.
- Adding the current masked Qwen/VOSR SR prior on top of the same cleanup gives `-0.009 dB` relative to no-SR cleanup, effectively no benefit.
- SR prior still injects substantial high-frequency change: about `55%` of the cleanup HF magnitude.
- However, the prior-added HF aligns weakly with GT residual direction (`0.068`) compared with the cleanup HF (`0.177`).

Current conclusion:

```text
The SR prior is not useless, but direct strong RGB/HF supervision is not reliable in this corrected controlled setting.
It should be treated as a gated/weak high-frequency suggestion or early bootstrap signal, not as a late strong target.
The layer-frequency cleanup itself is the more reliable improvement in this test.
```

The old 2026-06-06 bad-baseline analysis and any models derived from the `images_8 -r4` checkpoint should be deleted or marked invalid.

# Surface Route SR Injection Handoff v0

Date: 2026-06-02

This document summarizes the current state of the surface-route SR injection experiments so another AI/agent can continue without rediscovering the context.

## 1. Current Goal

We are trying to inject more SR-consistent high-frequency detail into a processed surface-bearing Gaussian field.

The current working loss is:

```text
L_route = sum_p w(p) * | HP(I_surface_route_branch(p)) - T_sr_consensus(p) |
```

where:

- `I_surface_route_branch` is a render of a masked Gaussian subset, usually surface or surface+proxy depending on `SURFACE_MASK_KEY`.
- `HP(.)` is a Laplacian high-pass.
- `T_sr_consensus` is the direct multi-view-consistent SR high-frequency target produced by the builder.
- `w(p)` is a consensus/route weight from prior mask, low-frequency gate, route quality, and multi-view consistency.

The full training loss still includes the original full-field RGB training:

```text
L_total = L_full_L1_SSIM(images_8) + lambda_route * L_route
```

The important conceptual point is that the route loss is no longer fitting an `SR-anchor residual`. It is fitting direct SR high-frequency consensus on selected sparse pixels.

## 2. Repository And Runtime Layout

Local development paths:

- SOF repo: `/Users/ltl/Desktop/VGGTSR/SOF`
- mip-splatting repo: `/Users/ltl/Desktop/VGGTSR/mip-splatting`
- builder: `/Users/ltl/Desktop/VGGTSR/SOF/scripts/build_surface_route_consensus_v0.py`
- builder wrapper: `/Users/ltl/Desktop/VGGTSR/SOF/scripts/run_build_surface_route_consensus_v0_kitchen.sh`
- route training wrapper: `/Users/ltl/Desktop/VGGTSR/SOF/scripts/run_mipsplatting_surface_route_consensus_v0_kitchen.sh`
- underlying hybrid train: `/Users/ltl/Desktop/VGGTSR/mip-splatting/hybrid_sdfgs/train.py`

Server runtime paths:

- server workspace: `/root/autodl-tmp/SOFSR`
- scene root: `/root/autodl-tmp/kitchen`
- SR prior root:
  `/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/qwen_steps1_seed42_rcgm_aligned_order_v0`
- processed augmented baseline model:
  `/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model`
- current built consensus root:
  `/root/autodl-tmp/SOFSR/output/surface_route_consensus_v0/kitchen/mip30k_rerun_v0_surface_candidate_route_consensus_v0_local3_p24000_directsr_augproxy4k`
- consensus per-view payloads:
  `/root/autodl-tmp/SOFSR/output/surface_route_consensus_v0/kitchen/mip30k_rerun_v0_surface_candidate_route_consensus_v0_local3_p24000_directsr_augproxy4k/per_view`
- current route-trained output:
  `/root/autodl-tmp/SOFSR/output/kitchen_mipsplatting_surface_route_consensus_v0/mip30k_rerun_v0_surface_route_consensus_v0_surfaceplusproxy_local3_p24000_directsr_augproxy4k_continue4k_r1`
- diagnostic original-only latest render:
  `/root/autodl-tmp/SOFSR/output/surface_only_debug/kitchen/continue4k_r1_original_only/test/ours_8000/renders/00000.png`
- older augmented full render used for visual comparison:
  `/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model/test/ours_4000/renders/00000.png`

## 3. Current Assets

Augmented model:

- model path:
  `/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model`
- iteration: `4000`
- Gaussian count at iteration 4000: `1029678`
- observed route builder surface count: `599363/1029678`
- contains `point_cloud/iteration_4000/point_cloud.ply`
- contains tracking metadata: `point_cloud/iteration_4000/gaussian_tags.pt`
- contains original SOF checkpoint: `chkpnt4000.pth`

Checkpoint compatibility:

- `chkpnt4000.pth` is a SOF-format checkpoint with 14 model items, including `filter_3D`.
- hybrid mip-splatting `GaussianModel.restore()` expects 12 or 13 items.
- Directly using the SOF checkpoint caused:

```text
ValueError: too many values to unpack (expected 12)
```

- A hybrid-compatible checkpoint was produced by dropping `filter_3D` while preserving tracking state:
  `/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model/chkpnt4000_hybrid_compat.pth`

Mask payloads:

- Do not use a 423895-length mask with the augmented 1029678-Gaussian model.
- That mismatch caused:

```text
ValueError: Mask length mismatch for key 'surface_candidate': expected 1029678, got 423895
```

- The current mask payload must be aligned to `augmented_model@4000`, using `gaussian_tags.pt`.
- The cleaner recommended aligned payload path is:
  `/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model_train_mask_iter4000_cleaner_v0.pt`
- The earlier broader payload path is:
  `/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model_train_mask_iter4000_v0.pt`

Expected keys in the aligned mask payload:

- `surface_candidate`: original/source surface-side Gaussians, derived from `source_tag == 0`
- `proxy_candidate`: proxy/prior-injected Gaussians, derived from `source_tag == 1`
- `proxy_gen1`: proxy Gaussians with `generation <= 1`
- `surface_plus_proxy`: union of `surface_candidate | proxy_candidate`
- `surface_plus_proxy_gen1`: union of `surface_candidate | proxy_gen1`
- `source_tag`
- `generation`

## 4. Builder Code Framework

Main builder:

```text
/Users/ltl/Desktop/VGGTSR/SOF/scripts/build_surface_route_consensus_v0.py
```

Relevant implementation:

- CLI args are defined around lines 597-629.
- It loads the model, the surface mask, and train cameras around lines 644-658.
- View selection currently uses `_select_uniform(train_cameras, max_views)` around line 658.
- First pass over selected views starts around line 682.
- For each view, it renders both full field and masked surface subset around lines 688-702.
- It selects SR high-frequency candidate pixels around lines 714-730.
- It projects surface subset Gaussians and builds route bins around lines 732-743.
- It queries per-pixel route keys around lines 744-758.
- It aggregates route observations around lines 759-766.
- Local route consensus/write-back starts around lines 829-870.
- Sparse payload writing starts around line 872.
- Sparse payload uses `signal_mode = "direct_sr_highfreq"` around line 892.

Current consensus construction logic:

```text
SR prior image -> Laplacian highfreq -> candidate pixels
masked surface subset render -> route key per candidate pixel
local window route observations -> multi-view route consensus
per-view sparse payload -> target_highfreq_samples + target_weight_samples
```

Current builder run of interest:

```bash
cd /root/autodl-tmp/SOFSR

BASELINE_MODEL_DIR=/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model \
BASELINE_ITERATION=4000 \
PREPARED_SR_PRIOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/qwen_steps1_seed42_rcgm_aligned_order_v0 \
SURFACE_STATE_PAYLOAD=/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model_train_mask_iter4000_v0.pt \
SURFACE_MASK_KEY=surface_candidate \
CONSENSUS_TAG=mip30k_rerun_v0_surface_candidate_route_consensus_v0_local3_p24000_directsr_augproxy4k \
MAX_VIEWS=32 \
MAX_CANDIDATE_PIXELS=24000 \
LOCAL_CONSENSUS_VIEWS=3 \
ANCHOR_MODE=full \
LOWFREQ_GATE_KERNEL=9 \
LOWFREQ_GATE_TAU=0.08 \
SPARSE_PAYLOAD=1 \
SAVE_DEBUG_PNG=0 \
bash scripts/run_build_surface_route_consensus_v0_kitchen.sh
```

Observed builder progress:

```text
[surface-route-consensus-v0] scene=/root/autodl-tmp/kitchen model=.../augmented_model views=32 surface=599363/1029678 lowfreq_anchor=full
[surface-route-consensus-v0] first-pass view=DSCF0657 candidate=24000 valid=23997 routes=3194
[surface-route-consensus-v0] first-pass view=DSCF0665 candidate=24000 valid=23993 routes=3395
[surface-route-consensus-v0] first-pass view=DSCF0674 candidate=24000 valid=23999 routes=3704
[surface-route-consensus-v0] first-pass view=DSCF0683 candidate=24000 valid=24000 routes=6515
```

Builder cost assessment:

- Heavy part: first pass, because every selected view does full render, surface-subset render, SR prior load, candidate selection, projection/binning, and route query.
- Cheaper part: second pass local consensus and sparse write-back.
- Increasing `MAX_CANDIDATE_PIXELS` or `MAX_VIEWS` increases build cost.
- Increasing training lambda, update mask coverage, iteration count, or surface+proxy train branch can increase injection amount without rebuilding consensus.

Known builder limitations:

- `MAX_VIEWS=2` alone is unsafe in the current script because `_select_uniform` with 2 views tends to choose sequence endpoints, not adjacent views.
- Current script has no explicit per-view top-N output cap after consensus; it writes all `target_weight > 0` sparse pixels.
- Default `top_k=2` and `cell_grid=4` can be too fine for 2-view pair mode.
- `--prior_delta_clip` currently exists but effectively only enters manifest/state; it does not materially affect route signal construction.
- User currently requested reuse, not new pair-mode code.

## 5. Training Code Framework

Main training wrapper:

```text
/Users/ltl/Desktop/VGGTSR/SOF/scripts/run_mipsplatting_surface_route_consensus_v0_kitchen.sh
```

Relevant wrapper behavior:

- It optionally builds consensus if missing.
- It passes route loss settings into `run_mipsplatting_stablesr_prior_scene.sh`.
- It passes `PRIOR_LOSS_GAUSSIAN_MASK_PAYLOAD="${SURFACE_STATE_PAYLOAD}"`.
- It passes `PRIOR_LOSS_GAUSSIAN_MASK_KEY="${SURFACE_MASK_KEY}"`.
- It passes `SURFACE_ROUTE_SURFACE_ONLY`.

Underlying train:

```text
/Users/ltl/Desktop/VGGTSR/mip-splatting/hybrid_sdfgs/train.py
```

Relevant implementation:

- Sparse route payload loading handles `sample_y`, `sample_x`, `target_highfreq_samples`, `target_weight_samples`, and `signal_mode` around lines 1260-1287.
- The prior-loss/update mask is loaded around lines 2403-2412.
- The surface route render branch is built with `combine_gaussian_update_masks(prior_loss_update_mask, external_update_mask)` around lines 2434-2458.
- Route payload is fetched during training around lines 3420-3430.
- Direct SR high-frequency mode uses `pred_highfreq` vs `route_target` around lines 3456-3459.
- The weighted sparse route loss is computed around lines 3470-3472.
- If a prior-loss mask exists, this route loss is added to `masked_prior_loss` around lines 3473-3478.

Important training interpretation:

- If `SURFACE_ROUTE_SURFACE_ONLY=1` and `SURFACE_MASK_KEY=surface_plus_proxy`, then the route branch renders `surface_plus_proxy`, not just original surface.
- A later debug image named `original_only` is not equivalent to the actual route training branch if training used `surface_plus_proxy`.
- Full RGB L1/SSIM is still computed on the full Gaussian field.

Current route training command:

```bash
cd /root/autodl-tmp/SOFSR

PREPARED_SR_PRIOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/qwen_steps1_seed42_rcgm_aligned_order_v0 \
BASELINE_MODEL_DIR=/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model \
BASELINE_ITERATION=4000 \
START_CHECKPOINT=/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model/chkpnt4000_hybrid_compat.pth \
SURFACE_STATE_PAYLOAD=/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model_train_mask_iter4000_v0.pt \
SURFACE_MASK_KEY=surface_plus_proxy \
CONSENSUS_TAG=mip30k_rerun_v0_surface_candidate_route_consensus_v0_local3_p24000_directsr_augproxy4k \
SURFACE_ROUTE_CONSENSUS_ROOT=/root/autodl-tmp/SOFSR/output/surface_route_consensus_v0/kitchen/mip30k_rerun_v0_surface_candidate_route_consensus_v0_local3_p24000_directsr_augproxy4k \
FORCE_BUILD_CONSENSUS=0 \
ROUTE_SPARSE_PAYLOAD=1 \
ROUTE_ANCHOR_MODE=full \
ROUTE_LOWFREQ_GATE_KERNEL=9 \
ROUTE_LOWFREQ_GATE_TAU=0.08 \
SURFACE_ROUTE_SURFACE_ONLY=1 \
LAMBDA_SURFACE_ROUTE_CONSENSUS=0.12 \
SURFACE_ROUTE_CONSENSUS_MIN_PIXELS=32 \
PRIOR_EDGE_UPDATE_SCALE=0.50 \
EXPERIMENT_NAME=mip30k_rerun_v0_surface_route_consensus_v0_surfaceplusproxy_local3_p24000_directsr_augproxy4k_continue4k_r1 \
ITERATIONS=8000 \
bash scripts/run_mipsplatting_surface_route_consensus_v0_kitchen.sh
```

## 6. Diagnostic / Render Code

Useful existing scripts:

- Render full model without GT copy:
  `/Users/ltl/Desktop/VGGTSR/SOF/scripts/render_model_no_gt.py`
- Render/export a selected Gaussian subset from a mask payload:
  `/Users/ltl/Desktop/VGGTSR/SOF/scripts/export_gaussian_mask_subset_v0.py`
- Render full-model variants from tracking tags or payload masks:
  `/Users/ltl/Desktop/VGGTSR/SOF/scripts/export_gaussian_group_variant_v0.py`
- Render surface-state class groups:
  `/Users/ltl/Desktop/VGGTSR/SOF/scripts/render_surface_state_class_groups_v0.py`

Surface proxy migration update:

- `build_surface_proxy_augmented_field_v0.py` now keeps the old nearest-anchor logic as `--proxy_anchor_mode anchor`.
- The new default is `--proxy_anchor_mode ray_surface`.
- The old output composition can be recovered with `--proxy_output_mode surface_plus_proxy`.
- The new default output composition is `--proxy_output_mode replace_cloned_donors`, meaning:

```text
base model - donor gaussians that received surface clones + ray-surface proxy clones
```

- Donors without a ray-surface hit are kept in the base model by default.
- `--proxy_ray_fallback_to_anchor` can force unmatched donors to use the old payload anchor, but that is not the default.
- The wrapper `run_mipsplatting_surface_proxy_augmented_supervision_v0_kitchen.sh` now defaults to:

```text
PROXY_ANCHOR_MODE=ray_surface
PROXY_OUTPUT_MODE=replace_cloned_donors
```

- To run the cached old behavior:

```bash
PROXY_ANCHOR_MODE=anchor \
PROXY_OUTPUT_MODE=surface_plus_proxy \
bash scripts/run_mipsplatting_surface_proxy_augmented_supervision_v0_kitchen.sh
```

Most important diagnostic now:

The user compared:

- old full augmented render:
  `/root/autodl-tmp/SOFSR/output/mip_surface_proxy_augmented_supervision_v0/kitchen/mip30k_rerun_v0_surface_candidate_plus_surface_complement_active_surface_proxy_augmented_supervision_v0_images2_4k/export/augmented_model/test/ours_4000/renders/00000.png`
- latest `original_only` debug render:
  `/root/autodl-tmp/SOFSR/output/surface_only_debug/kitchen/continue4k_r1_original_only/test/ours_8000/renders/00000.png`

These are not the same render口径. The old image is full augmented render, while the latest path explicitly says `original_only`. Therefore the current first hypothesis is:

```text
coverage不足 is more likely because proxy/carrier Gaussians were not rendered in the diagnostic image,
not because the full model was necessarily pulled/broken.
```

Run this decomposition before micro-tuning:

```bash
cd /root/autodl-tmp/SOFSR

SCENE=/root/autodl-tmp/kitchen
LATEST=/root/autodl-tmp/SOFSR/output/kitchen_mipsplatting_surface_route_consensus_v0/mip30k_rerun_v0_surface_route_consensus_v0_surfaceplusproxy_local3_p24000_directsr_augproxy4k_continue4k_r1
OUT=/root/autodl-tmp/SOFSR/output/surface_only_debug/kitchen/coverage_ab_continue4k_r1

python scripts/export_gaussian_group_variant_v0.py \
  --scene_root "$SCENE" \
  --model_path "$LATEST" \
  --output_root "$OUT/latest_full" \
  --images_subdir images_2 \
  --iteration 8000 \
  --split test \
  --max_views 1 \
  --selection_source tracking \
  --selection_key full \
  --selection_mode full \
  --save_alpha

python scripts/export_gaussian_group_variant_v0.py \
  --scene_root "$SCENE" \
  --model_path "$LATEST" \
  --output_root "$OUT/latest_original_only" \
  --images_subdir images_2 \
  --iteration 8000 \
  --split test \
  --max_views 1 \
  --selection_source tracking \
  --selection_key original \
  --selection_mode selected_only \
  --save_alpha

python scripts/export_gaussian_group_variant_v0.py \
  --scene_root "$SCENE" \
  --model_path "$LATEST" \
  --output_root "$OUT/latest_proxy_only" \
  --images_subdir images_2 \
  --iteration 8000 \
  --split test \
  --max_views 1 \
  --selection_source tracking \
  --selection_key proxy \
  --selection_mode selected_only \
  --save_alpha

find "$OUT" -path "*/test/ours_8000/renders/00000.png" -print
```

Observed result from this decomposition:

```text
latest_full:
  selected_gaussians = 1233486 / 1233486

latest_original_only:
  selected_gaussians = 638999 / 1233486
  selected_ratio = 0.5180431719533095

latest_proxy_only:
  selected_gaussians = 594487 / 1233486
  selected_ratio = 0.48195682804669043
```

These counts sum exactly:

```text
638999 + 594487 = 1233486
```

So the latest model is effectively split into about 52% original and 48% proxy/prior-injected Gaussians under the tracking tags. This strongly supports the idea that an `original_only` render is missing a large carrier component.

The diagnostic script also printed:

```text
[export-gaussian-variant] warning: tracking source_tag is longer than the loaded point cloud (1237616 vs 1233486); truncating to the current Gaussian count for visualization.
```

This warning should be remembered. The visualization is still useful for coarse original/proxy decomposition, but exact per-Gaussian tracking should be treated cautiously until `gaussian_tags.pt` is regenerated or verified to be aligned after prune/densify.

Interpretation:

- If `latest_full` has good coverage and `latest_original_only` has holes, the model is probably not globally pulled; the diagnostic simply omitted proxy.
- If `latest_proxy_only` fills the missing regions, the coverage is proxy/carrier-supported.
- If `latest_full` also has coverage holes, then training likely damaged opacity/scale/position or removed support.
- If only original-only got worse, explanation weight may have shifted from original/surface to proxy, which is different from full-field failure.

## 7. Quality Evaluation So Far

Recorded conclusions:

- `surface_candidate` standalone training and directly extracted 30k subset looked almost the same.
- This suggests the existing贴表层 alone does not have enough carrier capacity; continued training alone is unlikely to grow all missing detail.
- Medium-strength `surface + proxy` augmentation was effective.
- Recorded improvement was approximately:

```text
PSNR 21.52 / SSIM 0.712
to
PSNR 23.45 / SSIM 0.741
```

- This supports the idea that supplementing the surface with selected proxy/carrier Gaussians can move information into a surface-compatible carrier system.
- But full proxy expansion was harmful, both with high opacity and soft opacity.
- Therefore more proxy is not automatically better.
- Only some donors are suitable for surface proxy conversion.
- Future proxy use needs stricter donor gating, not all `surface_complement_active`.

Current latest qualitative result:

- The `continue4k_r1` route injection appears to have stronger SR detail injection.
- It also shows more clutter/noise.
- Some areas appear to have insufficient Gaussian coverage in the `original_only` diagnostic render.
- Coverage should be diagnosed before noise, because the current comparison likely mixes full-render and subset-render口径.

Current working interpretation:

```text
Effective injection requires both:
1. reliable multi-view SR high-frequency targets,
2. enough continuous carrier capacity in the rendered/updateable Gaussian subset.
```

The important failure mode is not just wrong supervision. A broken or too sparse carrier layer cannot absorb the signal cleanly.

## 8. Current Open Questions

Coverage:

- Is the apparent latest coverage loss present in the full model render, or only in `original_only`?
- Does `proxy_only` explain the missing regions?
- Did route training shift explanation from original surface to proxy, or did it truly damage the full field?

Noise/clutter:

- Not analyzed yet by request.
- Likely needs separate analysis after coverage decomposition.
- Candidate causes include route target noise, too-high route lambda, too-wide update mask, too permissive proxy inclusion, or high-frequency objective fighting full RGB/SSIM.

Injection amount without extra builder compute:

- Reuse the existing completed consensus payloads.
- Increase train-side consumption rather than rebuild-side candidate search.
- Candidate train-side knobs include `LAMBDA_SURFACE_ROUTE_CONSENSUS`, update mask choice (`surface_plus_proxy` vs cleaner `surface_plus_proxy_gen1`), route view sampling probability, training iterations, and `PRIOR_EDGE_UPDATE_SCALE`.
- Avoid raising `MAX_VIEWS` or `MAX_CANDIDATE_PIXELS` unless willing to pay build cost again.

## 9. Important Cautions For Next Agent

- Do not compare a full render against an `original_only` diagnostic render as if they were equivalent.
- Do not use the old 423895-length surface-state payload with the 1029678-Gaussian augmented model.
- Do not load the original SOF `chkpnt4000.pth` directly into hybrid mip-splatting training; use `chkpnt4000_hybrid_compat.pth`.
- Do not assume `MAX_VIEWS=2` gives adjacent views in the current builder. It uses uniform selection.
- Do not rebuild consensus unless necessary; the current `per_view` folder is complete.
- Do not micro-tune noise before checking whether coverage loss is real full-field loss or just subset omission.

## 10. Suggested Immediate Next Step

First run the `latest_full / latest_original_only / latest_proxy_only` decomposition in Section 6.

Then decide:

- If full render is fine: proceed to noise/clutter analysis, probably by separating route target quality, proxy contribution, and update mask strength.
- If full render is broken: inspect opacity/scale/position distributions before and after route training, then decide whether to reduce route strength or constrain update targets.
- If proxy fills missing regions: treat proxy as necessary carrier, but gate it more carefully for future experiments.

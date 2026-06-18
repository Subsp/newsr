# N-PSE Edge/Trust Cache v0

This is the first offline step for false-continuity-aware neighborhood surface expansion.
It does not train GS directly. It builds frame-aligned targets that let us inspect whether
sharp edges and smooth regions are reasonable before adding losses.

## Inputs

- `anchor_dir`: low-frequency direct x1 LR-GS render.
- `sr_dir`: enhancement SR prior, typically prepared `fused_priors`.
- `depth_dir`: mesh-aligned external depth-prior frames. Depth jump is the main geometric edge cue.
- `reference_dir`: optional frame-name reference, usually `images_2`, used for LLFF train-subset alignment.

Depth priors should be aligned to the gs2mesh/COLMAP camera depth domain before N-PSE.
The mesh is only the scale/camera reference here; it is not treated as the edge oracle.
Use `scripts/run_build_mesh_aligned_depth_prior_cache_v0_kitchen.sh` to create:

```text
.../_hrgsrefiner_assets/depth_prior_aligned_gs2mesh/<name>/aligned_depth
```

## Outputs

The cache writes:

- `edge_depth`: raw sharp cue from depth-prior jumps.
- `edge_depth_confirmed`: geometry edge strength after SR-structure confirmation.
- `edge_sr`: structural edge cue from the SR prior.
- `edge_fused`: fused edge probability used for visualization.
- `edge_position`: the actual edge-position seed used to build the edge band.
- `edge_band`: dilated edge-position band used by edge targets and continuous masks.
- `edge_type`: black continuous, red SR-confirmed geometry edge, yellow appearance edge, blue uncertain edge.
- `depth_only_uncertain`: depth-only jumps that were not confirmed by SR structure.
- `barrier`: propagation barrier. Continuous residual diffusion should not cross strong barriers.
- `trust_sr`: SR-prior reliability from low-frequency agreement and local residual consistency.
- `trust_edge_raw`: edge-target weight before optional direction gating.
- `edge_direction_gate`: optional non-oracle gate from anchor/SR/residual gradient-direction agreement.
- `trust_edge`: conservative edge-target injection weight.
- `continuous_mask`: smooth-region mask where neighborhood residual diffusion is allowed.
- `residual_raw`: signed visualization of `HP(SR) - HP(anchor)`.
- `residual_npse`: edge-stopped neighborhood propagated residual.
- `edge_target`: narrow-band edge enhancement target.
- `debug_overlay`: SR prior with edge/trust overlay.
- `npz`: float arrays for later training integration.

## Kitchen Smoke

First align the raw external depth prior to gs2mesh:

```bash
cd /root/autodl-tmp/newsr/SOF
MESH_PATH=/path/to/gs2mesh_mesh.ply \
DEPTH_PRIOR_DIR=/root/autodl-tmp/kitchen/raw_depth_prior_x1 \
LIMIT=8 \
OVERWRITE=1 \
bash scripts/run_build_mesh_aligned_depth_prior_cache_v0_kitchen.sh
```

Then set `DEPTH_PRIOR_DIR` to the produced `aligned_depth` directory.

```bash
cd /root/autodl-tmp/newsr/SOF
DEPTH_PRIOR_DIR=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/depth_prior_aligned_gs2mesh/render_x1_depthprior_images_2_train_gs2mesh_aligned_v0/aligned_depth \
LIMIT=8 \
OVERWRITE=1 \
bash scripts/run_build_npse_edge_trust_cache_v0_kitchen.sh
```

Expected output root:

```text
/root/autodl-tmp/kitchen/_hrgsrefiner_assets/npse_cache/render_x1_restormer_depthprior_npse_v0
```

Inspect `debug_overlay`, `edge_type`, `continuous_mask`, `residual_raw`, and
`residual_npse` before wiring the cache into training.

In the default `GEOMETRY_CONFIRM_MODE=sr_confirmed`, a depth jump alone is not enough
to become a red geometry edge. Depth-only jumps are marked blue and receive only a weak
barrier, because gs2mesh-aligned depth can inherit LR-GS clutter or mesh holes.

By default the kitchen wrapper uses `EDGE_POSITION_MODE=appearance`, so yellow
SR-structure edges provide the edge locations. Red/blue remain diagnostics and
barrier hints rather than the primary source of edge position.

For fidelity-first edge targets, the wrapper defaults to `EDGE_TARGET_MODE=fidelity`
and `EDGE_RESIDUAL_CLIP=0.08`. This keeps the anchor low-frequency image unchanged
and injects only trust-gated high-frequency residuals inside the yellow-derived
edge band.

The direction-gated probe is enabled explicitly, so the original yellow-fidelity
cache remains reproducible:

```bash
DEPTH_PRIOR_DIR=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/depth_prior_aligned_gs2mesh/render_x1_depthprior_images_2_train_gs2mesh_aligned_v0/aligned_depth \
OUTPUT_NAME=render_x1_restormer_depthprior_npse_yellow_fidelity_dirgate_v0 \
EDGE_TARGET_DIRECTION_GATE=anchor_sr_residual_gradient \
OVERWRITE=1 \
bash scripts/run_build_npse_edge_trust_cache_v0_kitchen.sh
```

This does not use GT. GT high-frequency alignment is only an offline diagnostic
for deciding whether the gate is worth training with.

## NoSR Anchor Probe

To test whether NoSR's cleaned GS field gives a better low-frequency anchor, run:

```bash
cd /root/autodl-tmp/newsr/SOF
DEPTH_PRIOR_DIR=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/depth_prior_aligned_gs2mesh/render_x1_depthprior_images_2_train_gs2mesh_aligned_v0/aligned_depth \
LIMIT=8 \
OVERWRITE=1 \
bash scripts/run_build_npse_edge_trust_cache_v0_kitchen_nosr_anchor.sh
```

This changes only the anchor render. If red/blue still track clutter, the next
probe should regenerate `DEPTH_PRIOR_DIR` using a NoSR-derived mesh rather than
the vanilla gs2mesh mesh.

## EdgeCarrier / Continuous Diffusion v1

N-PSE v1 keeps the direct edge target as the conservative image signal, but adds
two structure-aware training hooks:

- Edge contrast: within `trust_edge`, match the target's directional luminance
  contrast instead of only fitting high-frequency pixels.
- Edge carrier shape: gaussians touched by `trust_edge` can be pushed toward a
  line-like carrier shape, using small/mid and mid/large scale-ratio penalties.

Both are off by default. Enable them through the existing NoSR cleanup runner:

```bash
PRIOR_EDGE_CONTRAST_WEIGHT=0.25 \
LAMBDA_PRIOR_EDGE_SHAPE=0.002 \
PRIOR_EDGE_SHAPE_THIN_RATIO=0.35 \
PRIOR_EDGE_SHAPE_LINE_RATIO=0.60 \
bash scripts/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh
```

For continuous-surface diffusion, materialize trainable targets from an existing
cache first:

```bash
python scripts/materialize_npse_continuous_targets_v0.py \
  --npse_cache_root /root/autodl-tmp/kitchen/_hrgsrefiner_assets/npse_cache/render_x1_restormer_depthprior_npse_yellow_fidelity_nogate_full_v0
```

This creates:

- `continuous_target`: `anchor + residual_npse * continuous_mask`
- `trust_continuous`: `trust_sr * continuous_mask`

Use them through the existing `prior_local` branch with a small weight, e.g.
`LAMBDA_PRIOR_LOCAL=0.005`, so surface interiors receive propagated residual
supervision without crossing the edge/barrier band.

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
- `edge_type`: black continuous, red SR-confirmed geometry edge, yellow appearance edge, blue uncertain edge.
- `depth_only_uncertain`: depth-only jumps that were not confirmed by SR structure.
- `barrier`: propagation barrier. Continuous residual diffusion should not cross strong barriers.
- `trust_sr`: SR-prior reliability from low-frequency agreement and local residual consistency.
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

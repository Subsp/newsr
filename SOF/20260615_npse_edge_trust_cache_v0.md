# N-PSE Edge/Trust Cache v0

This is the first offline step for false-continuity-aware neighborhood surface expansion.
It does not train GS directly. It builds frame-aligned targets that let us inspect whether
sharp edges and smooth regions are reasonable before adding losses.

## Inputs

- `anchor_dir`: low-frequency direct x1 LR-GS render.
- `sr_dir`: enhancement SR prior, typically prepared `fused_priors`.
- `depth_dir`: external depth-prior frames. Depth jump is the main geometric edge cue.
- `reference_dir`: optional frame-name reference, usually `images_2`, used for LLFF train-subset alignment.

## Outputs

The cache writes:

- `edge_depth`: sharp geometry cue from depth-prior jumps.
- `edge_sr`: structural edge cue from the SR prior.
- `edge_fused`: fused edge probability used for visualization.
- `edge_type`: black continuous, red geometry edge, yellow appearance edge, blue uncertain SR edge.
- `barrier`: propagation barrier. Continuous residual diffusion should not cross strong barriers.
- `trust_sr`: SR-prior reliability from low-frequency agreement and local residual consistency.
- `continuous_mask`: smooth-region mask where neighborhood residual diffusion is allowed.
- `residual_raw`: signed visualization of `HP(SR) - HP(anchor)`.
- `residual_npse`: edge-stopped neighborhood propagated residual.
- `edge_target`: narrow-band edge enhancement target.
- `debug_overlay`: SR prior with edge/trust overlay.
- `npz`: float arrays for later training integration.

## Kitchen Smoke

Set `DEPTH_PRIOR_DIR` to a frame-aligned depth-prior directory first.

```bash
cd /root/autodl-tmp/newsr/SOF
DEPTH_PRIOR_DIR=/root/autodl-tmp/kitchen/depth_prior_x1 \
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

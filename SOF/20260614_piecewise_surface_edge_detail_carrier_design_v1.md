# Piecewise-Smooth Surface-Edge Detail Carrier Prior v1

Working name: `PSE-DC`, short for `Piecewise-Smooth Surface-Edge Detail Carrier Prior`.

## 1. Core Claim

The central prior is:

> A real 3D scene is better described as piecewise-smooth surfaces separated by
> sharp geometric or visibility edges than as an unconstrained volumetric cloud.

For 3DGS super-resolution, this means high-frequency SR detail should not be
absorbed uniformly by all Gaussians. It should be routed into thin, stable
surface carriers, transported along local tangent planes, and stopped at
geometric or occlusion edges.

This differs from dense depth supervision. Depth maps tell the model where the
rendered geometry should be. PSE-DC turns depth/mesh structure into a
carrier-level optimization rule:

- smooth surface regions should become continuous, thin, tangent-moving detail
  carriers;
- sharp edges should preserve discontinuities and receive extra carrier
  capacity;
- off-surface or unstable Gaussians should not absorb SR hallucinations.

## 2. Positioning Against Prior Work

Existing 3DGS SR methods use geometry in useful but different ways:

- `GaussianSR` uses 2D diffusion SDS and controls random generative disturbance
  through timestep annealing and redundant-Gaussian discard.
- `SRGS` fits the broader pattern of 2D prior injection plus cross-view
  regularization.
- `IE-SRGS` uses 2DSR and monocular depth estimation as external guidance, and
  fuses them with internal multi-scale 3DGS image/depth guidance.
- `SuperGS` backprojects HR depth/error cues for multi-view consistent
  densification.
- `SplatSuRe` uses scene geometry to decide where SR supervision is needed,
  avoiding uniform SR injection.

PSE-DC is complementary to these: it does not merely add a depth loss or a
selection mask. It defines how 3D Gaussians should carry SR detail under a
piecewise-smooth surface and sharp-edge prior.

## 3. Inputs

The method can start from assets we already have:

- LR scene with calibrated cameras.
- Enhancement-SR or diffusion-SR prior images.
- A mesh or surface proxy from the existing NoSR pipeline.
- NoSR-style per-Gaussian surface-state payload:
  `surface_carrier`, `near_surface_uncertain`, `off_surface_near_mesh`,
  `anchor_xyz`, `anchor_normal`, `attach_conf`, `mesh_coverage_weight`,
  `signed_normal_offset`, and `tangent_offset`.

For the first implementation, the mesh can be the same mesh used by NoSR.
Later, monocular depth maps can be fused into the same payload when the mesh is
weak or incomplete.

## 4. Representation

For each Gaussian `g_i`, build or update a carrier payload:

```text
surface_conf_i      : confidence that g_i belongs to a stable surface
edge_conf_i         : confidence that g_i lies near a sharp surface/visibility edge
continuity_conf_i   : confidence that g_i has coherent surface neighbors
anchor_xyz_i        : closest mesh/depth surface point
anchor_normal_i     : local surface normal
tangent_u_i,v_i     : local tangent basis
normal_offset_i     : signed distance from anchor along normal
tangent_offset_i    : distance from anchor inside tangent plane
```

The basic detail uptake gate is:

```text
gate_i = surface_conf_i * continuity_conf_i * uptake_policy(edge_conf_i)
```

A simple first policy:

```text
uptake_policy(edge_conf) = flat_weight + edge_boost * edge_conf
```

where `flat_weight` is small but nonzero on stable textured planes, and
`edge_boost` increases high-frequency capacity around reliable edges.

## 5. Edge Definition

PSE-DC should support multiple edge signals, ordered from easiest to strongest:

1. Mesh normal discontinuity:
   high dihedral angle or high variation of adjacent face normals.
2. Mesh curvature / local normal variance:
   high curvature gets stronger edge confidence.
3. Depth discontinuity:
   projected rendered depth or monocular depth has a strong gradient.
4. Visibility/silhouette instability:
   Gaussian is repeatedly near alpha/depth boundaries across views.
5. SR-LR edge agreement:
   SR adds high-frequency signal where LR/render already has compatible
   low-frequency structure.

The key design choice is that `edge_conf` is not only a place to sharpen. It is
also a boundary where smoothness should stop.

## 6. Losses

### 6.1 Surface Carrier Formation Loss

For Gaussians selected as carriers, keep them thin and attached to the surface:

```text
L_attach = mean_i surface_conf_i * | dot(x_i - anchor_i, normal_i) - target_n |
L_tangent = mean_i weak_tangent_weight * || tangent(x_i - anchor_i) ||
L_thin = mean_i max(scale_normal_i / mean(scale_tangent_i) - tau_thin, 0)
L_align = mean_i (1 - | dot(thin_axis_i, normal_i) |)
```

This turns reinitialized Gaussians into thin surface carriers instead of thick
volumetric blobs.

Implementation hook:

- Existing `SurfaceMigrationRegularizer` already covers normal attachment,
  tangent pull, normal alignment, and thinness.
- Existing `SurfaceNormalLock` can prevent carrier drift along the normal after
  optimizer steps.

### 6.2 Surface-Routed Detail Uptake Loss

Let `HF(.)` be a Laplacian or high-pass residual. Let `P` be SR prior and `R`
be the current render.

```text
L_detail = sum_p W_surface_edge(p) * | HF(R)(p) - HF(P)(p) |
```

`W_surface_edge` is rendered from per-Gaussian `gate_i`, so gradients are routed
mainly to surface carriers. This is the important difference from ordinary 2D
prior loss: high-frequency supervision is not allowed to update arbitrary
off-surface Gaussians.

Implementation hook:

- Existing `LayerFrequencyRegularizer` already routes surface high-frequency
  gradients to `surface_carrier`.
- For prior-from-scratch, set `layer_frequency_surface_target=prior` for this
  term, while retaining LR/anchor low-frequency guards.

### 6.3 Non-Surface High-Frequency Suppression

Non-surface Gaussians should not explain SR hallucinations:

```text
L_non_surface = mean_p | HF(Render(non_surface_gaussians))(p) |
```

Optionally add alpha high-frequency and alpha mass penalties for non-surface
regions if floating shells appear.

Implementation hook:

- Existing `lambda_non_surface_hf` and related NoSR parameters already express
  this.

### 6.4 Edge-Stopped Surface Continuity

For neighboring carriers `i,j` on the same smooth patch:

```text
L_cont = sum_(i,j) w_ij * || uptake_i - uptake_j ||
```

with:

```text
w_ij = same_patch_ij * exp(-normal_angle_ij / sigma_n) * (1 - edge_barrier_ij)
```

This smooths detail uptake on smooth surfaces, but it explicitly stops across
sharp edges. That gives us the paper claim: continuity is encouraged only where
the scene prior says the surface is smooth.

First implementation can approximate this with per-face or kNN carrier groups.
It does not need to be differentiable in the graph at v0; it can precompute
edge-stopped neighbor pairs and then apply a regularization term.

### 6.5 Edge-Aware Densification

Carrier capacity should increase near reliable edges:

```text
densify_score_i =
    base_grad_i
  * (1 + beta_edge * edge_conf_i)
  * surface_conf_i
  * consistency_conf_i
```

Child Gaussians near edges should be initialized as thin surface splats:

```text
child_normal_axis = anchor_normal
child_tangent_axes = edge_direction, normal x edge_direction
child_scale_normal << child_scale_tangent
```

If no stable edge direction is available, use the local tangent basis and rely
on normal-lock plus thinness.

## 7. Training Pipeline

### Stage A: Prior Scratch Baseline

Generate enhancement-SR priors, prepare aligned prior cache, and train the
canonical prior-from-scratch model.

This is the current baseline and should remain unchanged as a control.

### Stage B: Surface Payload Bootstrap

Build a surface-state payload for the scratch model early or from an LR/Mip
anchor model:

```text
scratch or LR model -> mesh query -> surface_state payload
```

For v0, reuse the existing NoSR classifier:

```text
run_classify_mip_surface_state_v0_kitchen.sh
```

### Stage C: PSE-DC Scratch Training

Train from scratch or continue from an early scratch checkpoint using
`hybrid_sdfgs.train`, enabling:

```text
surface_normal_lock=1
surface_migration_payload=<surface_state>
layer_frequency_mask_payload=<surface_state>
layer_frequency_surface_key=surface_carrier
layer_frequency_non_surface_key=non_surface_active
layer_frequency_surface_target=prior
lambda_non_surface_hf > 0
lambda_surface_hf_closure > 0
```

At v0, keep densification conservative until masks can support dynamic roots.

### Stage D: Optional NoSR Cleanup

After the PSE-DC model is trained, run the preserved NoSR cleanup as the final
artifact suppressor.

This keeps the strong `~28.57` NoSR behavior while testing whether PSE-DC makes
the reinitialized surface sharper before cleanup.

## 8. Minimal v0 We Should Build First

Do not start with the full edge-aware densification. The first meaningful v0 is:

1. Use `hybrid_sdfgs.train` instead of plain `train.py` for the prior scratch
   branch.
2. Build a surface-state payload for the scratch initialization or LR anchor.
3. Enable:
   `surface_normal_lock`, `surface_migration`, `lambda_non_surface_hf`, and
   `lambda_surface_hf_closure`.
4. Set `layer_frequency_surface_target=prior`.
5. Disable or restrict densification if fixed masks are used.
6. Evaluate whether surface blur decreases before final NoSR cleanup.

This v0 tests the main hypothesis:

> Surface-constrained carrier formation plus 3D-routed SR detail produces a
> sharper reinitialized surface than ordinary prior-only scratch training.

## 9. v1 Extensions

After v0 works, add:

- `edge_conf` payload from mesh dihedral / normal variance.
- Edge-boosted detail uptake.
- Edge-stopped continuity over carrier neighbors.
- Edge-aware densification and child orientation.
- Dynamic-root support so masks survive splitting and pruning.

## 10. Ablations

Minimum paper ablations:

1. Prior-only scratch.
2. Prior-only scratch + NoSR cleanup.
3. PSE-DC v0 without edge boost.
4. PSE-DC v0 + NoSR cleanup.
5. PSE-DC with edge boost.
6. PSE-DC with edge boost + edge-aware densification.

Useful diagnostic metrics:

- PSNR / SSIM / LPIPS.
- Edge-band PSNR around depth or mesh edges.
- Surface-vs-non-surface high-frequency energy ratio.
- Rendered sharpness on surface carriers only.
- Floating-HF artifact count.
- Thickness statistics: normal scale / tangent scale.
- Cross-edge bleeding score: high-frequency leakage across depth/normal
  discontinuities.

## 11. Expected Failure Modes

- Bad mesh causes wrong carriers:
  use confidence gating and keep low-frequency LR/anchor supervision active.
- Over-strong continuity blurs texture:
  stop continuity across edges and keep it weak on high-texture flat regions.
- Surface lock prevents needed geometry correction:
  schedule normal lock from weak to strong, or use migration only after early
  geometry settles.
- Fixed masks conflict with densification:
  start with fixed topology, then move to dynamic-root masks once v0 is
  validated.
- SR hallucination is cross-view inconsistent:
  combine PSE-DC with prior masks and low-frequency agreement checks.

## 12. Clean Method Statement

PSE-DC transforms mesh/depth information from a passive geometric target into
an active detail-routing prior. Instead of asking all Gaussians to fit SR
outputs, it forms thin surface carriers on piecewise-smooth regions, boosts
carrier capacity near reliable sharp edges, suppresses off-surface
high-frequency uptake, and prevents smoothness from crossing discontinuities.

This should be the main 3D-domain innovation on top of enhancement-SR priors
and the preserved NoSR cleanup stage.

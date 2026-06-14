# Surface-Edge Prior Idea v0

## Goal

Keep the current NoSR cleanup stage, but add one extra novelty that is truly
3D-prior-driven rather than "better 2D SR supervision only".

The proposed direction is:

**Surface-Edge Carrier Prior (SECP)**  
Use the existing NoSR mesh / surface-state payload to decide:

1. where high-frequency prior is allowed to live,
2. where it should be suppressed,
3. how it should move along the surface during optimization.

This turns SR priors from a plain image target into a **3D routed signal**.

## Why This Is Different

Current prior usage is still mostly "render vs prior image" plus cleanup.
Current NoSR already gives us something stronger:

1. a surface / non-surface partition,
2. mesh continuity,
3. boundary / edge geometry,
4. a post-process that already knows many bad HF artifacts are off-surface or on the wrong layer.

The new claim is:

**We do not only denoise bad SR detail after it appears.  
We use a 3D surface-edge prior to constrain which Gaussians may absorb SR detail in the first place.**

## Core Representation

Build a per-Gaussian or per-surface-point payload with:

1. `surface_confidence`
   from the existing mesh/surface support logic.
2. `edge_confidence`
   from projected mesh edges, dihedral angle, curvature, or depth/normal discontinuity.
3. `tangent_frame`
   surface normal plus two tangent axes.
4. `continuity_confidence`
   whether this point belongs to a stable connected surface carrier, not a floating bubble.

Then define an **SR uptake gate**:

`gate_sr = surface_confidence * continuity_confidence * edge_or_texture_policy`

where `edge_or_texture_policy` can favor:

1. strong uptake near reliable geometric edges,
2. weaker uptake on flat surfaces,
3. near-zero uptake on non-surface or unstable carriers.

## Method Block

### 1. Surface-Edge Guided Prior Loss

Instead of supervising all rendered HF equally, weight the prior residual by
surface ownership and edge reliability:

`L_secp = sum_i gate_sr(i) * || HF(render_i) - HF(prior_i) ||`

This says:

1. SR detail is trusted mostly on stable surface carriers.
2. Geometric edges get higher prior budget than interior ambiguous texture.
3. Off-surface Gaussians cannot freely chase hallucinated SR detail.

### 2. Tangent-Constrained HF Migration

When prior residual pushes a Gaussian, decompose the update into:

1. tangent motion,
2. normal motion.

Use a strong bias toward tangent transport on stable surfaces:

`delta_x = alpha_t * delta_tangent + alpha_n * delta_normal`

with `alpha_t >> alpha_n` on surface carriers.

Effect:

1. detail slides along the mesh instead of inflating outward,
2. edge sharpness can improve without creating thick fuzzy shells,
3. the prior behaves more like a surface texture carrier than a volumetric blob source.

### 3. Edge-Aware Densification

Use the mesh edge prior to drive split / clone decisions:

1. split more aggressively on high `edge_confidence` carriers,
2. reduce birth on low-confidence floating non-surface regions,
3. orient child Gaussians along tangent directions aligned with edge flow.

This is stronger than generic geometry-aware densification because it is not
just "near the surface"; it is specifically "near reliable surface edges or
high-curvature carriers".

### 4. Continuity-Preserving HF Closure

Project local HF uptake onto connected mesh neighborhoods.
If one small region wants to absorb large HF prior but its geodesic neighbors do
not support it, penalize that inconsistency:

`L_cont = sum_(i,j in mesh-neighbors) w_ij * || uptake_i - uptake_j ||`

This turns mesh continuity into a filter against isolated hallucinated spikes.

## Clean Paper Story

The method story can be:

1. **Enhancement SR priors** give higher-PSNR, lower-hallucination image guidance.
2. **NoSR cleanup** remains the final artifact suppressor.
3. **SECP** is the new 3D bridge between them:
   mesh continuity and geometric edges decide how SR detail enters the 3D field.

That gives a neat separation:

1. diffusion SR: optional visual-detail branch,
2. enhancement SR: PSNR-oriented branch,
3. SECP: shared 3D prior mechanism that makes either branch safer in 3D.

## Minimal First Ablation

The smallest convincing experiment is:

1. enhancement-SR prior scratch model,
2. enhancement-SR prior + NoSR cleanup,
3. enhancement-SR prior + NoSR cleanup + SECP gate,
4. enhancement-SR prior + NoSR cleanup + SECP gate + edge-aware densify.

Measure:

1. PSNR / SSIM,
2. edge-band PSNR,
3. starburst / floating-HF artifact count,
4. surface-vs-non-surface HF energy ratio.

## Most Likely Strong Name

The safest naming options:

1. `Surface-Edge Carrier Prior`
2. `Mesh-Routed SR Prior`
3. `Surface-Constrained Detail Uptake`

The first one is probably the cleanest because it directly matches the
mechanism and sounds like a real method block.

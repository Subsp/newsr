# Pseudo-SDF vs True SDF (2D)

This folder contains a small matplotlib demo to visualize why a GS-bootstrapped pseudo-SDF is not a globally valid SDF.

## Run

```bash
cd /Users/ltl/Desktop/codex_playground/mip-splatting/hybrid_sdfgs/visualizations/pseudo_vs_true_sdf_2d
python plot_pseudo_vs_true_sdf_2d.py --out_dir .
python plot_pseudo_vs_true_sdf_2d_complex.py --out_dir . --region_sparse_keep 1.0 --out_name pseudo_vs_true_sdf_2d_complex.png
# Optional: stronger or weaker local thinning of anchors in one region.
python plot_pseudo_vs_true_sdf_2d_complex.py --out_dir . --region_sparse_keep 0.20

# Test hypothesis: enlarge sparse-anchor support to fill missing field.
python plot_pseudo_sdf_support_expansion_compare.py --out_dir .

# Try tangential-only support expansion in sparse region.
python plot_pseudo_sdf_tangential_expansion_compare.py --out_dir .

# Try normal-consistency gating + adaptive tangential expansion.
python plot_pseudo_sdf_gated_adaptive_compare.py --out_dir .

# Sweep where gated+adaptive beats baseline in a two-Gaussian setup.
python sweep_pseudo_sdf_gated_adaptive_two_gaussian.py --out_dir .

# Introduce 1D SR prior into toy scene (analytic source).
python plot_pseudo_sdf_srprior_real_slice_compare.py \
  --out_dir . \
  --scene_source analytic \
  --prior_mode sr \
  --noise_deg 20 \
  --prior_weight_sr 0.35 \
  --prior_conf_gain 0.15

# Better balanced prior fusion (analytic), usually more stable:
python plot_pseudo_sdf_srprior_real_slice_compare.py \
  --out_dir . \
  --scene_source analytic \
  --prior_mode lr+sr \
  --noise_deg 20 \
  --prior_weight_lr 0.20 \
  --prior_weight_sr 0.35 \
  --prior_conf_gain 0.18

# Realistic toy profile: mimic real GS traits (non-uniform density, shell thickness,
# overlap, leakage, broken closure, normal corruption, long-tail confidence).
python plot_pseudo_sdf_srprior_real_slice_compare.py \
  --out_dir . \
  --scene_source analytic \
  --analytic_profile realistic \
  --realistic_preset mild \
  --prior_mode lr+sr \
  --noise_deg 16 \
  --out_name pseudo_sdf_srprior_real_slice_compare_realistic_toy.png \
  --seq_out_name pseudo_sdf_srprior_sequence_realistic_toy.png

# Harder realistic preset (stronger corruption/sparsity).
python plot_pseudo_sdf_srprior_real_slice_compare.py \
  --out_dir . \
  --scene_source analytic \
  --analytic_profile realistic \
  --realistic_preset hard \
  --prior_mode lr+sr \
  --noise_deg 16 \
  --out_name pseudo_sdf_srprior_real_slice_compare_realistic_hard_toy.png \
  --seq_out_name pseudo_sdf_srprior_sequence_realistic_hard_toy.png

# Use custom realistic parameters instead of preset:
# --realistic_preset custom and tune --real_* flags manually.

# Use a real 3D mesh slice to make the toy scene more realistic:
python plot_pseudo_sdf_srprior_real_slice_compare.py \
  --out_dir . \
  --scene_source mesh \
  --mesh_path /path/to/scene_mesh.ply \
  --plane_origin 0,0,0 \
  --plane_normal 0,0,1 \
  --prior_mode sr

# Use real LR Gaussian field (good_fused.ply), crop a local block, then slice:
python plot_pseudo_sdf_srprior_gaussian_slice_compare.py \
  --out_dir . \
  --gaussian_ply_path /Users/ltl/Desktop/codex_playground/good_fused.ply \
  --plane_origin 0,0,0 \
  --plane_normal 0,0,1 \
  --crop_center 0,0,0 \
  --crop_extent 1.2,1.2,1.2 \
  --prior_mode lr+sr

# Shovel-focused zoomed slice (example tuned on good_fused.ply):
python plot_pseudo_sdf_srprior_gaussian_slice_compare.py \
  --out_dir . \
  --gaussian_ply_path /Users/ltl/Desktop/codex_playground/good_fused.ply \
  --plane_origin 0.03,0.82,0.66 \
  --plane_normal 0,0,1 \
  --crop_center 0.03,0.82,0.66 \
  --crop_extent 0.60,0.45,0.40 \
  --opacity_quantile 0.30 \
  --prior_mode lr+sr \
  --noise_deg 24 \
  --out_name pseudo_sdf_srprior_gaussian_slice_compare_shovel_zoom.png \
  --seq_out_name pseudo_sdf_srprior_gaussian_slice_sequence_shovel_zoom.png
```

## Output

- `pseudo_vs_true_sdf_2d.png`
  - Top-left: true SDF
  - Top-right: pseudo-SDF from sparse noisy anchors
  - Bottom-left: absolute error
  - Bottom-right: pseudo field gradient norm (`|∇f|`, ideal SDF should be near 1)

- `pseudo_vs_true_sdf_2d_complex.png`
  - Top-left: true SDF (shape with corners + curved regions)
  - Top-right: pseudo-SDF from sparse/noisy anchors with missing regions
  - Bottom-left: absolute error
  - Bottom-right: pseudo field gradient norm (`|∇f|`, ideal SDF should be near 1)

- `pseudo_vs_true_sdf_2d_complex_sparse_region.png` (default output of the complex script)
  - Same panels as above, but with **extra local thinning** in one spatial region.
  - Yellow anchors: normal density; orange anchors: survived in the thinned region.

- `pseudo_sdf_support_expansion_compare.png`
  - Compare baseline sparse pseudo-SDF vs sparse-region support expansion.
  - Includes global/sparse-region MAE in titles and an error-difference map.

- `pseudo_sdf_tangential_expansion_compare.png`
  - Compare baseline sparse pseudo-SDF vs **tangential-only** support expansion.
  - Expansion acts along tangent direction, keeps normal direction scale unchanged.

- `pseudo_sdf_gated_adaptive_compare.png`
  - Compare baseline sparse pseudo-SDF vs **normal-consistency gating + adaptive tangential expansion**.
  - Gaussian centers are overlaid in top-row subplots.

- `two_gaussian_sweep_gated_adaptive.png`
  - 2x2 heatmaps over many settings:
  - `distance(separation) x noise` and `distance x outlier-tilt`.
  - Each includes mean ΔMAE (`baseline - gated`) and win rate.
  - CSV tables are also exported in the same folder.

- `pseudo_sdf_srprior_real_slice_compare.png`
  - Baseline (no prior) vs prior-guided pseudo-SDF.
  - Supports `prior_mode`: `none | lr | sr | lr+sr`.
  - Supports scene source: analytic toy or real 3D mesh slice.
  - Gaussian centers are overlaid.

- `pseudo_sdf_srprior_sequence.png`
  - 1D sequence view over sorted anchors:
  - true normal sequence / base noisy sequence / LR prior / SR prior.

- `pseudo_sdf_srprior_gaussian_slice_compare.png`
  - Reference field from real LR Gaussian point-cloud slice.
  - Baseline vs prior-guided pseudo-SDF on the same cropped/sliced local block.

- `pseudo_sdf_srprior_gaussian_slice_sequence.png`
  - 1D sequence with **spatial crop visibility** (outside cropped region marked).

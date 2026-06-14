# Training Diagnostics v0

This note records the first training-time diagnostic pass added to `train.py`.

## Goal

Add two cheap diagnostics without changing the main optimization target:

1. `gradient tracking`
2. `2D dropout -> gradient consistency`

Both are meant to help find regions that are:

- still being pushed in a stable direction
- underconstrained and view-sensitive
- likely to become bad geometry later

## Phase Logic

The current `early4k_soft` schedule is not stable over the full first 30k steps.

Important switches:

- densify starts at `500`
- distortion / opacity-field / extent turn on at `distortion_from_iter`
- depth-normal / smoothness turn on at `depth_normal_from_iter`
- densify stops at `densify_until_iter`

Because of this, the diagnostics default to:

`start_iter = max(densify_until_iter, distortion_from_iter, depth_normal_from_iter)`

So by default they start after the representation topology and the main regularizers are both relatively stable.

This behavior can be overridden with:

- `--gradient_tracking_from_iter`
- `--dropout_diagnostic_from_iter`

## Basis

Two basis modes are supported:

- `gaussian_frame`
  Uses the smallest local Gaussian scale axis as a normal proxy. This is the default and does not need extra payloads.

- `surface_payload`
  Uses `nearest_surface_normal` from a saved payload such as `select_mesh_outside_gaussians_v0.py`.

## New CLI Flags

Core:

- `--diagnostic_output_subdir`
- `--diagnostic_basis_mode`
- `--diagnostic_surface_payload`
- `--diagnostic_disable_phase_reset`

Gradient tracking:

- `--enable_gradient_tracking`
- `--gradient_tracking_from_iter`
- `--gradient_tracking_snapshot_interval`
- `--gradient_tracking_tile_size`

2D dropout:

- `--enable_2d_dropout_diagnostic`
- `--dropout_diagnostic_from_iter`
- `--dropout_diagnostic_interval`
- `--dropout_diagnostic_num_masks`
- `--dropout_diagnostic_tile_size`
- `--dropout_diagnostic_keep_ratio`
- `--dropout_diagnostic_loss_mode`
- `--dropout_diagnostic_alpha_threshold`
- `--dropout_diagnostic_min_active_pixels`

## Export Layout

Each exported snapshot lives under:

`<model_path>/<diagnostic_output_subdir>/iter_XXXXXX_<camera_name>/`

Files:

- `snapshot.pt`
  Raw payload with per-Gaussian visible ids, projected coordinates, per-view metrics, and expanded tile maps.

- `summary.json`
  Scalar summary for the snapshot.

- `gt.png`
- `render.png`

- `grad_current_signed_overlay.png`
- `grad_running_bias_overlay.png`
- `grad_running_jitter_overlay.png`
- `grad_running_flip_overlay.png`
- `grad_running_dominance_overlay.png`

- `dropout_bias_overlay.png`
- `dropout_std_overlay.png`
- `dropout_sign_agreement_overlay.png`
- `dropout_coverage_overlay.png`

- `contact_sheet.png`
  Quick overview of the available overlays.

## Offline Aggregation

Use:

```bash
python visualize_training_diagnostics_v0.py \
  --diagnostic_dir /path/to/output/training_diagnostics_v0 \
  --latest_n 12
```

This creates:

`<diagnostic_dir>/aggregate_view_v0/`

with:

- `manifest.json`
- `contact_sheet_contact_sheet.png` by default

# standalone_x8to2_prior

Standalone first-pass prior fusion pipeline for MipNeRF360-style multi-scale image folders.

Goal:

- Input: `images_8`
- Output: `images_2`-sized fused renders
- Prior: external video-SR directory stored separately
- Evaluation: compare fused outputs against `images_2`

This directory is intentionally decoupled from the existing GS / hybrid training code.

## What it does

For each frame:

1. Load LR input from `images_8`
2. Upsample to `images_2` size with bicubic interpolation
3. Load the external prior frame with the same image stem
4. Run a lightweight multi-level Haar DWT fusion
5. Keep the bicubic low-frequency structure
6. Inject only gated high-frequency prior residuals
7. Save:
   - `renders/`: fused outputs
   - `bicubic/`: bicubic baseline
   - `gt/`: symlink or copied `images_2` directory when available
   - `manifest.json`: frame mapping and settings
   - `quick_eval.json`: PSNR / SSIM / MAE for bicubic and fused if GT exists

This is a data-level image experiment, not a training pipeline.

## Usage

Run from the `mip-splatting` repo root:

```bash
python hybrid_sdfgs/standalone_x8to2_prior/run_x8to2.py \
  --scene_root /path/to/mipnerf360/scene \
  --prior_dir /path/to/video_sr_priors \
  --output_dir /path/to/output/x8to2_firstpass
```

By default it expects:

- input: `<scene_root>/images_8`
- gt: `<scene_root>/images_2`

## Existing metrics script

After generation, you can reuse the existing GS-SDF metrics script:

```bash
python hybrid_sdfgs/standalone_x8to2_prior/eval_with_existing_metrics.py \
  --output_dir /path/to/output/x8to2_firstpass
```

This calls:

```bash
python /path/to/GS-SDF/eval/image_metrics/metrics2.py \
  --gt_color_dir .../gt \
  --renders_color_dir .../renders
```

The wrapper auto-resolves the workspace root and uses
`/Users/ltl/Desktop/codex_playground/GS-SDF/eval/image_metrics/metrics2.py`
by default, but you can override it with `--metrics_script`.

## Notes

- Matching is done by image stem first.
- If stem matching fails but file counts match, it falls back to sorted-order pairing.
- If `images_2` is missing, target size is inferred as `scale x` the LR size.
- The fusion is conservative on purpose: it keeps bicubic low-frequency content and only adds gated prior high-frequency residuals.

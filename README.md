# VGGTSR Mainline Extraction

This repo is a copy-only extraction created on 2026-06-14 from the larger
`VGGTSR` workspace.

It is intentionally centered on the two actively supported pipelines:

- the fixed-topology NoSR cleanup mainline
- the prior-from-scratch mainline

Enhancement-style SR priors are supported as the current wrapper for the
prior-from-scratch branch.

## Supported entrypoints

Canonical NoSR mainline:

- `SOF/scripts/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh`

Canonical prior-from-scratch mainline:

- `SOF/scripts/run_mipsplatting_prior_only_from_scratch_v0_kitchen.sh`

Supported enhancement-SR wrapper:

- `SOF/scripts/run_mipsplatting_enhancement_prior_scratch_v0_kitchen.sh`

Supported same-size render-restoration wrapper:

- `SOF/scripts/run_mipsplatting_render_restoration_prior_scratch_v0_kitchen.sh`

The detailed rationale and ownership split is documented in:

- `SOF/MAINLINES_20260614.md`
- `SOF/20260614_render_x1_restoration_prior_support_v0.md`

## Repo contents

This extraction keeps:

- `SOF/`
- `mip-splatting/`

The copy intentionally excludes:

- nested `.git` directories
- Python cache directories and `.pyc` files
- `SOF/output/`
- `SOF/toy_outputs/`
- `mip-splatting/media/`

## Quick start

Run from `SOF/`:

```bash
bash scripts/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh
bash scripts/run_mipsplatting_prior_only_from_scratch_v0_kitchen.sh
RUN_NOSR_AFTER=1 bash scripts/run_mipsplatting_enhancement_prior_scratch_v0_kitchen.sh
SOURCE_IMAGES_SUBDIR=renders_lr_same_size ENHANCEMENT_BACKEND=nafnet bash scripts/run_mipsplatting_render_restoration_prior_scratch_v0_kitchen.sh
```

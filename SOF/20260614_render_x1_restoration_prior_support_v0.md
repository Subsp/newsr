# Render X1 Restoration Prior Support v0

Decision date: 2026-06-14

## Purpose

This branch supports same-resolution restoration priors for LR-trained 3DGS
renders.

It is intentionally separate from classical x4 image SR:

```text
images_8
  -> swinir/drct/mambairv2 x4 SR prior

target-sized LR 3DGS renders
  -> nafnet/restormer x1 restoration prior
```

The x1 branch is useful when the render is already aligned to the target camera
and resolution, but surfaces look blurred. The restoration model should sharpen
or deblur image-space evidence without changing the frame size.

## Supported Backends

`generate_enhancement_sr_priors.py` now accepts:

- `swinir`: in-repo classical x4 SR baseline.
- `nafnet`: external NAFNet x1 restoration.
- `restormer`: external Restormer x1 restoration.

The external repos are not vendored into this repo. Set their paths at runtime:

```bash
export NAFNET_ROOT=/root/autodl-tmp/external/NAFNet
export RESTORMER_ROOT=/root/autodl-tmp/external/Restormer
```

For NAFNet, the default option file is:

```text
options/test/REDS/NAFNet-width64.yml
```

Override it with:

```bash
EXTERNAL_RESTORATION_CONFIG=options/test/REDS/NAFNet-width64.yml
```

For Restormer, the default task is:

```text
Single_Image_Defocus_Deblurring
```

Override it with:

```bash
RESTORMER_TASK=Motion_Deblurring
```

## Generate Only

NAFNet:

```bash
python SOF/scripts/generate_enhancement_sr_priors.py \
  --input_dir /root/autodl-tmp/kitchen/renders_lr_same_size \
  --output_dir /root/autodl-tmp/kitchen/render_x1_priors_nafnet \
  --backend nafnet \
  --external_repo_root /root/autodl-tmp/external/NAFNet \
  --external_config options/test/REDS/NAFNet-width64.yml
```

Restormer:

```bash
python SOF/scripts/generate_enhancement_sr_priors.py \
  --input_dir /root/autodl-tmp/kitchen/renders_lr_same_size \
  --output_dir /root/autodl-tmp/kitchen/render_x1_priors_restormer \
  --backend restormer \
  --external_repo_root /root/autodl-tmp/external/Restormer \
  --restormer_task Single_Image_Defocus_Deblurring \
  --restormer_tile 720
```

## Full Prior-From-Scratch Wrapper

NAFNet render-restoration branch:

```bash
cd SOF
SOURCE_IMAGES_SUBDIR=renders_lr_same_size \
ENHANCEMENT_BACKEND=nafnet \
NAFNET_ROOT=/root/autodl-tmp/external/NAFNet \
EXTERNAL_RESTORATION_CONFIG=options/test/REDS/NAFNet-width64.yml \
bash scripts/run_mipsplatting_render_restoration_prior_scratch_v0_kitchen.sh
```

Restormer render-restoration branch:

```bash
cd SOF
SOURCE_IMAGES_SUBDIR=renders_lr_same_size \
ENHANCEMENT_BACKEND=restormer \
RESTORMER_ROOT=/root/autodl-tmp/external/Restormer \
RESTORMER_TASK=Single_Image_Defocus_Deblurring \
RESTORMER_TILE=720 \
bash scripts/run_mipsplatting_render_restoration_prior_scratch_v0_kitchen.sh
```

The wrapper forwards into:

```text
scripts/run_mipsplatting_enhancement_prior_scratch_v0_kitchen.sh
```

with x1-safe defaults:

```text
PREPARE_IMAGES8=0
DISABLE_PRIOR_USABLE_MASKS=1
PRIOR_MATCH_POLICY=order_if_needed
PRIOR_LLFFHOLD=8
RAW_PRIOR_SUBDIR=render_x1_priors_${ENHANCEMENT_BACKEND}
PREPARED_SR_PRIOR_NAME=render_x1_${ENHANCEMENT_BACKEND}_aligned_images_2_scratch_v0
PRIOR_ONLY_RUN_TAG=mip30k_r1_renderx1_${ENHANCEMENT_BACKEND}_prioronly_scratch_v0
```

## Notes

- The source render directory must contain one image per frame stem.
- Output priors are saved as flat `<stem>.png` files.
- The render-restoration wrapper disables usable masks by default, so
  `fused_priors` are direct copies of the restored priors rather than
  mask-blended prior/reference images.
- Direct x1 render prior stems can differ from `images_2` stems; the wrapper
  therefore uses `PRIOR_MATCH_POLICY=order_if_needed`. It first tries stem
  matching, then full sorted order matching, then LLFF/Mip-NeRF360 train-order
  matching with `PRIOR_LLFFHOLD=8` when the prior count equals the non-holdout
  reference count. Outputs are written with reference frame stems.
- Restormer runs selected frames in one batch call, writes into a task
  subdirectory internally, and the wrapper copies restored images back into the
  expected flat prior cache.
- Restormer also supports Hugging Face TorchScript defocus weights through
  `RESTORMER_CHECKPOINT_MODE=auto` or `torchscript`; this avoids depending on
  Google Drive checkpoint-dict downloads.
- NAFNet currently uses the official single-image `basicsr/demo.py` path for
  reliability. A batch runner can be added after the server environment is
  fixed.
- These priors should be evaluated separately from x4 SR priors because they
  start from already view-aligned render evidence.

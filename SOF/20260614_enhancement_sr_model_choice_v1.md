# Enhancement SR Model Choice v1

Decision date: 2026-06-14

## Decision

Use `HAT` classical image super-resolution as the first non-diffusion
high-frequency prior generator.

Preferred checkpoint family:

- `HAT_SRx4_ImageNet-pretrain.pth` or the strongest available classical SRx4
  HAT checkpoint.

Avoid as the primary PSNR-oriented branch:

- diffusion SR models;
- GAN/perceptual models such as `Real-ESRGAN` or `Real_HAT_GAN_SRx4_sharper`.

GAN/perceptual variants may remain useful for visual-only experiments, but they
should not be the first prior source for geometry-aware 3DGS training.

## Why HAT

The new prior branch is not asking for the most visually dramatic SR output.
It needs a prior that is:

- deterministic;
- PSNR/fidelity oriented;
- less hallucination-prone than diffusion or GAN SR;
- strong enough to inject high-frequency detail;
- practical to batch over scene frames;
- compatible with tile inference for limited GPU memory.

HAT fits this better than the alternatives:

- Compared with `SwinIR`, HAT is a stronger transformer SR model and was
  designed to activate more input pixels through hybrid attention and
  overlapping cross-attention.
- Compared with `DAT`, HAT has a very usable official repo, model zoo, and
  BasicSR-style inference path.
- Compared with `Real-ESRGAN`, HAT classical SR is more appropriate for our
  PSNR-oriented and multi-view-consistency-sensitive prior branch.
- Compared with diffusion SR, HAT is deterministic and avoids random
  generative details that tend to become cross-view inconsistent in 3DGS.

## How It Fits Our Pipeline

The intended chain is:

```text
images_8
  -> HAT classical SRx4 raw priors
  -> prepared aligned SR prior cache
  -> prior-from-scratch 3DGS training
  -> PSE-DC surface/edge detail carrier training
  -> optional NoSR cleanup
```

Naming convention:

```text
ENHANCEMENT_BACKEND=hat
RAW_PRIOR_SUBDIR=priors_hat
PREPARED_SR_PRIOR_NAME=hat_aligned_images2_scratch_v0
```

## First Baseline

Run three baselines before adding more SR models:

1. `swinir` existing wrapper, as a compatibility baseline.
2. `hat` classical SRx4, as the main selected model.
3. `hat` classical SRx4 + PSE-DC v0, to test whether the 3D surface prior can
   absorb HAT detail without surface blur.

The first pass should compare:

- PSNR / SSIM / LPIPS;
- edge-band PSNR;
- cross-view flicker or inconsistency around high-frequency textures;
- surface-vs-non-surface high-frequency energy ratio;
- qualitative blur on surface-carrier-only renders.

## Fallback

If HAT classical priors are too smooth, test `HAT` real-world fidelity variant
before trying any GAN-sharpened model.

If HAT installation is too heavy for the training machine, use `SwinIR`
classical SRx4 as the temporary implementation fallback, but keep HAT as the
method target.

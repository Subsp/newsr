# 20260512 Kitchen 00000 Init-Repair Prune-More v1 Baseline

## Status

`view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1_mip_hr_anchor_v0_miphr_v1` is the current accepted single-view baseline for kitchen view `00000`.

This is the render downloaded as `00000 (22).png`.

It is better than the later `initrepair_softbright_aggressive_v0` render (`00000 (23).png` / `00000 (24).png`) on this view.

## Output Paths

Render:

```text
/root/autodl-tmp/SOFSR/output/recover_cleaned_mip_lr_v0/kitchen/view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1_mip_hr_anchor_v0_miphr_v1/recovered_mip_renders_no_gt_hr_v0/test/ours_31600/renders/00000.png
```

Recovered model:

```text
/root/autodl-tmp/SOFSR/output/recover_cleaned_mip_lr_v0/kitchen/view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1_mip_hr_anchor_v0_miphr_v1/recovered_mip_model_lr_miphr_v1
```

Cleanup model:

```text
/root/autodl-tmp/SOFSR/output/cleanup_mip_view_aligned_volume_artifacts_v0/kitchen/view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1/cleaned_mip_model_view_volume_v1
```

## Repro Script

Use:

```bash
SCENE_ROOT=/root/autodl-tmp/kitchen \
bash scripts/run_cleanup_mip_view_aligned_volume_artifacts_initrepair_v0_prune_more_v1_kitchen.sh
```

Equivalent explicit settings:

```bash
SCENE_ROOT=/root/autodl-tmp/kitchen \
MIP_MODEL_PATH=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/kitchen_mip_vanilla_images8_v1/mip30k_sof_native_input_init_repair_v0 \
RUN_NAME=view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1 \
LR_RECOVER_RUN_NAME=view_aligned_volume_delete_v1_initrepair_v0_prune_more_v1_mip_hr_anchor_v0_miphr_v1 \
RECOMPUTE_FILTER3D=0 \
DELETE_QUANTILE=0.930 \
MAX_PRUNE_FRACTION=0.120 \
MAX_OPACITY=1.0 \
RUN_RENDER=1 \
RUN_LR_RECOVER=1 \
bash scripts/run_cleanup_mip_view_aligned_volume_artifacts_v0_kitchen.sh
```

## 00000 Single-View Metrics

Reference image: `/Users/ltl/Downloads/00000gt.png`.

Candidate image: `/Users/ltl/Downloads/00000 (22).png`.

Image size: `1558 x 1039`.

Patch heatmap window: `64 px`.

Global PSNR:

```text
RGB : 24.603800 dB
Low : 30.740283 dB
Mid : 32.124340 dB
High: 29.741415 dB
```

Local patch PSNR mean:

```text
RGB : 27.397404 dB
Low : 35.362877 dB
Mid : 36.425182 dB
High: 32.596004 dB
```

Versus `initrepair_softbright_aggressive_v0` (`00000 (23).png` / `00000 (24).png`), this baseline is better on view `00000`:

```text
RGB global delta 22-23 : +0.571107 dB
Low global delta 22-23 : +1.263175 dB
Mid global delta 22-23 : +0.462376 dB
High global delta 22-23: +0.035985 dB
```

Local RGB patch area:

```text
22 better than 23 by >0.5 dB: 37.789681%
23 better than 22 by >0.5 dB: 7.291683%
abs(delta) > 1 dB           : 24.059806%
```

Machine-readable metrics are recorded in:

```text
records/kitchen_00000_initrepair_v0_prune_more_v1_metrics.json
```

Heatmap artifacts generated locally:

```text
/private/tmp/vggtsr_psnr_23_22/contact_rgb_psnr_patch64.png
/private/tmp/vggtsr_psnr_23_22/contact_frequency_psnr_patch64.png
/private/tmp/vggtsr_psnr_23_22/overlay_delta_rgb_23_minus_22.png
```

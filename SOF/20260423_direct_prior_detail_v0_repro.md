# 20260423 Direct Prior Detail v0 复现记录

## 代码标识

当前成功路径对应代码分支:

```text
codex/prior-fusion-v0
```

记录时 HEAD:

```text
420e8f8 Record mesh patch observation workflow
d15fab2 Add mesh patch observation preparation
f75b1ed Add direct prior mask preparation
```

本次会额外保留 tag:

```text
exp/direct-prior-detail-v0-27p16
```

该 tag 用来标识 `direct_prior_detail_v0` 首次得到明显 PSNR/SSIM 提升的代码状态。

## 实验结论

## 20260424 当前最佳配置更新

当前 direct prior 主线最佳结果已经从最早的 `direct_prior_detail_v0`
更新为:

```text
direct_prior_detail_stronger_v1
```

指标:

```json
{
  "baseline": {
    "mean_psnr": 27.001565462246006,
    "mean_ssim": 0.7566523791353454,
    "n_views": 35
  },
  "current": {
    "mean_psnr": 27.21915271680064,
    "mean_ssim": 0.7680387810372707,
    "n_views": 35
  },
  "delta": {
    "psnr": 0.21758725455463335,
    "ssim": 0.011386401901925303
  }
}
```

关键改动参数相对 `direct_prior_detail_v0` 为:

```text
--lambda_prior_edge 0.3
--prior_edge_detail_alpha 0.4
--prior_edge_detail_alpha_final 0.7
--prior_edge_update_scale 0.5
```

其余 direct prior detail 相关参数保持与 v0 相同。

说明:

```text
当前最有效的提升来自更强的 detail prior 注入，
而不是单纯拉长训练时长。
```

实验名:

```text
direct_prior_detail_v0
```

核心结论:

```text
不使用 mesh，不使用 outlier GS 选择，不做 prior-only。
先用 2D direct prior mask 找低频一致但 prior 高频增强的区域，
再用 detail_v1 高频残差 loss 弱注入，原 SOF RGB/geometry loss 仍然保留。
```

指标结果:

```json
{
  "baseline": {
    "mean_psnr": 27.001565462246013,
    "mean_ssim": 0.7566523791353454,
    "n_views": 35
  },
  "current": {
    "mean_psnr": 27.16296014240031,
    "mean_ssim": 0.7652058458517683,
    "n_views": 35
  },
  "delta": {
    "psnr": 0.16139468015429514,
    "ssim": 0.008553466716422875
  }
}
```

指标文件:

```text
/root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_detail_v0_metrics.json
```

## 关键路径

baseline:

```text
/root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft
/root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft/chkpnt30000.pth
/root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft/test/ours_30000
```

StableSR prior:

```text
/root/autodl-tmp/priors/StableSRpriors
```

LR / anchor images:

```text
/root/autodl-tmp/kitchen/images_2
```

direct prior masks:

```text
/root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_masks_v0
/root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_masks_v0/direct_prior_masks
```

selected GS payload:

```text
/root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_gs_v0/edge_region_gaussians_v0.pt
```

training output:

```text
/root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_detail_v0
```

render output:

```text
/root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_detail_v0/test/ours_32000
```

## Mask 生成参数

生成脚本:

```text
prepare_direct_prior_masks_v0.py
```

命令:

```bash
python prepare_direct_prior_masks_v0.py \
  --prior_dir /root/autodl-tmp/priors/StableSRpriors \
  --anchor_dir /root/autodl-tmp/kitchen/images_2 \
  --output_dir /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_masks_v0 \
  --blur_kernel 9 \
  --lowfreq_threshold 0.08 \
  --highfreq_gain_threshold 0.015 \
  --prior_highfreq_threshold 0.02 \
  --confidence_threshold 0.15 \
  --dilate_kernel 3
```

默认参数同时生效:

```text
confidence_power = 1.0
erode_kernel = 1
debug_limit = 32
mask_suffix = _inject.png
```

mask 含义:

```text
M(x)=1 当且仅当:

lowfreq prior 与 anchor 足够接近
prior 高频强于 anchor
prior 本身有足够高频
confidence >= threshold
```

数学形式:

```text
L(x) = blur(x)
H(x) = x - L(x)

|L(P)(x) - L(A)(x)| <= tau_low
|H(P)(x)| - |H(A)(x)| >= tau_gain
|H(P)(x)| >= tau_prior
C(x) >= tau_conf
```

其中:

```text
P = StableSR prior
A = LR / anchor image
```

## GS 选择参数

脚本:

```text
select_edge_region_gaussians_v0.py
```

这里虽然复用了 edge selector 名字，但输入 mask 是 direct prior mask，不是 mesh edge mask。

命令:

```bash
python select_edge_region_gaussians_v0.py \
  -s /root/autodl-tmp/SOFSR/data_aliases/kitchen_images8bicubic_to_images2 \
  -m /root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft \
  --eval \
  --data_device cpu \
  --iteration 30000 \
  --start_checkpoint /root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft/chkpnt30000.pth \
  --edge_mask_dir /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_masks_v0/direct_prior_masks \
  --output_dir /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_gs_v0 \
  --min_touch_views 2 \
  --min_visible_views 2 \
  --radius_scale 1.0 \
  --min_touch_radius_px 1 \
  --max_touch_radius_px 16
```

训练时使用:

```text
--optimize_gaussian_mask_payload /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_gs_v0/edge_region_gaussians_v0.pt
--optimize_gaussian_mask_key selected_mask
```

## 训练参数

命令:

```bash
python train.py \
  -s /root/autodl-tmp/SOFSR/data_aliases/kitchen_images8bicubic_to_images2 \
  -m /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_detail_v0 \
  --eval \
  --data_device cpu \
  --splatting_config /root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft/config.json \
  --start_checkpoint /root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft/chkpnt30000.pth \
  --iterations 32000 \
  --test_iterations 32000 \
  --save_iterations 32000 \
  --checkpoint_iterations 32000 \
  --distortion_from_iter 4000 \
  --depth_normal_from_iter 4000 \
  --lambda_distortion 200 \
  --lambda_depth_normal 0.02 \
  --lambda_smoothness 0.005 \
  --prior_edge_dir /root/autodl-tmp/priors/StableSRpriors \
  --prior_edge_mask_dir /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_masks_v0/direct_prior_masks \
  --lambda_prior_edge 0.2 \
  --prior_edge_loss_mode detail_v1 \
  --prior_edge_detail_alpha 0.4 \
  --prior_edge_detail_alpha_final 0.6 \
  --prior_edge_detail_warmup_iters 2000 \
  --prior_edge_detail_weight 1.0 \
  --prior_edge_lowfreq_weight 0.05 \
  --prior_edge_grad_weight 0.05 \
  --prior_edge_lowfreq_threshold 0.08 \
  --prior_edge_lowfreq_anchor gt \
  --prior_edge_detail_min_gain 0.005 \
  --prior_edge_confidence_power 1.5 \
  --prior_edge_update_scale 0.5 \
  --optimize_gaussian_mask_payload /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_gs_v0/edge_region_gaussians_v0.pt \
  --optimize_gaussian_mask_key selected_mask \
  --prior_edge_min_pixels 64 \
  --prior_edge_touch_min_radius_px 1 \
  --prior_edge_touch_radius_scale 1.0 \
  --prior_edge_touch_max_radius_px 16 \
  --densify_until_iter 0 \
  --port 6035 \
  2>&1 | tee /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_detail_v0.log
```

重要点:

```text
没有使用 --prior_only_edge_finetune
所以原 SOF rgb / distortion / depth-normal / smoothness loss 仍在。

densify_until_iter = 0
因此不新增 GS，只微调已有 GS。

lambda_prior_edge = 0.2
prior_edge_update_scale = 0.5
因此 prior 是弱引导。

prior_edge_lowfreq_anchor = gt
因此低频被原图/GT anchor 锁住。
```

## 渲染和评估

渲染:

```bash
python render.py \
  -m /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_detail_v0 \
  -s /root/autodl-tmp/kitchen \
  -i images_2 \
  --iteration 32000 \
  --eval \
  --skip_train \
  --data_device cpu
```

评估输入:

```text
baseline = /root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft/test/ours_30000
current  = /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/direct_prior_detail_v0/test/ours_32000
```

## 与其他路径对比

已知结果:

```text
early4k_soft baseline:
  PSNR 27.001565
  SSIM 0.756652

direct_prior_detail_v0:
  PSNR 27.162960
  SSIM 0.765206
  delta +0.161395 / +0.008553

outlier prior-only tau12:
  PSNR 26.809174
  SSIM 0.755057

outlier prior detail tau08 candidate:
  PSNR 25.997648
  SSIM 0.754563

surface-thin stronger_v1:
  PSNR 27.030759
  SSIM 0.758997
```

当前判断:

```text
direct prior detail 是目前最成功的 prior 注入路径。
成功原因不是 prior 强，而是:
1. mask 只允许低频一致区域;
2. loss 只注入高频残差;
3. 原 SOF 监督仍然保留;
4. GS update 被 selected_mask 限制;
5. prior 梯度 scale 只有 0.5。
```

## 后续调参空间

最值得优先扫:

```text
mask:
  lowfreq_threshold: 0.06 / 0.08 / 0.10
  highfreq_gain_threshold: 0.010 / 0.015 / 0.020
  confidence_threshold: 0.10 / 0.15 / 0.20

detail strength:
  lambda_prior_edge: 0.1 / 0.2 / 0.3
  alpha_final: 0.5 / 0.6 / 0.7
  update_scale: 0.25 / 0.5 / 0.75
  confidence_power: 1.0 / 1.5 / 2.0
```

推荐先固定训练参数，只扫 mask，因为这次成功大概率来自选区安全。

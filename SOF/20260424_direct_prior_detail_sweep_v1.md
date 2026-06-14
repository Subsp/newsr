# 20260424 Direct Prior Detail Sweep v1

## 基线

```json
{
  "mean_psnr": 27.001565462246006,
  "mean_ssim": 0.7566523791353454,
  "n_views": 35
}
```

baseline 路径:

```text
/root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft/test/ours_30000
```

## Sweep 结果

来源文件:

```text
direct_prior_detail_sweep_metrics.json
```

结果汇总:

| experiment | iter | PSNR | SSIM | delta PSNR | delta SSIM | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| direct_prior_detail_stronger_v1 | 32000 | 27.219153 | 0.768039 | +0.217587 | +0.011386 | 当前最好 |
| direct_prior_detail_updatescale075_v1 | 32000 | 27.153226 | 0.765224 | +0.151661 | +0.008572 | 有效但不如 stronger |
| direct_prior_detail_confpower2_v1 | 32000 | 27.153103 | 0.765289 | +0.151537 | +0.008636 | 与 updatescale075 基本同级 |
| direct_prior_detail_31000_v1 | 31000 | 27.145622 | 0.762642 | +0.144057 | +0.005989 | 起效较快 |
| direct_prior_detail_34000_v1 | 34000 | 27.132505 | 0.764555 | +0.130939 | +0.007902 | 延长训练收益有限 |
| direct_prior_detail_conservative_v1 | 32000 | N/A | N/A | N/A | N/A | 未完成 / 未渲染 |

## 结论

### 1. 当前最优主线

```text
direct_prior_detail_stronger_v1
PSNR = 27.21915271680064
SSIM = 0.7680387810372707
```

相对早期最成功的 `direct_prior_detail_v0`:

```text
v0:          27.162960 / 0.765206
stronger_v1: 27.219153 / 0.768039
```

即:

```text
在 direct prior detail 这条线上，继续增强 prior 注入强度是有效的。
```

### 2. 主要收益来自“更强注入”，不是“更久训练”

`31000 -> 32000 -> 34000` 没体现单调上升，说明:

```text
当前瓶颈不在训练时长，而在注入策略本身。
```

### 3. updatescale / confpower 仍有作用，但属于二级调参

`updatescale075_v1` 和 `confpower2_v1` 都明显优于 baseline，
但都没有超过 `stronger_v1`，说明:

```text
第一优先级仍然是 prior 强度 / alpha 调度 / mask 质量。
update_scale 与 confidence_power 更像是微调旋钮。
```


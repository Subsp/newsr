# 2026-02-28 实验记录：Hybrid FLUX LEGO（报错后完成训练）

## 1. 实验目的
验证 `hybrid_sdfgs` 在 `lego` 数据集上的 `Hybrid + FLUX prior` 训练链路是否可跑通，并记录报错后的最终结果。

## 2. 实验配置（对应 30 系列命令）
- 训练入口：`python hybrid_sdfgs/train.py`
- 数据集：`/data3/liutl/nerf_synthetic/lego`
- 输出目录：`/data3/liutl/HBSR/outputs/hybrid_flux_lego`
- 关键开关：
  - `--hybrid_enable --sdf_mode analytic`
  - `--flux_enable --flux_mode control --flux_model_name flux-dev-depth-lora`
  - `--flux_steps 30 --flux_guidance 10.0`
  - `--flux_offline`
  - `--flux_l1_weight 0.05 --flux_hf_weight 0.10`
  - `--iterations 30000`

## 3. 报错现象（训练中途曾出现）
在 FLUX prior 生成阶段出现离线缓存相关报错，核心点为：
- 离线模式下缺少 `LiheYoung/depth-anything-large-hf` 缓存文件；
- FLUX 子进程 prior 生成失败。

该问题属于 `flux-dev-depth-lora` 分支的依赖缺失问题（depth control 需要 depth backbone）。

## 4. 结果（本次提供日志片段）
训练最终跑满 30000 iter，并保存模型。关键指标如下：

- 训练总进度：`30000/30000`，耗时约 `33:47`
- 测试集（iter=30000）：
  - `L1 = 0.00557858660700731`
  - `PSNR = 34.86061625480652`
- 训练集（iter=30000）：
  - `L1 = 0.003960882080718875`
  - `PSNR = 38.14757232666016`

末轮损失日志（iter=30000）：
- `HYBRID`: `stage=C scale=0.350 surf=0.178186 norm=0.138356 off=0.286026 total=0.004604`
- `FLUX`: `total=0.001128 l1=0.003582 hf=0.009489 valid=1.000`

## 5. 结果解读与注意事项
- 从结果看，训练链路已稳定跑通并达到较高 PSNR。
- 但该实验属于“报错恢复后完成”的结果，需额外确认完整日志是否出现：
  - `[FLUX] fallback enabled: using input image as prior for subsequent views.`
- 若出现该行，则代表该次运行中 FLUX prior 存在回退（input-as-prior），科学归类应标记为“fallback run”。
- 若未出现该行且 depth 依赖缓存完整，则可归类为“真实 FLUX depth prior run”。

## 6. 建议的复现实验标签
- 建议保存标签：`hybrid_flux_lego_issue_recovery_2026-02-28`
- 建议在结果表中增加一列：`flux_prior_mode = {true_flux | fallback_input}`

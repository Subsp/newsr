# 2026-03-01 项目总结与解耦测试方案（SDF + GS + SR Prior）

## 1. 项目目标（当前共识）
- 目标不是单纯“生成更好看图像”，而是构建可控的 `LR -> 3D SR` 框架。
- 主流程共识：
  - 以 COLMAP 初始化（当前可先接受较差初始化）；
  - 在 LR 下联合训练 GS + SDF，先拿到可收敛粗几何与粗纹理；
  - 再引入 SR prior 做细节优化，同时尽量避免几何漂移与风格偏航。
- 当前关键问题共识：
  - FLUX prior 可能带来风格化偏移（和真实视图内容脱耦）；
  - SR 阶段新形变需要合理回传到 SDF；
  - 离线环境下 HF 模型依赖不完整会触发 fallback。

## 2. 已完成的代码能力（hybrid_sdfgs）
- 基础混合训练：
  - `HybridScene + SDF adapter + stage scheduler` 已可跑。
- FLUX prior 管线：
  - 支持 `control/redux/kontext`，支持 offline 变量透传；
  - 支持 persistent worker，减少每视角重复启动开销；
  - 失败时可 fallback 为 input-as-prior（保持训练不中断）。
- 损失模块（可独立开关）：
  - `SDF densify proxy`、`FM-SDS-like`、`Frequency block`、`FLUX-grad block`、`Scaffold block`。
- 现新增（本轮）：
  - 训练新增 `--external_prior_root`（外部先验目录）；
  - 可在不启用 FLUX 生成的情况下，直接读取外部 prior 参与 prior-loss；
  - 新增 Canny/视频SR 先验生成工具与解耦脚本。

## 3. 历史问题与修复记录（关键节点）

### 3.1 环境与编译
- `cudatoolkit-dev=11.8` Conda 解析失败：
  - 结论：改为“pip 为主、conda 为辅”的安装路线。
- `numpy 2.x` 与 PyTorch/扩展不兼容：
  - 现象：`numpy is not available`、`A module compiled using NumPy 1.x...`
  - 处理：回退到 `numpy==1.26.4`。
- `torch` CUDA 版本与本机 CUDA mismatch：
  - 现象：`detected CUDA 11.8 mismatches torch 12.1` 导致 rasterization 编译失败；
  - 处理：对齐 PyTorch CUDA 版本后重编译扩展。

### 3.2 数据读取
- Blender 数据集在原始 GS 训练报 `pcd NoneType`：
  - 根因：代码路径默认优先走 COLMAP PCD 初始化；
  - 处理：切换为 Blender transforms 分支读取，后续训练可正常启动。

### 3.3 FLUX offline 依赖
- 离线情况下常见缺失：
  - `Falconsai/nsfw_image_detection`
  - `openai/clip-vit-large-patch14`
  - `google/t5-v1_1-xxl`
  - `LiheYoung/depth-anything-large-hf`（depth 控制分支）
- 现象：
  - 训练前期 prior 生成失败并触发 fallback；
  - 或重复尝试访问 huggingface 导致等待时间过长。
- 结论：
  - 仅设置 `HF_HUB_OFFLINE=1` 不够，必须保证缓存目录完整且结构可被 transformers/hf_hub 识别。

## 4. 已有实验结果（对话中给出的关键数值）

### 4.1 Hybrid Analytic（球面 teacher）验证
- 目的：验证链路可跑，不用于最终质量结论。
- 日志结果（30000 iter）：
  - Test: `L1=0.1012`, `PSNR=14.02`
  - Train: `L1=0.0134`, `PSNR=27.67`
- 解释：可跑通但几何先验过强且不真实，属于“工程连通性验证”。

### 4.2 Hybrid + FLUX（LEGO）一轮完整跑通
- 已记录在：
  - `hybrid_sdfgs/experiment_reports/2026-02-28_hybrid_flux_lego_issue_recovery.md`
- 关键结果（30000 iter）：
  - Test: `L1=0.00558`, `PSNR=34.86`
  - Train: `L1=0.00396`, `PSNR=38.15`
- 注意：
  - 需区分 true-flux prior 与 fallback-input prior 两类运行。

## 5. 当前核心争议与结论
- 争议：prior 设计过于“分层复杂”，难定位贡献来源。
- 结论：改为“解耦测试”优先，先单独验证每条 prior 路线，再谈融合。
- 本轮执行策略：
  - Canny prior 单独生成、单独训练；
  - 视频SR prior 单独生成、单独训练；
  - 统一通过 `--external_prior_root` 接入训练端。

## 6. 本轮新增的解耦测试入口

### 6.1 训练端新增参数
- 文件：`hybrid_sdfgs/train.py`
- 新参数：
  - `--external_prior_root`
  - `--external_prior_subdir`（默认 `priors`）
  - `--external_prior_exts`（默认 `png,jpg,jpeg,webp`）
- 行为：
  - 从外部目录按 `camera.image_name` 匹配 prior 图；
  - 成功匹配即进入 prior-guided loss；
  - 不再依赖 FLUX 子进程在线生成。

### 6.2 新工具脚本
- `hybrid_sdfgs/tools/generate_video_sr_priors.py`
  - 支持 `realbasic/rvrt/vrt` 三类视频SR模型的单序列包装推理；
  - 输出统一写入 `<output_root>/priors/*.png`，可直接被训练端读取。

### 6.3 新实验脚本（hardcoded baseline）
- `hybrid_sdfgs/exp_scripts/42_generate_video_priors_realbasic_lego.sh`
- `hybrid_sdfgs/exp_scripts/43_run_hybrid_external_prior_lego.sh`
- `hybrid_sdfgs/exp_scripts/44_run_hybrid_external_video_prior_lego.sh`

## 7. 下一步建议（按风险从低到高）
1. 先跑 `42 -> 44`，确认视频SR prior 对几何保持和细节提升的净收益。
2. 再跑 `42(model=rvrt)` 与 `42(model=vrt)`，比较不同视频SR先验表现。
3. 依据结果再决定是否需要引入其他前馈几何/频域模块进行拼接。

## 8. 判据建议（每轮都要记录）
- 保真：PSNR/SSIM/LPIPS
- 先验可信度：valid ratio、prior-residual 分布
- 几何稳定性：多视角重投影一致性、mesh 法向噪声
- 风格偏航：与 GT/LR 的颜色直方图偏移、结构关键点偏移

## 9. 新增阶段性结论（2026-03-01）
- 目前可基本确认：`FLUX`（含 `depth/canny control`）在本任务上的“生成性强于超分保真性”。
- 观察到的具体问题：
  - 输出容易偏向语义重绘，和输入真实视角出现形态/材质脱耦；
  - `canny` 控制虽能约束边缘轮廓，但对真实纹理与几何一致性的约束不足；
  - 在 3D 训练中会放大 prior 偏差，导致对真实重建帮助有限。
- 因此当前决策：
  - `FLUX/canny` 不再作为主线 SR prior；
  - 主线改为保真型视频SR prior（RealBasicVSR / RVRT / VRT）+ 外部先验解耦训练；
  - `FLUX` 路线在代码层已移除，仅保留历史记录用于复盘。

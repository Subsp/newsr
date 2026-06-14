# 2026-03-07 讨论记录：Pseudo-SDF vs True SDF（2D直观实验）

## 1. 背景与目标
本轮讨论目标是把“GS 引导得到的 pseudo-SDF”与“可训练的全局 true SDF”差异直观化，先在 2D 上做可解释实验，再映射回 3D SDF-GS 主线。

核心关注点：
- pseudo-SDF 在局部是否可用。
- 为什么在缺锚点/稀疏区域会明显劣化。
- 从单个高斯到局部 SDF 片段的机制是什么。

---

## 2. 已完成可视化资产
目录：
- `/Users/ltl/Desktop/codex_playground/mip-splatting/hybrid_sdfgs/visualizations/pseudo_vs_true_sdf_2d`

脚本：
- `plot_pseudo_vs_true_sdf_2d.py`：圆形基准。
- `plot_pseudo_vs_true_sdf_2d_complex.py`：复杂形状（角点 + 曲面 + 切角 + 凹槽 + 孔洞），可注入局部稀疏化。

输出图：
- `pseudo_vs_true_sdf_2d.png`
- `pseudo_vs_true_sdf_2d_complex.png`
- `pseudo_vs_true_sdf_2d_complex_sparse_region.png`

统一面板：
1. true SDF
2. pseudo-SDF
3. |pseudo - true| 误差
4. pseudo 场梯度范数 |∇f|（理想 SDF 约为 1）

---

## 3. 两类 SDF 的定义边界（本轮统一口径）

### 3.1 True SDF（全局）
- 由隐式函数网络（如 Neuralangelo 风格）学习。
- 通过渲染监督 + Eikonal + 几何正则获得全局一致符号距离场。
- 具备 inside/outside 语义与跨区域可传递的一致几何约束。

### 3.2 Pseudo-SDF（局部）
- 由 GS 样式锚点（位置 + 法向 + 尺度/置信）局部拼接得到。
- 本质是“局部平面 signed distance 的加权融合”，不是严格全局 SDF。
- 锚点覆盖好处可用；缺锚点/法向噪声/高曲率处容易漂移。

---

## 4. 从“单个高斯”到“一小片 pseudo-SDF”的机制

### 4.1 单高斯几何抽象
给定高斯中心 \(\mu_i\)，协方差 \(\Sigma_i\)：
- 特征分解后最小特征值方向可作法向候选 \(n_i\)（扁高斯假设）。
- 取锚点 \(p_i\approx\mu_i\)。

得到局部平面 signed distance：
\[
d_i(x)=\langle x-p_i, n_i\rangle
\]

这就是“一个高斯贡献的一小片局部 SDF 近似”，仅在邻域内可信。

### 4.2 多锚点拼接（当前脚本实现）
对每个查询点 \(x\) 取 KNN 邻居，计算：
\[
\tilde f(x)=\frac{\sum_{i\in\mathcal N_k(x)} w_i(x)\,d_i(x)}{\sum_{i\in\mathcal N_k(x)} w_i(x)}
\]
权重由距离尺度和置信控制（脚本简化为指数衰减 * conf）。

对应实现位置：
- KNN/邻域：`plot_pseudo_vs_true_sdf_2d_complex.py` 第 121-129 行。
- 局部平面距离：第 137-139 行。
- 加权融合：第 140-143 行。
- 法向同向对齐：第 131-135 行。

结论：
- 这是一阶局部近似拼接，不保证全局 signed distance 一致性。

---

## 5. 关键观察与解释

### 5.1 “为什么右上角误差很大”
在早期图中，右上区域误差高主要由三因素叠加：
1. 锚点被人为挖空（missing sector）。
2. 法向有随机噪声。
3. pseudo-SDF 使用局部 KNN 外推，缺约束区被远邻居主导。

### 5.2 “为什么锚点附近看起来还行”
- 锚点密集且法向一致时，局部平面一阶近似有效。
- 所以视觉上会出现“局部可用、全局不稳”的模式。

### 5.3 提升难度后的结果（局部稀疏化）
在复杂形状上新增“局部区域仅保留部分锚点（默认 keep=0.25）”：
- 局部误差图明显抬升。
- |∇f| 偏离 1 的区域扩大。
- 但非稀疏区仍保持相对稳定。

这支持当前判断：
- pseudo-SDF 更像局部 teacher / warm-start。
- 若要全局几何可靠，仍需 true SDF 训练约束闭环。

---

## 6. 与 800->3200 主线的关系
对当前主线（LR 初始化 + 几何约束 + SR 优化）的意义：
- pseudo-SDF 可以提供早期几何方向感，但不能单独承担全局几何真值角色。
- 在锚点不足、视角不一致、细节复杂区域，直接把 pseudo-SDF 当强真值会引入结构偏差。
- 工程上更适合“软约束/暖启动”，并配合后续全局 SDF 或多视角一致性机制。

---

## 7. 下一步待讨论（未执行）
1. 是否引入“局部置信驱动的权重退火”，避免稀疏区硬约束。
2. 是否将 pseudo-SDF 仅用于窄带区域，远场交给 true SDF / 渲染约束。
3. 3D 实验中如何定义“局部可靠区”并可视化其覆盖率。
4. 在 SDF densify 方案里，pseudo-SDF 作为初始化而非终态监督的接口设计。

---

## 8. 快速复现实验命令
```bash
cd /Users/ltl/Desktop/codex_playground/mip-splatting/hybrid_sdfgs/visualizations/pseudo_vs_true_sdf_2d

# 圆形
python plot_pseudo_vs_true_sdf_2d.py --out_dir .

# 复杂形状（无局部稀疏化）
python plot_pseudo_vs_true_sdf_2d_complex.py --out_dir . --region_sparse_keep 1.0 --out_name pseudo_vs_true_sdf_2d_complex.png

# 复杂形状（局部稀疏化，默认较难）
python plot_pseudo_vs_true_sdf_2d_complex.py --out_dir . --region_sparse_keep 0.25 --out_name pseudo_vs_true_sdf_2d_complex_sparse_region.png
```

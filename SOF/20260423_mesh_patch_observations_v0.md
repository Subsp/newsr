# 20260423 Mesh Patch Observations v0 记录

## 最新修改

commit:

```text
d15fab2 Add mesh patch observation preparation
```

新增脚本:

```text
prepare_mesh_patch_observations_v0.py
```

这版不是 prior fusion，也不是训练脚本。它只做一件事：

```text
mesh face / carrier patch -> 遍历真实相机 -> 记录每个相机看到哪些 patch、投影到哪里、深度和视角权重
```

也就是先把“每个相机到 mesh patch 的几何对应关系”建出来，后续再接 prior fusion。

## 输出资产

脚本输出目录由 `--output_dir` 指定。建议 smoke test 目录:

```text
/root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/mesh_patch_observations_smoke_v0
```

主要输出:

```text
mesh_patch_bank_v0.npz
camera_patch_observations/
mesh_patch_render_stub_v0.npz
mesh_patch_bank_preview_v0.ply
mesh_patch_observations_v0_summary.json
```

字段含义:

```text
mesh_patch_bank_v0.npz
  保存 patch 的 3D 几何：center / normal / tangent_u / tangent_v / scale_u / scale_v / face_id / barycentric

camera_patch_observations/*.npz
  每个真实 train camera 一个文件，保存该 view 可见 patch 的 patch_id / pixel_xy / depth / view_cosine / sample_weight / prior_path

mesh_patch_render_stub_v0.npz
  预留渲染接口，几何字段已经填好，但 fused_rgb / confidence / disagreement / valid_mask 目前为空

mesh_patch_bank_preview_v0.ply
  patch center 预览点云，用 view_count 着色

mesh_patch_observations_v0_summary.json
  统计信息和所有路径
```

## 渲染接口约定

`mesh_patch_render_stub_v0.npz` 兼容:

```text
utils.mesh_fusion_render.load_mesh_fusion_payload
```

后续 prior fusion 阶段只需要在同一个几何 payload 上填:

```text
fused_rgb
confidence
disagreement
valid_mask
```

保持这些字段不变:

```text
centers
normals
tangent_u
tangent_v
scale_u
scale_v
face_ids
bary_coords
```

这样 render / train 端不用再重新理解 mesh patch 几何。

## 安全阀

mesh 有约 1200 万 faces，不能误跑全量。

脚本默认:

```text
--huge_patch_threshold 2000000
```

如果估计 patch 数超过该阈值，脚本会直接报错。真的要全量必须显式加:

```text
--allow_huge_patch_bank
```

一般先用:

```text
--face_stride 200
--view_limit 16
```

确认接口和统计没问题。

## 服务器同步指令

```bash
cd /root/autodl-tmp/SOFSR
git pull --ff-only origin codex/prior-fusion-v0
conda activate sof
```

## Smoke Test 指令

```bash
python -u prepare_mesh_patch_observations_v0.py \
  -s /root/autodl-tmp/SOFSR/data_aliases/kitchen_images8bicubic_to_images2 \
  -m /root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft \
  --eval \
  --data_device cpu \
  --mesh_path /root/autodl-tmp/SOFSR/output/kitchen_sof_lr_ablation_v1/early4k_soft/test/ours_30000/mesh_lr_stp0039_7.ply \
  --prior_dir /root/autodl-tmp/priors/StableSRpriors \
  --output_dir /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/mesh_patch_observations_smoke_v0 \
  --face_selection all_faces \
  --face_stride 200 \
  --carriers_per_face 1 \
  --view_limit 16 \
  --visibility_mode patch_zbuffer \
  --save_corners \
  2>&1 | tee /root/autodl-tmp/SOFSR/output/kitchen_prior_fusion_v0_early4k_soft/mesh_patch_observations_smoke_v0.log
```

## 后续要接的 Fusion

下一步不应该直接训练，先做:

```text
camera_patch_observations/*.npz + prior images -> patch-space fusion -> 填 mesh_patch_render_stub_v0.npz
```

fusion 逻辑留空，计划仍然是:

```text
低频一致 -> average
高频一致 -> 注入
高频冲突 -> 降权或 reject
单 view -> 低 confidence 保留
未覆盖 -> 不注入
```

## 当前定位

这条线的目标不是替代 SOFGS，也不是先解决遮挡。它只是建立:

```text
真实相机 prior 与 mesh surface patch 的稳定几何对应
```

如果这个对应关系可视化正确，再进入 patch-space prior fusion；如果 patch 对应错，后续训练一定会把错 prior 固化进 GS。

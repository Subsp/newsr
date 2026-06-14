from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(num_channels: int) -> nn.Module:
    groups = min(32, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            _make_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _make_norm(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return self.act(x + y)


class PyramidEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c4 = base_channels * 4
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, c1, kernel_size=5),
            ResidualBlock(c1),
        )
        self.down2 = nn.Sequential(
            ConvNormAct(c1, c2, stride=2),
            ResidualBlock(c2),
        )
        self.down4 = nn.Sequential(
            ConvNormAct(c2, c4, stride=2),
            ResidualBlock(c4),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_1 = self.stem(x)
        feat_2 = self.down2(feat_1)
        feat_4 = self.down4(feat_2)
        return {
            "scale1": feat_1,
            "scale2": feat_2,
            "scale4": feat_4,
        }


def _canonicalize_camera_tensors(
    cameras: Optional[Dict[str, torch.Tensor]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if cameras is None:
        return None, None, None
    if "intrinsics" not in cameras:
        raise KeyError("cameras must contain 'intrinsics'.")
    intrinsics = cameras["intrinsics"]
    if "cam_to_world" in cameras:
        cam_to_world = cameras["cam_to_world"]
        world_to_view = torch.linalg.inv(cam_to_world)
    elif "world_to_view" in cameras:
        world_to_view = cameras["world_to_view"]
        cam_to_world = torch.linalg.inv(world_to_view)
    else:
        raise KeyError("cameras must contain either 'cam_to_world' or 'world_to_view'.")
    return intrinsics, cam_to_world, world_to_view


def _resize_intrinsics(
    intrinsics: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> torch.Tensor:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out = intrinsics.clone()
    out[..., 0, 0] *= sx
    out[..., 1, 1] *= sy
    out[..., 0, 2] *= sx
    out[..., 1, 2] *= sy
    return out


def _backproject_depth_map(
    depth_hw: torch.Tensor,
    intrinsics: torch.Tensor,
    cam_to_world: torch.Tensor,
) -> torch.Tensor:
    device = depth_hw.device
    dtype = depth_hw.dtype
    h, w = depth_hw.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype) + 0.5,
        torch.arange(w, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)
    rays_cam = pixels @ torch.linalg.inv(intrinsics).transpose(0, 1)
    points_cam = rays_cam * depth_hw.reshape(-1, 1)
    rot = cam_to_world[:3, :3]
    trans = cam_to_world[:3, 3]
    points_world = points_cam @ rot.transpose(0, 1) + trans.unsqueeze(0)
    return points_world.reshape(h, w, 3)


def _project_world_points(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_view: torch.Tensor,
    target_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = target_hw
    points_cam = points_world.reshape(-1, 3) @ world_to_view[:3, :3].transpose(0, 1)
    points_cam = points_cam + world_to_view[:3, 3].unsqueeze(0)
    z = points_cam[:, 2].reshape(h, w)
    z_safe = z.clamp_min(1e-6)
    uv = points_cam @ intrinsics.transpose(0, 1)
    u = (uv[:, 0] / z_safe.reshape(-1)).reshape(h, w)
    v = (uv[:, 1] / z_safe.reshape(-1)).reshape(h, w)
    grid_x = ((u + 0.5) / max(float(w), 1.0)) * 2.0 - 1.0
    grid_y = ((v + 0.5) / max(float(h), 1.0)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid, z


class CrossViewProjectiveFusion(nn.Module):
    def __init__(self, channels: int, depth_tolerance: float) -> None:
        super().__init__()
        self.channels = channels
        self.depth_tolerance = float(depth_tolerance)
        self.reduce = nn.Sequential(
            ConvNormAct(channels * 3 + 2, channels, kernel_size=1),
            ResidualBlock(channels),
        )

    def _fallback_fusion(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=1, keepdim=True).expand_as(features)
        var = (features - mean).pow(2)
        support = torch.ones(
            features.shape[0],
            features.shape[1],
            1,
            features.shape[-2],
            features.shape[-1],
            device=features.device,
            dtype=features.dtype,
        )
        fused = torch.cat([features, mean, var, support, support], dim=2)
        return self.reduce(fused.flatten(0, 1)).unflatten(0, features.shape[:2])

    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
        cameras: Optional[Dict[str, torch.Tensor]],
        full_res_hw: Tuple[int, int],
    ) -> torch.Tensor:
        b, v, c, h, w = features.shape
        if v <= 1 or cameras is None:
            return self._fallback_fusion(features)

        intrinsics, cam_to_world, world_to_view = _canonicalize_camera_tensors(cameras)
        if intrinsics is None or cam_to_world is None or world_to_view is None:
            return self._fallback_fusion(features)

        intrinsics = _resize_intrinsics(intrinsics, full_res_hw, (h, w))
        fused_views = []
        for bi in range(b):
            fused_batch = []
            for ref_idx in range(v):
                ref_feat = features[bi, ref_idx]
                ref_depth = depth[bi, ref_idx, 0]
                world_points = _backproject_depth_map(
                    ref_depth,
                    intrinsics[bi, ref_idx],
                    cam_to_world[bi, ref_idx],
                )
                accum = torch.zeros_like(ref_feat)
                accum_sq = torch.zeros_like(ref_feat)
                weight_sum = torch.zeros((1, h, w), device=ref_feat.device, dtype=ref_feat.dtype)
                support = torch.zeros((1, h, w), device=ref_feat.device, dtype=ref_feat.dtype)

                for src_idx in range(v):
                    if src_idx == ref_idx:
                        continue
                    grid, proj_depth = _project_world_points(
                        world_points,
                        intrinsics[bi, src_idx],
                        world_to_view[bi, src_idx],
                        (h, w),
                    )
                    src_feat = features[bi : bi + 1, src_idx]
                    src_depth = depth[bi : bi + 1, src_idx]
                    sampled_feat = F.grid_sample(
                        src_feat,
                        grid.unsqueeze(0),
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )[0]
                    sampled_depth = F.grid_sample(
                        src_depth,
                        grid.unsqueeze(0),
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )[0, 0]
                    inside = (
                        (grid[..., 0] >= -1.0)
                        & (grid[..., 0] <= 1.0)
                        & (grid[..., 1] >= -1.0)
                        & (grid[..., 1] <= 1.0)
                        & (proj_depth > 1e-6)
                    )
                    tol = self.depth_tolerance * torch.clamp(proj_depth.abs(), min=1.0)
                    depth_gate = torch.exp(-torch.abs(sampled_depth - proj_depth) / tol.clamp_min(1e-6))
                    weight = inside.to(dtype=ref_feat.dtype) * depth_gate
                    accum = accum + sampled_feat * weight.unsqueeze(0)
                    accum_sq = accum_sq + sampled_feat.pow(2) * weight.unsqueeze(0)
                    weight_sum = weight_sum + weight.unsqueeze(0)
                    support = support + (weight > 0.25).to(dtype=ref_feat.dtype).unsqueeze(0)

                denom = weight_sum.clamp_min(1e-6)
                mean = torch.where(weight_sum > 0.0, accum / denom, ref_feat)
                var = torch.relu(accum_sq / denom - mean.pow(2))
                support_ratio = support / float(max(v - 1, 1))
                observed_ratio = weight_sum.clamp(max=float(max(v - 1, 1))) / float(max(v - 1, 1))
                fused = torch.cat(
                    [ref_feat, mean, var, support_ratio, observed_ratio],
                    dim=0,
                )
                fused_batch.append(fused)
            fused_views.append(torch.stack(fused_batch, dim=0))

        fused_tensor = torch.stack(fused_views, dim=0)
        return self.reduce(fused_tensor.flatten(0, 1)).unflatten(0, (b, v))


class HRGSRefiner(nn.Module):
    """SR-guided surface-aware GS refinement head.

    v0 keeps the outputs in image space:
      - refined surface proxy (depth / normal / confidence / ownership maps)
      - update maps that later get lifted or aggregated by helper utilities

    It intentionally does not emit raw Gaussian parameters.
    """

    def __init__(
        self,
        *,
        sr_guide_in_channels: int = 6,
        gs_diag_channels: int = 3,
        vggt_feat_channels: int = 256,
        base_channels: int = 32,
        depth_residual_scale: float = 0.25,
        cross_view_depth_tolerance: float = 0.05,
        action_feature_channels: int = 16,
    ) -> None:
        super().__init__()
        self.gs_diag_channels = int(gs_diag_channels)
        self.vggt_feat_channels = int(vggt_feat_channels)
        self.depth_residual_scale = float(depth_residual_scale)
        self.action_feature_channels = int(action_feature_channels)

        gs_in_channels = 3 + 1 + 3 + 1 + self.gs_diag_channels
        self.sr_encoder = PyramidEncoder(sr_guide_in_channels, base_channels)
        self.gs_encoder = PyramidEncoder(gs_in_channels, base_channels)

        c1 = base_channels
        c2 = base_channels * 2
        c4 = base_channels * 4
        self.cross_view = CrossViewProjectiveFusion(c4, cross_view_depth_tolerance)
        self.vggt_proj = nn.Sequential(
            ConvNormAct(self.vggt_feat_channels, c4, kernel_size=1),
            ResidualBlock(c4),
        )

        self.fuse4 = nn.Sequential(
            ConvNormAct(c4 * 4, c4, kernel_size=1),
            ResidualBlock(c4),
        )
        self.up4_to2 = ConvNormAct(c4, c2)
        self.fuse2 = nn.Sequential(
            ConvNormAct(c2 * 3, c2, kernel_size=1),
            ResidualBlock(c2),
        )
        self.up2_to1 = ConvNormAct(c2, c1)
        self.fuse1 = nn.Sequential(
            ConvNormAct(c1 * 3, c1, kernel_size=1),
            ResidualBlock(c1),
            ResidualBlock(c1),
        )

        self.surface_head = nn.Sequential(
            ResidualBlock(c1),
            ResidualBlock(c1),
        )
        self.delta_depth_head = nn.Conv2d(c1, 1, 3, padding=1)
        self.normal_head = nn.Conv2d(c1, 3, 3, padding=1)
        self.conf_head = nn.Conv2d(c1, 1, 3, padding=1)
        self.mask_surface_head = nn.Conv2d(c1, 1, 3, padding=1)
        self.mask_detail_head = nn.Conv2d(c1, 1, 3, padding=1)
        self.mask_update_head = nn.Conv2d(c1, 1, 3, padding=1)

        self.update_head = nn.Sequential(
            ResidualBlock(c1),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            _make_norm(c1),
            nn.SiLU(inplace=True),
        )
        self.prior_color_head = nn.Conv2d(c1, 1, 3, padding=1)
        self.action_feature_head = nn.Conv2d(c1, self.action_feature_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for head in (
            self.delta_depth_head,
            self.normal_head,
            self.conf_head,
            self.mask_surface_head,
            self.mask_detail_head,
            self.mask_update_head,
            self.prior_color_head,
            self.action_feature_head,
        ):
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    @staticmethod
    def _extract_gs_buffers(gs_buffers: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        def _get(keys):
            for key in keys:
                if key in gs_buffers:
                    return gs_buffers[key]
            raise KeyError(f"Missing any of keys={keys} in gs_buffers.")

        render_rgb = _get(("render_rgb", "rgb", "R_gs"))
        depth = _get(("depth", "D_gs"))
        normal = _get(("normal", "N_gs"))
        alpha = _get(("alpha", "A_gs"))
        diagnostics = gs_buffers.get("diagnostics", gs_buffers.get("Q_gs"))
        if diagnostics is None:
            diagnostics = torch.zeros(
                *depth.shape[:2],
                3,
                depth.shape[-2],
                depth.shape[-1],
                device=depth.device,
                dtype=depth.dtype,
            )
        return render_rgb, depth, normal, alpha, diagnostics

    def _canonicalize_vggt_prior(
        self,
        vggt_prior: Optional[Dict[str, torch.Tensor]],
        *,
        batch_size: int,
        num_views: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        feat_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_h, feat_w = feat_hw
        if vggt_prior is None:
            zero_depth = torch.zeros((batch_size, num_views, 1, height, width), device=device, dtype=dtype)
            zero_feat = torch.zeros((batch_size, num_views, self.vggt_feat_channels, feat_h, feat_w), device=device, dtype=dtype)
            return zero_depth, zero_depth, zero_feat

        depth_hr = vggt_prior.get("depth_hr")
        conf_hr = vggt_prior.get("conf_hr")
        feat_s4 = vggt_prior.get("feat_s4")
        if depth_hr is None or conf_hr is None or feat_s4 is None:
            raise KeyError("vggt_prior must contain 'depth_hr', 'conf_hr', and 'feat_s4'.")

        depth_hr = depth_hr.to(device=device, dtype=dtype)
        conf_hr = conf_hr.to(device=device, dtype=dtype)
        feat_s4 = feat_s4.to(device=device, dtype=dtype)
        if depth_hr.shape != (batch_size, num_views, 1, height, width):
            raise ValueError(
                f"vggt_prior['depth_hr'] shape mismatch: {tuple(depth_hr.shape)} vs "
                f"{(batch_size, num_views, 1, height, width)}"
            )
        if conf_hr.shape != (batch_size, num_views, 1, height, width):
            raise ValueError(
                f"vggt_prior['conf_hr'] shape mismatch: {tuple(conf_hr.shape)} vs "
                f"{(batch_size, num_views, 1, height, width)}"
            )
        if feat_s4.shape[0] != batch_size or feat_s4.shape[1] != num_views:
            raise ValueError(
                f"vggt_prior['feat_s4'] batch/view shape mismatch: {tuple(feat_s4.shape[:2])} vs "
                f"{(batch_size, num_views)}"
            )
        if feat_s4.shape[-2:] != (feat_h, feat_w):
            feat_s4 = F.interpolate(
                feat_s4.flatten(0, 1),
                size=(feat_h, feat_w),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (batch_size, num_views))
        return depth_hr, conf_hr, feat_s4

    def forward(
        self,
        sr_images: torch.Tensor,
        lr_up_images: torch.Tensor,
        gs_buffers: Dict[str, torch.Tensor],
        cameras: Optional[Dict[str, torch.Tensor]] = None,
        vggt_prior: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if sr_images.shape != lr_up_images.shape:
            raise ValueError(
                f"sr_images and lr_up_images must share shape, got "
                f"{tuple(sr_images.shape)} vs {tuple(lr_up_images.shape)}"
            )
        render_rgb, depth_gs, normal_gs, alpha_gs, diagnostics = self._extract_gs_buffers(gs_buffers)

        b, v, _, h, w = sr_images.shape
        sr_input = torch.cat([sr_images, lr_up_images], dim=2)
        gs_input = torch.cat([render_rgb, depth_gs, normal_gs, alpha_gs, diagnostics], dim=2)

        sr_feats = self.sr_encoder(sr_input.flatten(0, 1))
        gs_feats = self.gs_encoder(gs_input.flatten(0, 1))

        sr_scale1 = sr_feats["scale1"].unflatten(0, (b, v))
        sr_scale2 = sr_feats["scale2"].unflatten(0, (b, v))
        sr_scale4 = sr_feats["scale4"].unflatten(0, (b, v))
        gs_scale1 = gs_feats["scale1"].unflatten(0, (b, v))
        gs_scale2 = gs_feats["scale2"].unflatten(0, (b, v))
        gs_scale4 = gs_feats["scale4"].unflatten(0, (b, v))
        depth_vggt, conf_vggt, feat_vggt_s4 = self._canonicalize_vggt_prior(
            vggt_prior,
            batch_size=b,
            num_views=v,
            height=h,
            width=w,
            device=sr_images.device,
            dtype=sr_images.dtype,
            feat_hw=gs_scale4.shape[-2:],
        )
        feat_vggt_s4 = self.vggt_proj(feat_vggt_s4.flatten(0, 1)).unflatten(0, (b, v))

        depth_scale4 = F.interpolate(
            depth_gs.flatten(0, 1),
            size=gs_scale4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).unflatten(0, (b, v))
        cross_scale4 = self.cross_view(
            gs_scale4,
            depth_scale4,
            cameras=cameras,
            full_res_hw=(h, w),
        )

        fuse4 = self.fuse4(torch.cat([sr_scale4, gs_scale4, cross_scale4, feat_vggt_s4], dim=2).flatten(0, 1))
        fuse4 = fuse4.unflatten(0, (b, v))

        up2 = F.interpolate(
            fuse4.flatten(0, 1),
            size=sr_scale2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up2 = self.up4_to2(up2).unflatten(0, (b, v))
        fuse2 = self.fuse2(torch.cat([up2, sr_scale2, gs_scale2], dim=2).flatten(0, 1))
        fuse2 = fuse2.unflatten(0, (b, v))

        up1 = F.interpolate(
            fuse2.flatten(0, 1),
            size=sr_scale1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up1 = self.up2_to1(up1).unflatten(0, (b, v))
        fuse1 = self.fuse1(torch.cat([up1, sr_scale1, gs_scale1], dim=2).flatten(0, 1))

        surface_feat = self.surface_head(fuse1)
        delta_depth = self.delta_depth_head(surface_feat).unflatten(0, (b, v))
        normal_delta = self.normal_head(surface_feat).unflatten(0, (b, v))
        conf_geo = torch.sigmoid(self.conf_head(surface_feat)).unflatten(0, (b, v))
        surface_ownership = torch.sigmoid(self.mask_surface_head(surface_feat)).unflatten(0, (b, v))
        appearance_ownership = torch.sigmoid(self.mask_detail_head(surface_feat)).unflatten(0, (b, v))
        mask_update2d = torch.sigmoid(self.mask_update_head(surface_feat)).unflatten(0, (b, v))

        depth_scale = torch.clamp(torch.maximum(depth_gs.detach().abs(), depth_vggt.detach().abs()), min=1e-3)
        vggt_support = conf_vggt.clamp_min(0.0)
        # For mip-style inputs, alpha alone is not a trustworthy geometry cue.
        # Let the network decide where GS depth is surface-owned, then blend
        # against the VGGT prior accordingly.
        gs_support = (surface_ownership.detach() * alpha_gs.detach()).clamp_min(1e-3)
        blend = vggt_support / (vggt_support + gs_support)
        depth_anchor = torch.where(
            torch.isfinite(depth_vggt) & (depth_vggt > 0.0),
            blend * depth_vggt + (1.0 - blend) * depth_gs,
            depth_gs,
        )
        depth_surf = depth_anchor + torch.tanh(delta_depth) * depth_scale * self.depth_residual_scale
        normal_surf = F.normalize(normal_gs + normal_delta, dim=2, eps=1e-6)

        update_feat = self.update_head(fuse1)
        prior_color_weight2d = torch.sigmoid(self.prior_color_head(update_feat)).unflatten(0, (b, v))
        action_features2d = self.action_feature_head(update_feat).unflatten(0, (b, v))

        return {
            "surface_2d": {
                "delta_depth": delta_depth,
                "depth_anchor": depth_anchor,
                "depth_surf": depth_surf,
                "normal_surf": normal_surf,
                "conf_geo": conf_geo,
                "conf_surface": conf_geo,
                "surface_ownership": surface_ownership,
                "appearance_ownership": appearance_ownership,
                "mask_surface": surface_ownership,
                "mask_detail": appearance_ownership,
                "mask_update2d": mask_update2d,
            },
            "update_2d": {
                "prior_color_weight2d": prior_color_weight2d,
                "action_features2d": action_features2d,
            },
            "debug": {
                "sr_scale4": sr_scale4,
                "gs_scale4": gs_scale4,
                "cross_scale4": cross_scale4,
                "vggt_scale4": feat_vggt_s4,
                "vggt_conf": conf_vggt,
                "depth_blend": blend,
            },
        }

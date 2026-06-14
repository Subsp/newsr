from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SOFPriorConfig:
    lambda_prior_local: float = 0.0
    prior_local_min_pixels: float = 64.0
    prior_local_from_iter: int = 0
    lambda_prior_edge: float = 0.0
    prior_edge_loss_mode: str = "rgb"
    prior_edge_blend_alpha: float = 1.0
    prior_edge_min_pixels: float = 64.0
    prior_edge_from_iter: int = 0
    prior_edge_touch_min_radius_px: float = 2.0
    prior_edge_touch_radius_scale: float = 0.5
    prior_edge_touch_max_radius_px: float = 16.0
    prior_edge_detail_blur_kernel: int = 9
    prior_edge_detail_alpha: float = 0.6
    prior_edge_detail_alpha_final: float = -1.0
    prior_edge_detail_warmup_iters: int = 0
    prior_edge_detail_weight: float = 1.0
    prior_edge_lowfreq_weight: float = 0.05
    prior_edge_grad_weight: float = 0.0
    prior_edge_lowfreq_threshold: float = 0.08
    prior_edge_lowfreq_anchor: str = "render"
    prior_edge_detail_min_gain: float = 0.0
    prior_edge_confidence_power: float = 1.0


def _normalize_mask_hw(mask: Optional[torch.Tensor], *, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim != 2:
        raise ValueError(f"mask must be [H,W] or [1,H,W], got shape {tuple(mask.shape)}")
    return mask.to(device=device, dtype=dtype)


def _odd_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 1
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def blur_image_chw(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = _odd_kernel_size(kernel_size)
    if kernel_size <= 1:
        return image
    pad = kernel_size // 2
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)[0]


def compute_masked_l1_loss(
    image: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    mask_hw = _normalize_mask_hw(mask, dtype=image.dtype, device=image.device)
    if mask_hw is None:
        return None
    active = float(mask_hw.sum().item())
    if active <= 0:
        return None
    mask3 = mask_hw.unsqueeze(0)
    return (torch.abs(image - target) * mask3).sum() / (mask3.sum() * image.shape[0]).clamp_min(1.0)


def compute_weighted_l1_loss(
    image: torch.Tensor,
    target: torch.Tensor,
    weight_hw: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    weight_hw = _normalize_mask_hw(weight_hw, dtype=image.dtype, device=image.device)
    if weight_hw is None:
        return None
    active = weight_hw.sum()
    if float(active.item()) <= 0:
        return None
    return (torch.abs(image - target) * weight_hw.unsqueeze(0)).sum() / (active * image.shape[0]).clamp_min(1.0)


def compute_weighted_gradient_l1_loss(
    image: torch.Tensor,
    target: torch.Tensor,
    weight_hw: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    weight_hw = _normalize_mask_hw(weight_hw, dtype=image.dtype, device=image.device)
    if weight_hw is None:
        return None
    dx_image = image[:, :, 1:] - image[:, :, :-1]
    dx_target = target[:, :, 1:] - target[:, :, :-1]
    dy_image = image[:, 1:, :] - image[:, :-1, :]
    dy_target = target[:, 1:, :] - target[:, :-1, :]
    weight_x = 0.5 * (weight_hw[:, 1:] + weight_hw[:, :-1])
    weight_y = 0.5 * (weight_hw[1:, :] + weight_hw[:-1, :])
    loss_x = compute_weighted_l1_loss(dx_image, dx_target, weight_x)
    loss_y = compute_weighted_l1_loss(dy_image, dy_target, weight_y)
    if loss_x is None and loss_y is None:
        return None
    if loss_x is None:
        return loss_y
    if loss_y is None:
        return loss_x
    return 0.5 * (loss_x + loss_y)


def _dilate_binary_mask(mask_hw: torch.Tensor, radius_px: int) -> torch.Tensor:
    if radius_px <= 0:
        return mask_hw.to(dtype=torch.bool)
    mask = mask_hw.to(dtype=torch.float32)[None, None]
    kernel = int(radius_px) * 2 + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=int(radius_px))
    return dilated[0, 0] > 0.5


class SOFPriorBlock:
    def __init__(self, cfg: SOFPriorConfig):
        self.cfg = cfg

    def get_prior_edge_detail_alpha(self, iteration: int, train_start_iter: int) -> float:
        start = float(self.cfg.prior_edge_detail_alpha)
        final = float(self.cfg.prior_edge_detail_alpha_final)
        if final < 0.0:
            return start
        warmup = max(int(self.cfg.prior_edge_detail_warmup_iters), 0)
        if warmup <= 0:
            return final
        t = max(0.0, min(1.0, float(iteration - train_start_iter) / float(warmup)))
        return start + (final - start) * t

    def compute_local_loss(
        self,
        render_image: torch.Tensor,
        prior_image: Optional[torch.Tensor],
        prior_mask: Optional[torch.Tensor],
        *,
        iteration: int,
    ) -> Optional[torch.Tensor]:
        if float(self.cfg.lambda_prior_local) <= 0.0 or prior_image is None:
            return None
        if int(iteration) < int(self.cfg.prior_local_from_iter):
            return None
        mask_hw = _normalize_mask_hw(prior_mask, dtype=render_image.dtype, device=render_image.device)
        if mask_hw is None:
            return None
        if float(mask_hw.sum().item()) < float(self.cfg.prior_local_min_pixels):
            return None
        return compute_masked_l1_loss(render_image, prior_image, mask_hw)

    def compute_edge_loss(
        self,
        render_image: torch.Tensor,
        prior_image: Optional[torch.Tensor],
        prior_mask: Optional[torch.Tensor],
        *,
        iteration: int,
        train_start_iter: int,
        lowfreq_anchor: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[float]]:
        if float(self.cfg.lambda_prior_edge) <= 0.0 or prior_image is None:
            return None, None
        if int(iteration) < int(self.cfg.prior_edge_from_iter):
            return None, None
        mask_hw = _normalize_mask_hw(prior_mask, dtype=render_image.dtype, device=render_image.device)
        if mask_hw is None:
            return None, None
        if float(mask_hw.sum().item()) < float(self.cfg.prior_edge_min_pixels):
            return None, None

        if str(self.cfg.prior_edge_loss_mode).strip().lower() == "detail_v1":
            detail_alpha = self.get_prior_edge_detail_alpha(iteration, train_start_iter)
            return (
                self.compute_prior_edge_detail_loss(
                    image=render_image,
                    prior_target=prior_image,
                    image_mask=mask_hw,
                    detail_alpha=detail_alpha,
                    lowfreq_anchor=lowfreq_anchor,
                ),
                detail_alpha,
            )

        blend_alpha = float(self.cfg.prior_edge_blend_alpha)
        prior_target = prior_image
        if blend_alpha < 1.0:
            blend_alpha = max(0.0, blend_alpha)
            prior_target = blend_alpha * prior_target + (1.0 - blend_alpha) * render_image.detach()
        return compute_masked_l1_loss(render_image, prior_target, mask_hw), None

    def compute_prior_edge_detail_loss(
        self,
        image: torch.Tensor,
        prior_target: torch.Tensor,
        image_mask: Optional[torch.Tensor],
        *,
        detail_alpha: float,
        lowfreq_anchor: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        image_mask = _normalize_mask_hw(image_mask, dtype=image.dtype, device=image.device)
        if image_mask is None:
            return None
        if float(image_mask.sum().item()) <= 0:
            return None

        blur_kernel = int(self.cfg.prior_edge_detail_blur_kernel)
        image_low = blur_image_chw(image, blur_kernel)
        prior_low = blur_image_chw(prior_target, blur_kernel)
        anchor = image.detach() if lowfreq_anchor is None else lowfreq_anchor.detach()
        anchor_low = blur_image_chw(anchor, blur_kernel)

        image_high = image - image_low
        prior_high = prior_target - prior_low
        anchor_high = image.detach() - blur_image_chw(image.detach(), blur_kernel)

        lowfreq_diff = torch.abs(prior_low - anchor_low).mean(dim=0)
        lowfreq_threshold = float(self.cfg.prior_edge_lowfreq_threshold)
        if lowfreq_threshold > 0.0:
            confidence = torch.clamp(1.0 - lowfreq_diff / lowfreq_threshold, min=0.0, max=1.0)
        else:
            confidence = torch.ones_like(image_mask)

        detail_min_gain = float(self.cfg.prior_edge_detail_min_gain)
        if detail_min_gain > 0.0:
            prior_detail = torch.abs(prior_high).mean(dim=0)
            current_detail = torch.abs(anchor_high).mean(dim=0)
            detail_confidence = torch.clamp(
                (prior_detail - current_detail - detail_min_gain) / detail_min_gain,
                min=0.0,
                max=1.0,
            )
            confidence = confidence * detail_confidence

        confidence_power = float(self.cfg.prior_edge_confidence_power)
        if confidence_power != 1.0:
            confidence = torch.clamp(confidence, min=0.0, max=1.0).pow(confidence_power)
        confidence = confidence * image_mask

        if float(confidence.sum().item()) < float(self.cfg.prior_edge_min_pixels):
            return None

        detail_alpha = max(0.0, min(1.0, float(detail_alpha)))
        high_target = (1.0 - detail_alpha) * anchor_high + detail_alpha * prior_high
        detail_loss = compute_weighted_l1_loss(image_high, high_target.detach(), confidence)
        if detail_loss is None:
            return None

        total_loss = float(self.cfg.prior_edge_detail_weight) * detail_loss

        lowfreq_weight = float(self.cfg.prior_edge_lowfreq_weight)
        if lowfreq_weight > 0.0:
            low_loss = compute_weighted_l1_loss(image_low, anchor_low.detach(), image_mask)
            if low_loss is not None:
                total_loss = total_loss + lowfreq_weight * low_loss

        grad_weight = float(self.cfg.prior_edge_grad_weight)
        if grad_weight > 0.0:
            grad_target = anchor + detail_alpha * prior_high
            grad_loss = compute_weighted_gradient_l1_loss(image, grad_target.detach(), confidence)
            if grad_loss is not None:
                total_loss = total_loss + grad_weight * grad_loss

        return total_loss

    def build_touch_mask(
        self,
        viewpoint_cam,
        gaussians,
        image_mask: Optional[torch.Tensor],
        *,
        visibility_filter: Optional[torch.Tensor] = None,
        radii: Optional[torch.Tensor] = None,
        depth_min: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        image_mask_hw = _normalize_mask_hw(
            image_mask,
            dtype=torch.float32,
            device=gaussians.get_xyz.device,
        )
        if image_mask_hw is None:
            return None
        image_mask_bool = image_mask_hw > 0.5
        xyz = gaussians.get_xyz.detach()
        if xyz.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=xyz.device)

        R = torch.as_tensor(viewpoint_cam.R, device=xyz.device, dtype=xyz.dtype)
        T = torch.as_tensor(viewpoint_cam.T, device=xyz.device, dtype=xyz.dtype)
        xyz_cam = xyz @ R + T.unsqueeze(0)
        z = xyz_cam[:, 2]
        x = xyz_cam[:, 0] / torch.clamp_min(z, 1e-6) * float(viewpoint_cam.focal_x) + float(viewpoint_cam.image_width) / 2.0
        y = xyz_cam[:, 1] / torch.clamp_min(z, 1e-6) * float(viewpoint_cam.focal_y) + float(viewpoint_cam.image_height) / 2.0

        valid = z > float(depth_min)
        if visibility_filter is not None:
            valid = valid & visibility_filter.to(device=xyz.device, dtype=torch.bool)

        xi = torch.round(x).to(dtype=torch.int64)
        yi = torch.round(y).to(dtype=torch.int64)
        valid = valid & (xi >= 0) & (xi < int(viewpoint_cam.image_width)) & (yi >= 0) & (yi < int(viewpoint_cam.image_height))

        touched = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
        if not torch.any(valid):
            return touched
        valid_idx = valid.nonzero(as_tuple=True)[0]
        mask_values = image_mask_bool[yi[valid_idx], xi[valid_idx]]
        touched[valid_idx] = mask_values

        if radii is None:
            touch_radius = int(round(float(self.cfg.prior_edge_touch_min_radius_px)))
            if touch_radius > 0:
                dilated_mask = _dilate_binary_mask(image_mask_bool, touch_radius)
                touched[valid_idx] |= dilated_mask[yi[valid_idx], xi[valid_idx]]
            return touched

        radii = radii.detach().to(device=xyz.device, dtype=torch.float32)
        valid_radii = torch.ceil(
            torch.clamp(
                radii[valid_idx] * float(self.cfg.prior_edge_touch_radius_scale),
                min=float(self.cfg.prior_edge_touch_min_radius_px),
                max=float(self.cfg.prior_edge_touch_max_radius_px),
            )
        ).to(dtype=torch.int64)
        max_radius = int(valid_radii.max().item()) if valid_radii.numel() > 0 else 0
        for radius in range(1, max_radius + 1):
            bucket = valid_radii == radius
            if not torch.any(bucket):
                continue
            dilated_mask = _dilate_binary_mask(image_mask_bool, radius)
            bucket_idx = valid_idx[bucket]
            touched[bucket_idx] |= dilated_mask[yi[bucket_idx], xi[bucket_idx]]
        return touched

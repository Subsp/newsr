from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation


@dataclass
class SDFDensifyConfig:
    interval: int = 500
    topk: int = 256
    min_score: float = 0.05
    surface_coef: float = 1.0
    normal_coef: float = 0.5
    offsurface_coef: float = 0.5
    sr_levels: int = 2
    sr_samples_per_point: int = 2
    sr_jitter_scale: float = 0.6
    sr_max_points: int = 4096
    sr_score_coef: float = 0.0


class SDFDensifyBlock:
    """Collects high-error SDF regions and emits SR-style local resamples.

    The block remains decoupled from native GS densification, but now provides
    an extra set of local points that can be used for high-frequency SDF
    regularization (800->3200 style SR regime).
    """

    def __init__(self, cfg: SDFDensifyConfig):
        self.cfg = cfg
        self.last_selected_xyz = None

    @staticmethod
    def _gaussian_normals(rotations_raw: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        rotations = build_rotation(rotations_raw)
        idx = torch.argmin(scales, dim=1)
        batch_idx = torch.arange(rotations.shape[0], device=rotations.device)
        normals = rotations[batch_idx, :, idx]
        return F.normalize(normals, dim=-1)

    def compute(
        self,
        xyz_sample: torch.Tensor,
        linearized_sdf: torch.Tensor,
        rotations_raw: torch.Tensor,
        scales: torch.Tensor,
        sdf_grads: torch.Tensor,
        opacities: torch.Tensor,
        margin: float,
        iteration: int,
        sr_point_weight: torch.Tensor = None,
    ):
        device = linearized_sdf.device
        empty_bundle = {
            "xyz": None,
            "rotations_raw": None,
            "scales": None,
            "opacities": None,
        }
        if linearized_sdf.numel() == 0:
            zero = torch.zeros((), device=device)
            return zero, {
                "score_mean": 0.0,
                "score_max": 0.0,
                "selected_points": 0.0,
                "refresh": 0.0,
                "sr_points": 0.0,
                "sr_score_mean": 0.0,
                "sr_score_max": 0.0,
            }, empty_bundle

        sdf_abs = linearized_sdf.view(-1).abs()
        gauss_normals = self._gaussian_normals(rotations_raw, scales)
        sdf_normals = F.normalize(sdf_grads, dim=-1)
        normal_residual = 1.0 - (gauss_normals * sdf_normals).sum(dim=-1).abs()

        offsurface_mask = (sdf_abs > margin).float()
        offsurface_residual = opacities.view(-1) * offsurface_mask

        score = (
            self.cfg.surface_coef * sdf_abs
            + self.cfg.normal_coef * normal_residual
            + self.cfg.offsurface_coef * offsurface_residual
        )
        sr_score_mean = 0.0
        sr_score_max = 0.0
        if sr_point_weight is not None and float(self.cfg.sr_score_coef) != 0.0:
            sr_w = sr_point_weight.view(-1).to(device=device, dtype=score.dtype)
            if sr_w.numel() != score.numel():
                raise ValueError(
                    f"sr_point_weight size mismatch: got {sr_w.numel()} vs score {score.numel()}"
                )
            sr_w = sr_w.clamp(min=0.0)
            mean_w = sr_w.mean().detach()
            if torch.isfinite(mean_w) and mean_w > 0:
                sr_w = sr_w / (mean_w + 1e-6)
            score = score + float(self.cfg.sr_score_coef) * sr_w
            sr_score_mean = float(sr_w.mean().detach().item())
            sr_score_max = float(sr_w.max().detach().item())

        proxy_loss = score.mean()

        count = min(max(int(self.cfg.topk), 0), int(score.shape[0]))
        selected_points = 0
        sr_points = 0
        refresh = False
        selected_idx = None
        if count > 0:
            top_vals, top_idx = torch.topk(score, k=count, largest=True)
            valid = top_vals >= float(self.cfg.min_score)
            selected_idx = top_idx[valid]
            selected_points = int(selected_idx.shape[0])
            refresh = int(self.cfg.interval) > 0 and iteration % int(self.cfg.interval) == 0
            if refresh and selected_points > 0:
                self.last_selected_xyz = xyz_sample[selected_idx].detach()

        extra_bundle = empty_bundle
        if selected_idx is not None and selected_points > 0:
            levels = max(1, int(self.cfg.sr_levels))
            spp = max(1, int(self.cfg.sr_samples_per_point))
            base_xyz = xyz_sample[selected_idx]
            base_rot = rotations_raw[selected_idx]
            base_scale = scales[selected_idx]
            base_opacity = opacities[selected_idx]

            base_sigma = base_scale.min(dim=1, keepdim=True).values.clamp(min=1e-4)
            xyz_levels = []
            rot_levels = []
            scale_levels = []
            opacity_levels = []
            for lv in range(levels):
                jitter = float(self.cfg.sr_jitter_scale) / (2 ** lv)
                sigma_lv = base_sigma * jitter

                dirs = torch.randn(
                    (base_xyz.shape[0], spp, 3),
                    device=base_xyz.device,
                    dtype=base_xyz.dtype,
                )
                dirs = F.normalize(dirs, dim=-1)
                perturbed = base_xyz[:, None, :] + dirs * sigma_lv[:, None, :]

                xyz_levels.append(perturbed.reshape(-1, 3))
                rot_levels.append(base_rot[:, None, :].repeat(1, spp, 1).reshape(-1, 4))
                scale_levels.append(base_scale[:, None, :].repeat(1, spp, 1).reshape(-1, 3))
                opacity_levels.append(base_opacity[:, None, :].repeat(1, spp, 1).reshape(-1, 1))

            extra_xyz = torch.cat(xyz_levels, dim=0)
            extra_rot = torch.cat(rot_levels, dim=0)
            extra_scale = torch.cat(scale_levels, dim=0)
            extra_opacity = torch.cat(opacity_levels, dim=0)

            max_points = max(0, int(self.cfg.sr_max_points))
            if max_points > 0 and extra_xyz.shape[0] > max_points:
                keep = torch.randperm(extra_xyz.shape[0], device=extra_xyz.device)[:max_points]
                extra_xyz = extra_xyz[keep]
                extra_rot = extra_rot[keep]
                extra_scale = extra_scale[keep]
                extra_opacity = extra_opacity[keep]

            sr_points = int(extra_xyz.shape[0])
            extra_bundle = {
                "xyz": extra_xyz,
                "rotations_raw": extra_rot,
                "scales": extra_scale,
                "opacities": extra_opacity,
            }

        metrics = {
            "score_mean": float(score.mean().detach().item()),
            "score_max": float(score.max().detach().item()),
            "selected_points": float(selected_points),
            "refresh": 1.0 if refresh else 0.0,
            "sr_points": float(sr_points),
            "sr_score_mean": sr_score_mean,
            "sr_score_max": sr_score_max,
        }
        return proxy_loss, metrics, extra_bundle

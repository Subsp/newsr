from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FrequencyLossConfig:
    low_cutoff: float = 0.10
    high_cutoff: float = 0.35
    low_weight: float = 0.5
    mid_weight: float = 0.7
    high_weight: float = 1.0
    method: str = "fft"


class FrequencyDecompositionBlock:
    """Band-split frequency loss for GS/SDF future coupling."""

    def __init__(self, cfg: FrequencyLossConfig):
        self.cfg = cfg

    @staticmethod
    def _radial_freq_map(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        fy = torch.fft.fftfreq(height, d=1.0, device=device).view(height, 1)
        fx = torch.fft.fftfreq(width, d=1.0, device=device).view(1, width)
        radius = torch.sqrt(fx * fx + fy * fy)
        return radius / torch.clamp(radius.max(), min=1e-6)

    @staticmethod
    def _band_mean(power: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=power.dtype, device=power.device)[None, :, :]
        denom = torch.clamp(mask.sum() * power.shape[0], min=1.0)
        return (power * mask).sum() / denom

    @staticmethod
    def _haar_components(image: torch.Tensor):
        c = image.shape[0]
        dtype = image.dtype
        device = image.device
        base = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],   # LL
                [[1.0, -1.0], [1.0, -1.0]], # LH
                [[1.0, 1.0], [-1.0, -1.0]], # HL
                [[1.0, -1.0], [-1.0, 1.0]], # HH
            ],
            dtype=dtype,
            device=device,
        ) / 2.0
        kernel = base[:, None, :, :].repeat(c, 1, 1, 1)
        x = image.unsqueeze(0)
        y = F.conv2d(x, kernel, stride=2, groups=c)
        y = y.view(1, c, 4, y.shape[-2], y.shape[-1])[0]
        return y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    @staticmethod
    def _masked_l1(diff: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return diff.abs().mean()
        if mask.ndim == 4:
            mask = mask.squeeze(0)
        weighted = diff.abs() * mask
        denom = torch.clamp(mask.sum() * diff.shape[0], min=1.0)
        return weighted.sum() / denom

    def _compute_fft(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        source_f = torch.fft.fft2(source, dim=(-2, -1))
        target_f = torch.fft.fft2(target, dim=(-2, -1))
        diff_f = source_f - target_f
        power = diff_f.real.pow(2) + diff_f.imag.pow(2)

        h, w = int(source.shape[-2]), int(source.shape[-1])
        radius = self._radial_freq_map(h, w, source.device, source.dtype)

        low_cutoff = float(min(self.cfg.low_cutoff, self.cfg.high_cutoff))
        high_cutoff = float(max(self.cfg.low_cutoff, self.cfg.high_cutoff))
        low_mask = radius <= low_cutoff
        mid_mask = (radius > low_cutoff) & (radius <= high_cutoff)
        high_mask = radius > high_cutoff

        low_loss = self._band_mean(power, low_mask)
        mid_loss = self._band_mean(power, mid_mask)
        high_loss = self._band_mean(power, high_mask)
        total = (
            float(self.cfg.low_weight) * low_loss
            + float(self.cfg.mid_weight) * mid_loss
            + float(self.cfg.high_weight) * high_loss
        )
        metrics = {
            "low": float(low_loss.detach().item()),
            "mid": float(mid_loss.detach().item()),
            "high": float(high_loss.detach().item()),
        }
        return total, metrics

    def _compute_haar(
        self, source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict]:
        src_ll, src_lh, src_hl, src_hh = self._haar_components(source)
        tgt_ll, tgt_lh, tgt_hl, tgt_hh = self._haar_components(target)

        ds_mask = None
        if mask is not None:
            ds_mask = F.avg_pool2d(mask.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)

        low_loss = self._masked_l1(src_ll - tgt_ll, ds_mask)
        mid_loss = 0.5 * (
            self._masked_l1(src_lh - tgt_lh, ds_mask)
            + self._masked_l1(src_hl - tgt_hl, ds_mask)
        )
        high_loss = self._masked_l1(src_hh - tgt_hh, ds_mask)
        total = (
            float(self.cfg.low_weight) * low_loss
            + float(self.cfg.mid_weight) * mid_loss
            + float(self.cfg.high_weight) * high_loss
        )
        metrics = {
            "low": float(low_loss.detach().item()),
            "mid": float(mid_loss.detach().item()),
            "high": float(high_loss.detach().item()),
        }
        return total, metrics

    def compute(self, render_image: torch.Tensor, target_image: torch.Tensor, mask: torch.Tensor | None = None):
        if target_image is None:
            zero = torch.zeros((), device=render_image.device)
            return zero, {"low": 0.0, "mid": 0.0, "high": 0.0}

        source = render_image
        target = target_image.detach()
        if mask is not None:
            source = source * mask
            target = target * mask

        method = str(self.cfg.method).strip().lower()
        if method == "fft":
            return self._compute_fft(source, target)
        if method == "haar":
            return self._compute_haar(source, target, mask)
        raise ValueError(f"Unsupported frequency method: {self.cfg.method}")

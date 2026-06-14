from dataclasses import dataclass

import torch


@dataclass
class FMGuidanceConfig:
    t_min: float = 0.02
    t_max: float = 0.98
    gamma: float = 1.0
    huber_delta: float = 0.03


class FlowMatchingSDSLikeBlock:
    """Lightweight SDS-like bridge for future flow-matching guidance.

    Current implementation uses prior/render residual as a surrogate velocity gap,
    with a timestep-dependent weighting schedule.
    """

    def __init__(self, cfg: FMGuidanceConfig):
        self.cfg = cfg

    def _sample_t(self, device: torch.device) -> torch.Tensor:
        t_min = float(min(self.cfg.t_min, self.cfg.t_max))
        t_max = float(max(self.cfg.t_min, self.cfg.t_max))
        u = torch.rand((), device=device)
        return t_min + (t_max - t_min) * u

    def _huber(self, residual: torch.Tensor, delta: float) -> torch.Tensor:
        if delta <= 0:
            return residual.abs()
        abs_res = residual.abs()
        quad = torch.clamp(abs_res, max=delta)
        lin = abs_res - quad
        return 0.5 * quad * quad / delta + lin

    def compute(self, render_image: torch.Tensor, prior_image: torch.Tensor, mask: torch.Tensor | None = None):
        if prior_image is None:
            zero = torch.zeros((), device=render_image.device)
            return zero, {"t": 0.0, "weight_t": 0.0, "residual": 0.0}

        target = prior_image.detach()
        residual = render_image - target

        if mask is not None:
            residual = residual * mask
            denom = torch.clamp(mask.sum() * residual.shape[0], min=1.0)
        else:
            denom = torch.tensor(float(residual.numel()), device=residual.device)

        t = self._sample_t(residual.device)
        weight_t = torch.clamp(1.0 - t, min=1e-4) ** float(self.cfg.gamma)

        per_elem = self._huber(residual, float(self.cfg.huber_delta))
        base_loss = per_elem.sum() / denom
        loss = weight_t * base_loss

        metrics = {
            "t": float(t.detach().item()),
            "weight_t": float(weight_t.detach().item()),
            "residual": float(residual.abs().mean().detach().item()),
        }
        return loss, metrics

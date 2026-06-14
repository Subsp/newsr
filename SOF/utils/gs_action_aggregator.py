from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch


@dataclass
class GSActionAggregatorConfig:
    contribution_eps: float = 1e-8
    update_threshold: float = 0.5
    support_threshold: float = 0.25


def save_gs_action_payload(path: str | Path, payload: Dict[str, torch.Tensor]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            serializable[key] = value.detach().cpu()
        else:
            serializable[key] = value
    torch.save(serializable, path)
    return path


def _resolve_visibility_records(visibility_records: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    gaussian_ids = visibility_records.get("gaussian_ids", visibility_records.get("touched_gs_ids"))
    weights = visibility_records.get("weights", visibility_records.get("contribution_weights"))
    if gaussian_ids is None or weights is None:
        raise KeyError(
            "visibility_records must contain gaussian ids and weights "
            "under ('gaussian_ids', 'weights') or ('touched_gs_ids', 'contribution_weights')."
        )
    return gaussian_ids, weights


def aggregate_gs_actions(
    masks_2d: Dict[str, torch.Tensor],
    action_features_2d: Optional[torch.Tensor],
    visibility_records: Dict[str, torch.Tensor],
    num_gaussians: int,
    cfg: GSActionAggregatorConfig | None = None,
) -> Dict[str, torch.Tensor]:
    cfg = cfg or GSActionAggregatorConfig()
    gaussian_ids, weights = _resolve_visibility_records(visibility_records)
    device = weights.device
    dtype = weights.dtype

    mask_update = masks_2d["mask_update2d"]
    mask_surface = masks_2d["mask_surface"]
    mask_detail = masks_2d["mask_detail"]
    prior_color = masks_2d.get("prior_color_weight2d")
    if prior_color is None:
        prior_color = mask_update

    if mask_update.shape[:2] != gaussian_ids.shape[:2]:
        raise ValueError(
            f"Mask view shape {tuple(mask_update.shape[:2])} does not match "
            f"visibility shape {tuple(gaussian_ids.shape[:2])}"
        )

    if gaussian_ids.shape[-1] != weights.shape[-1]:
        raise ValueError("gaussian_ids and weights must share the last dimension K.")

    valid = gaussian_ids >= 0
    flat_ids = gaussian_ids.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_ids = flat_ids[flat_valid].to(dtype=torch.long)
    flat_weights = weights.reshape(-1)[flat_valid].to(dtype=dtype)

    expand_k = gaussian_ids.shape[-1]
    update_signal = mask_update.unsqueeze(-1).expand(*mask_update.shape, expand_k).reshape(-1)[flat_valid]
    surface_signal = mask_surface.unsqueeze(-1).expand(*mask_surface.shape, expand_k).reshape(-1)[flat_valid]
    detail_signal = mask_detail.unsqueeze(-1).expand(*mask_detail.shape, expand_k).reshape(-1)[flat_valid]
    prior_signal = prior_color.unsqueeze(-1).expand(*prior_color.shape, expand_k).reshape(-1)[flat_valid]

    denom = torch.zeros((num_gaussians,), device=device, dtype=dtype)
    update_sum = torch.zeros_like(denom)
    surface_sum = torch.zeros_like(denom)
    detail_sum = torch.zeros_like(denom)
    prior_sum = torch.zeros_like(denom)
    support_sum = torch.zeros_like(denom)

    denom.scatter_add_(0, flat_ids, flat_weights)
    update_sum.scatter_add_(0, flat_ids, flat_weights * update_signal)
    surface_sum.scatter_add_(0, flat_ids, flat_weights * surface_signal)
    detail_sum.scatter_add_(0, flat_ids, flat_weights * detail_signal)
    prior_sum.scatter_add_(0, flat_ids, flat_weights * prior_signal)
    support_sum.scatter_add_(0, flat_ids, (flat_weights > float(cfg.support_threshold)).to(dtype))

    denom_safe = denom.clamp_min(float(cfg.contribution_eps))
    update_strength = (update_sum / denom_safe).unsqueeze(1)
    attach_strength = (surface_sum / denom_safe).unsqueeze(1)
    detail_weight = (detail_sum / denom_safe).unsqueeze(1)
    prior_color_strength = (prior_sum / denom_safe).unsqueeze(1)

    densify_score = torch.relu(update_strength - detail_weight)
    prune_score = torch.relu((1.0 - attach_strength) * (1.0 - update_strength))

    payload = {
        "update_strength": update_strength,
        "attach_strength": attach_strength,
        "detail_weight": detail_weight,
        "prior_color_strength": prior_color_strength,
        "densify_score": densify_score,
        "prune_score": prune_score,
        "contribution_sum": denom.unsqueeze(1),
        "support_count": support_sum.unsqueeze(1),
    }

    if action_features_2d is not None:
        c = action_features_2d.shape[2]
        expanded_feat = action_features_2d.unsqueeze(-1).expand(
            *action_features_2d.shape, expand_k
        )
        flat_feat = expanded_feat.permute(0, 1, 3, 4, 5, 2).reshape(-1, c)[flat_valid]
        feat_sum = torch.zeros((num_gaussians, c), device=device, dtype=dtype)
        feat_sum.scatter_add_(
            0,
            flat_ids[:, None].expand(-1, c),
            flat_feat * flat_weights[:, None],
        )
        payload["action_embedding"] = feat_sum / denom_safe[:, None]

    return payload

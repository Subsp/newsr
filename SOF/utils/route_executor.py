from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


@dataclass
class RouteExecutionConfig:
    mode: str = "detail_preserve_v0p8"
    detail_protect_beta: float = 0.7
    detail_attach_strength: float = 0.0
    detail_geometry_strength: float = 0.0
    detail_prior_color_strength_scale: float = 0.25
    suppress_update_floor: float = 0.15
    suppress_opacity_scale: float = 0.25
    min_scale_gate: float = 0.3
    scale_gate_power: float = 1.0
    scale_gate_suppress_coupling: float = 1.0
    risky_useful_opacity_coupling: float = 0.35
    risky_useful_scale_coupling: float = 0.2
    harmful_scale_coupling: float = 1.0


def _nested_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nested_to_cpu(item) for item in value]
    return value


def save_route_payload(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_nested_to_cpu(payload), path)
    return path


def vector_stats(values: torch.Tensor) -> Dict[str, float | None]:
    flat = values.detach().reshape(-1).float().cpu()
    if flat.numel() == 0:
        return {"min": None, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "min": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "median": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "max": float(flat.max().item()),
    }


def _unwrap_route_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    direct_keys = {
        "update_strength",
        "attach_strength",
        "detail_weight",
        "prior_color_strength",
        "execution_update_strength",
        "execution_attach_weight",
        "execution_detail_weight",
        "execution_prior_color_strength",
        "execution_suppress_strength",
        "execution_position_weight",
        "execution_scaling_weight",
        "execution_rotation_weight",
        "execution_appearance_weight",
        "execution_opacity_weight",
        "p_attach",
        "p_detail",
        "p_suppress",
        "geometry_update_strength",
        "color_update_strength",
    }
    if any(key in payload for key in direct_keys):
        return payload

    for key in ("route_payload", "gaussian_route_payload", "gs_action_payload"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested

    nested = payload.get("hrgs_outputs")
    if isinstance(nested, dict):
        for key in ("route_payload", "gaussian_route_payload", "gs_action_payload"):
            child = nested.get(key)
            if isinstance(child, dict):
                return child

    raise KeyError(
        "Route payload must contain route/action keys directly or a nested "
        "'route_payload', 'gaussian_route_payload', or 'gs_action_payload' dictionary."
    )


def _optional_vector(value: Any, total_gaussians: int) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    vector = value.detach().to(dtype=torch.float32).reshape(-1)
    if vector.shape[0] != int(total_gaussians):
        raise ValueError(
            f"Route payload vector length mismatch: {vector.shape[0]} vs total_gaussians={int(total_gaussians)}"
        )
    return vector.cpu()


def _required_or_default(
    payload: Dict[str, Any],
    keys: tuple[str, ...],
    total_gaussians: int,
    default_value: float,
) -> torch.Tensor:
    for key in keys:
        vector = _optional_vector(payload.get(key), total_gaussians)
        if vector is not None:
            return vector
    return torch.full((int(total_gaussians),), float(default_value), dtype=torch.float32)


def _infer_total_gaussians(payload: Dict[str, Any]) -> int:
    declared = payload.get("num_gaussians")
    if declared is not None:
        try:
            return int(declared)
        except (TypeError, ValueError):
            pass
    for value in payload.values():
        if torch.is_tensor(value):
            return int(value.reshape(-1).shape[0])
        if isinstance(value, np.ndarray):
            return int(np.asarray(value).reshape(-1).shape[0])
    raise ValueError("Unable to infer total_gaussians from route payload.")


def _clamp01(values: torch.Tensor) -> torch.Tensor:
    return torch.clamp(values.to(dtype=torch.float32), 0.0, 1.0)


def _build_overlap_route_groups(
    p_detail: torch.Tensor,
    p_suppress: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_detail = _clamp01(p_detail * (1.0 - p_suppress))
    risky_useful = _clamp01(p_detail * p_suppress)
    harmful_outlier = _clamp01((1.0 - p_detail) * p_suppress)
    return clean_detail, risky_useful, harmful_outlier


def build_execution_payload_from_route(
    route_payload: Dict[str, torch.Tensor],
    cfg: RouteExecutionConfig | None = None,
) -> Dict[str, torch.Tensor]:
    cfg = cfg or RouteExecutionConfig()
    total_gaussians = _infer_total_gaussians(route_payload)

    p_attach = _required_or_default(route_payload, ("p_attach", "attach_strength"), total_gaussians, default_value=0.0)
    p_detail = _required_or_default(route_payload, ("p_detail", "detail_weight"), total_gaussians, default_value=1.0)
    p_suppress = _required_or_default(
        route_payload,
        ("p_suppress", "suppress_strength"),
        total_gaussians,
        default_value=0.0,
    )
    radius_gate = _required_or_default(route_payload, ("radius_gate",), total_gaussians, default_value=1.0)

    p_attach = _clamp01(p_attach)
    p_detail = _clamp01(p_detail)
    p_suppress = _clamp01(p_suppress)
    radius_gate = _clamp01(radius_gate)
    mode_name = str(cfg.mode)
    clean_detail, risky_useful, harmful_outlier = _build_overlap_route_groups(p_detail, p_suppress)
    suppress_effective = _clamp01(p_suppress * (1.0 - float(cfg.detail_protect_beta) * p_detail))

    if mode_name in {"suppress_only_v0p5", "suppress_only_v0p6", "suppress_only_v0p7"}:
        attach_weight = torch.zeros_like(p_attach)
        detail_weight = torch.ones_like(p_detail)
        prior_color_strength = torch.zeros_like(p_detail)
        position_weight = torch.zeros_like(p_attach)
        scaling_weight = torch.zeros_like(p_attach)
        rotation_weight = torch.zeros_like(p_attach)
        appearance_weight = torch.ones_like(p_detail)
        opacity_weight = _clamp01(torch.maximum(detail_weight, float(cfg.suppress_update_floor) * suppress_effective))
    elif mode_name == "suppress_only_v0p8":
        suppress_effective = harmful_outlier.clone()
        attach_weight = torch.zeros_like(p_attach)
        detail_weight = p_detail.clone()
        prior_color_strength = torch.zeros_like(p_detail)
        position_weight = torch.zeros_like(p_attach)
        scaling_weight = torch.zeros_like(p_attach)
        rotation_weight = torch.zeros_like(p_attach)
        appearance_weight = p_detail.clone()
        opacity_weight = _clamp01(torch.maximum(detail_weight, float(cfg.suppress_update_floor) * harmful_outlier))
    elif mode_name == "detail_preserve_v0p8":
        suppress_effective = harmful_outlier.clone()
        attach_guard = _clamp01(1.0 - p_suppress)
        attach_weight = _clamp01(float(cfg.detail_attach_strength) * p_attach * radius_gate * attach_guard)
        detail_weight = p_detail.clone()
        prior_color_strength = _clamp01(float(cfg.detail_prior_color_strength_scale) * p_detail * radius_gate)
        position_weight = _clamp01(float(cfg.detail_geometry_strength) * p_attach * radius_gate * attach_guard)
        scaling_weight = position_weight.clone()
        rotation_weight = position_weight.clone()
        appearance_weight = p_detail.clone()
        opacity_weight = _clamp01(torch.maximum(detail_weight, float(cfg.suppress_update_floor) * harmful_outlier))
    else:
        attach_weight = _clamp01(
            float(cfg.detail_attach_strength) * p_attach * radius_gate * (1.0 - suppress_effective)
        )
        # v0.7: risky-but-useful detail should keep appearance ownership even when
        # suppress stays high. Suppress only limits geometry/opacity/clamp.
        detail_weight = _clamp01(p_detail)
        prior_color_strength = _clamp01(float(cfg.detail_prior_color_strength_scale) * p_detail * radius_gate)
        position_weight = _clamp01(
            float(cfg.detail_geometry_strength) * p_attach * radius_gate * (1.0 - suppress_effective)
        )
        scaling_weight = position_weight.clone()
        rotation_weight = position_weight.clone()
        appearance_weight = detail_weight.clone()
        opacity_weight = _clamp01(
            torch.maximum(detail_weight, float(cfg.suppress_update_floor) * suppress_effective)
        )

    update_strength = _clamp01(
        torch.maximum(torch.maximum(position_weight, appearance_weight), opacity_weight)
    )
    color_update_strength = appearance_weight.clone()

    execution = {
        "execution_update_strength": update_strength,
        "execution_attach_weight": attach_weight,
        "execution_detail_weight": detail_weight,
        "execution_prior_color_strength": prior_color_strength,
        "execution_suppress_strength": suppress_effective,
        "execution_position_weight": position_weight,
        "execution_scaling_weight": scaling_weight,
        "execution_rotation_weight": rotation_weight,
        "execution_appearance_weight": appearance_weight,
        "execution_opacity_weight": opacity_weight,
        "update_strength": update_strength,
        "attach_strength": attach_weight,
        "detail_weight": detail_weight,
        "prior_color_strength": prior_color_strength,
        "suppress_strength": suppress_effective,
        "geometry_update_strength": position_weight,
        "color_update_strength": color_update_strength,
        "radius_gate": radius_gate,
        "p_attach": p_attach,
        "p_detail": p_detail,
        "p_suppress": p_suppress,
        "clean_detail_strength": clean_detail,
        "risky_useful_strength": risky_useful,
        "harmful_outlier_strength": harmful_outlier,
    }
    for key in ("p_promote", "p_spawn_source", "contribution_sum", "support_count"):
        vector = _optional_vector(route_payload.get(key), total_gaussians)
        if vector is not None:
            execution[key] = vector
    return execution


def load_route_payload(path: str | Path | None, total_gaussians: int) -> Dict[str, torch.Tensor] | None:
    if not path:
        return None

    payload_path = Path(path)
    suffix = payload_path.suffix.lower()
    if suffix == ".npz":
        loaded = np.load(payload_path)
        raw_payload: Dict[str, Any] = {key: loaded[key] for key in loaded.files}
    else:
        raw_payload = torch.load(payload_path, map_location="cpu")
    if not isinstance(raw_payload, dict):
        raise TypeError(f"Unsupported route payload object type: {type(raw_payload)!r}")

    payload = _unwrap_route_payload(raw_payload)
    route: Dict[str, torch.Tensor] = {}

    route["attach_strength"] = _required_or_default(
        payload,
        ("attach_strength", "effective_attach", "p_attach"),
        total_gaussians,
        default_value=1.0,
    )
    route["detail_weight"] = _required_or_default(
        payload,
        ("detail_weight", "effective_detail", "p_detail"),
        total_gaussians,
        default_value=1.0,
    )
    route["prior_color_strength"] = _required_or_default(
        payload,
        ("prior_color_strength",),
        total_gaussians,
        default_value=1.0,
    )
    route["suppress_strength"] = _required_or_default(
        payload,
        ("suppress_strength", "effective_suppress", "p_suppress"),
        total_gaussians,
        default_value=0.0,
    )
    route["radius_gate"] = _required_or_default(
        payload,
        ("radius_gate",),
        total_gaussians,
        default_value=1.0,
    )

    geom = _optional_vector(payload.get("geometry_update_strength"), total_gaussians)
    color = _optional_vector(payload.get("color_update_strength"), total_gaussians)
    update = _optional_vector(payload.get("update_strength"), total_gaussians)

    if geom is None:
        geom = torch.clamp(route["attach_strength"] * route["radius_gate"], 0.0, 1.0)
    if color is None:
        color = torch.clamp(torch.maximum(route["detail_weight"], route["prior_color_strength"]), 0.0, 1.0)
    if update is None:
        update = torch.maximum(torch.maximum(geom, color), route["suppress_strength"])

    route["geometry_update_strength"] = geom
    route["color_update_strength"] = color
    route["update_strength"] = torch.clamp(update, 0.0, 1.0)

    for key in (
        "p_attach",
        "p_detail",
        "p_suppress",
        "p_promote",
        "p_spawn_source",
        "contribution_sum",
        "support_count",
    ):
        vector = _optional_vector(payload.get(key), total_gaussians)
        if vector is not None:
            route[key] = vector

    execution_key_defaults = {
        "execution_update_strength": 1.0,
        "execution_attach_weight": 1.0,
        "execution_detail_weight": 1.0,
        "execution_prior_color_strength": 1.0,
        "execution_suppress_strength": 0.0,
        "execution_position_weight": 1.0,
        "execution_scaling_weight": 1.0,
        "execution_rotation_weight": 1.0,
        "execution_appearance_weight": 1.0,
        "execution_opacity_weight": 1.0,
    }
    for key, default_value in execution_key_defaults.items():
        vector = _optional_vector(payload.get(key), total_gaussians)
        if vector is not None:
            route[key] = vector

    meta = payload.get("meta")
    if isinstance(meta, dict):
        route["meta"] = meta

    return route


def _align_gaussian_vector(weight: torch.Tensor, total_gaussians: int, fill_value: float) -> torch.Tensor:
    weight = weight.to(device="cuda", dtype=torch.float32).reshape(-1)
    current = int(weight.shape[0])
    target = int(total_gaussians)
    if current == target:
        return weight
    if current > target:
        return weight[:target]
    pad = torch.full((target - current,), float(fill_value), device=weight.device, dtype=weight.dtype)
    return torch.cat((weight, pad), dim=0)


def resolve_route_runtime_state(
    route_payload: Dict[str, torch.Tensor] | None,
    total_gaussians: int,
    min_weight: float,
    attach_scale: float,
    detail_scale: float,
    prior_color_scale: float,
    geometry_scale: float = 1.0,
    appearance_scale: float = 1.0,
    suppress_scale: float = 1.0,
) -> Dict[str, torch.Tensor] | None:
    if route_payload is None:
        return None

    if "execution_position_weight" in route_payload:
        update_strength = _align_gaussian_vector(
            route_payload.get("execution_update_strength", route_payload["update_strength"]),
            total_gaussians,
            fill_value=1.0,
        )
        attach_strength = _align_gaussian_vector(
            route_payload.get("execution_attach_weight", route_payload["attach_strength"]),
            total_gaussians,
            fill_value=1.0,
        )
        detail_weight = _align_gaussian_vector(
            route_payload.get("execution_detail_weight", route_payload["detail_weight"]),
            total_gaussians,
            fill_value=1.0,
        )
        prior_color_strength = _align_gaussian_vector(
            route_payload.get("execution_prior_color_strength", route_payload["prior_color_strength"]),
            total_gaussians,
            fill_value=1.0,
        )
        suppress_strength = _align_gaussian_vector(
            route_payload.get("execution_suppress_strength", route_payload["suppress_strength"]),
            total_gaussians,
            fill_value=0.0,
        )
        radius_gate = _align_gaussian_vector(route_payload["radius_gate"], total_gaussians, fill_value=1.0)
        position_weight = _align_gaussian_vector(
            route_payload.get("execution_position_weight", route_payload.get("geometry_update_strength")),
            total_gaussians,
            fill_value=1.0,
        )
        scaling_weight = _align_gaussian_vector(
            route_payload.get("execution_scaling_weight", route_payload.get("geometry_update_strength")),
            total_gaussians,
            fill_value=1.0,
        )
        rotation_weight = _align_gaussian_vector(
            route_payload.get("execution_rotation_weight", route_payload.get("geometry_update_strength")),
            total_gaussians,
            fill_value=1.0,
        )
        appearance_weight = _align_gaussian_vector(
            route_payload.get("execution_appearance_weight", route_payload.get("color_update_strength")),
            total_gaussians,
            fill_value=1.0,
        )
        opacity_weight = _align_gaussian_vector(
            route_payload.get("execution_opacity_weight", route_payload["update_strength"]),
            total_gaussians,
            fill_value=1.0,
        )

        update_strength = torch.clamp(update_strength, 0.0, 1.0)
        attach_strength = torch.clamp(attach_strength * float(attach_scale), 0.0, 1.0)
        detail_weight = torch.clamp(detail_weight * float(detail_scale), 0.0, 1.0)
        prior_color_strength = torch.clamp(prior_color_strength * float(prior_color_scale), 0.0, 1.0)
        suppress_strength = torch.clamp(suppress_strength * float(suppress_scale), 0.0, 1.0)
        radius_gate = torch.clamp(radius_gate, 0.0, 1.0)
        position_weight = torch.clamp(position_weight * float(geometry_scale), 0.0, 1.0)
        scaling_weight = torch.clamp(scaling_weight * float(geometry_scale), 0.0, 1.0)
        rotation_weight = torch.clamp(rotation_weight * float(geometry_scale), 0.0, 1.0)
        appearance_weight = torch.clamp(appearance_weight * float(appearance_scale), 0.0, 1.0)
        opacity_weight = torch.clamp(opacity_weight, 0.0, 1.0)

        if float(min_weight) > 0.0:
            update_strength = torch.where(
                update_strength >= float(min_weight),
                update_strength,
                torch.zeros_like(update_strength),
            )

        active_mask = (update_strength > 0.0) | (suppress_strength > 0.0)
        state = {
            "active_mask": active_mask,
            "update_strength": update_strength,
            "attach_strength": attach_strength,
            "detail_weight": detail_weight,
            "prior_color_strength": prior_color_strength,
            "suppress_strength": suppress_strength,
            "radius_gate": radius_gate,
            "geometry_weight": position_weight,
            "position_weight": position_weight,
            "scaling_weight": scaling_weight,
            "rotation_weight": rotation_weight,
            "appearance_weight": appearance_weight,
            "opacity_weight": opacity_weight,
        }

        for key in ("p_attach", "p_detail", "p_suppress", "p_promote", "p_spawn_source"):
            vector = route_payload.get(key)
            if vector is not None:
                state[key] = _align_gaussian_vector(vector, total_gaussians, fill_value=0.0)
        return state

    update_strength = _align_gaussian_vector(route_payload["update_strength"], total_gaussians, fill_value=1.0)
    attach_strength = _align_gaussian_vector(route_payload["attach_strength"], total_gaussians, fill_value=1.0)
    detail_weight = _align_gaussian_vector(route_payload["detail_weight"], total_gaussians, fill_value=1.0)
    prior_color_strength = _align_gaussian_vector(
        route_payload["prior_color_strength"],
        total_gaussians,
        fill_value=1.0,
    )
    suppress_strength = _align_gaussian_vector(
        route_payload["suppress_strength"],
        total_gaussians,
        fill_value=0.0,
    )
    radius_gate = _align_gaussian_vector(route_payload["radius_gate"], total_gaussians, fill_value=1.0)
    geometry_update_strength = _align_gaussian_vector(
        route_payload["geometry_update_strength"],
        total_gaussians,
        fill_value=1.0,
    )
    color_update_strength = _align_gaussian_vector(
        route_payload["color_update_strength"],
        total_gaussians,
        fill_value=1.0,
    )

    update_strength = torch.clamp(update_strength, 0.0, 1.0)
    attach_strength = torch.clamp(attach_strength * float(attach_scale), 0.0, 1.0)
    detail_weight = torch.clamp(detail_weight * float(detail_scale), 0.0, 1.0)
    prior_color_strength = torch.clamp(prior_color_strength * float(prior_color_scale), 0.0, 1.0)
    suppress_strength = torch.clamp(suppress_strength * float(suppress_scale), 0.0, 1.0)
    radius_gate = torch.clamp(radius_gate, 0.0, 1.0)
    geometry_update_strength = torch.clamp(geometry_update_strength * float(geometry_scale), 0.0, 1.0)
    color_update_strength = torch.clamp(color_update_strength * float(appearance_scale), 0.0, 1.0)

    if float(min_weight) > 0.0:
        update_strength = torch.where(
            update_strength >= float(min_weight),
            update_strength,
            torch.zeros_like(update_strength),
        )

    attach_effective = torch.clamp(attach_strength * radius_gate * (1.0 - suppress_strength), 0.0, 1.0)
    detail_effective = torch.clamp(detail_weight * (1.0 - suppress_strength), 0.0, 1.0)
    prior_effective = torch.clamp(prior_color_strength * (1.0 - suppress_strength), 0.0, 1.0)

    position_weight = torch.clamp(geometry_update_strength * (1.0 - suppress_strength), 0.0, 1.0)
    scaling_weight = torch.clamp(torch.maximum(position_weight, suppress_strength), 0.0, 1.0)
    rotation_weight = position_weight
    appearance_weight = torch.clamp(
        torch.maximum(color_update_strength, torch.maximum(detail_effective, prior_effective))
        * (1.0 - 0.5 * suppress_strength),
        0.0,
        1.0,
    )
    opacity_weight = torch.clamp(torch.maximum(update_strength, suppress_strength), 0.0, 1.0)
    active_mask = (update_strength > 0.0) | (suppress_strength > 0.0)

    state = {
        "active_mask": active_mask,
        "update_strength": update_strength,
        "attach_strength": attach_effective,
        "detail_weight": detail_effective,
        "prior_color_strength": prior_effective,
        "suppress_strength": suppress_strength,
        "radius_gate": radius_gate,
        "geometry_weight": position_weight,
        "position_weight": position_weight,
        "scaling_weight": scaling_weight,
        "rotation_weight": rotation_weight,
        "appearance_weight": appearance_weight,
        "opacity_weight": opacity_weight,
    }

    for key in ("p_attach", "p_detail", "p_suppress", "p_promote", "p_spawn_source"):
        vector = route_payload.get(key)
        if vector is not None:
            state[key] = _align_gaussian_vector(vector, total_gaussians, fill_value=0.0)

    return state


@torch.no_grad()
def apply_route_suppression_to_detail_model(
    detail,
    route_payload: Dict[str, torch.Tensor] | None,
    suppress_opacity_scale: float,
    min_scale_gate: float,
    scale_gate_power: float,
    mode: str = "detail_preserve_v0p8",
    detail_protect_beta: float = 0.7,
    scale_gate_suppress_coupling: float = 1.0,
    risky_useful_opacity_coupling: float = 0.35,
    risky_useful_scale_coupling: float = 0.2,
    harmful_scale_coupling: float = 1.0,
    min_opacity: float = 1e-4,
    min_scale: float = 1e-6,
) -> Dict[str, Any]:
    if route_payload is None:
        return {"enabled": False}

    suppress = route_payload.get("p_suppress")
    if suppress is None:
        suppress = route_payload.get("suppress_strength")
    p_detail = route_payload.get("p_detail")
    radius_gate = route_payload.get("radius_gate")
    if suppress is None and radius_gate is None:
        return {"enabled": False}

    device = detail.get_xyz.device
    total_gaussians = int(detail.get_xyz.shape[0])
    suppress = (
        torch.zeros((total_gaussians,), device=device, dtype=torch.float32)
        if suppress is None
        else suppress.to(device=device, dtype=torch.float32).reshape(-1)
    )
    radius_gate = (
        torch.ones((total_gaussians,), device=device, dtype=torch.float32)
        if radius_gate is None
        else radius_gate.to(device=device, dtype=torch.float32).reshape(-1)
    )

    suppress = torch.clamp(suppress, 0.0, 1.0)
    mode_name = str(mode)
    clean_detail = None
    risky_useful = None
    harmful_outlier = None
    if mode_name in {"detail_preserve_v0p8", "suppress_only_v0p8"} and p_detail is not None:
        p_detail = torch.clamp(p_detail.to(device=device, dtype=torch.float32).reshape(-1), 0.0, 1.0)
        clean_detail, risky_useful, harmful_outlier = _build_overlap_route_groups(p_detail, suppress)
        base_scale_gate = torch.clamp(radius_gate, min=float(min_scale_gate), max=1.0)
        base_scale_gate = torch.pow(base_scale_gate, float(scale_gate_power))
        shrink_strength = torch.clamp(
            harmful_outlier * float(harmful_scale_coupling)
            + risky_useful * float(risky_useful_scale_coupling),
            0.0,
            1.0,
        )
        scale_gate = 1.0 - (1.0 - base_scale_gate) * shrink_strength
        opacity_suppress = torch.clamp(
            harmful_outlier + risky_useful * float(risky_useful_opacity_coupling),
            0.0,
            1.0,
        )
        opacity_gate = torch.clamp(
            1.0 - float(suppress_opacity_scale) * opacity_suppress,
            min=float(min_opacity),
            max=1.0,
        )
        effective_suppress = harmful_outlier
        scale_gate_mode = "overlap_aware_v0p8"
    else:
        if p_detail is not None:
            p_detail = p_detail.to(device=device, dtype=torch.float32).reshape(-1)
            suppress = torch.clamp(
                suppress * (1.0 - float(detail_protect_beta) * torch.clamp(p_detail, 0.0, 1.0)),
                0.0,
                1.0,
            )
        base_scale_gate = torch.clamp(radius_gate, min=float(min_scale_gate), max=1.0)
        base_scale_gate = torch.pow(base_scale_gate, float(scale_gate_power))
        shrink_strength = torch.clamp(suppress * float(scale_gate_suppress_coupling), 0.0, 1.0)
        scale_gate = 1.0 - (1.0 - base_scale_gate) * shrink_strength
        opacity_suppress = suppress
        opacity_gate = torch.clamp(
            1.0 - float(suppress_opacity_scale) * suppress,
            min=float(min_opacity),
            max=1.0,
        )
        effective_suppress = suppress
        scale_gate_mode = "suppress_coupled_v0p6"

    new_opacity = torch.clamp(detail.get_opacity.detach().reshape(-1) * opacity_gate, min=float(min_opacity), max=0.999)
    new_scaling = torch.clamp(detail.get_scaling.detach() * scale_gate[:, None], min=float(min_scale))
    detail._opacity.data.copy_(detail.inverse_opacity_activation(new_opacity[:, None]))
    detail._scaling.data.copy_(detail.scaling_inverse_activation(new_scaling))

    summary = {
        "enabled": True,
        "mode": mode_name,
        "detail_protect_beta": float(detail_protect_beta),
        "scale_gate_mode": scale_gate_mode,
        "scale_gate_suppress_coupling": float(scale_gate_suppress_coupling),
        "suppress_strength_stats": vector_stats(effective_suppress),
        "raw_suppress_strength_stats": vector_stats(suppress),
        "radius_gate_stats": vector_stats(radius_gate),
        "base_scale_gate_stats": vector_stats(base_scale_gate),
        "scale_shrink_strength_stats": vector_stats(shrink_strength),
        "applied_scale_gate_stats": vector_stats(scale_gate),
        "opacity_suppress_strength_stats": vector_stats(opacity_suppress),
        "applied_opacity_gate_stats": vector_stats(opacity_gate),
    }
    if clean_detail is not None and risky_useful is not None and harmful_outlier is not None:
        summary["clean_detail_strength_stats"] = vector_stats(clean_detail)
        summary["risky_useful_strength_stats"] = vector_stats(risky_useful)
        summary["harmful_outlier_strength_stats"] = vector_stats(harmful_outlier)
        summary["risky_useful_opacity_coupling"] = float(risky_useful_opacity_coupling)
        summary["risky_useful_scale_coupling"] = float(risky_useful_scale_coupling)
        summary["harmful_scale_coupling"] = float(harmful_scale_coupling)
    return summary

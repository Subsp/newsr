from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.gaussian_model import GaussianModel
from utils.route_executor import load_route_payload, vector_stats


def _resolve_model_iteration(model_path: Path, iteration: int) -> int:
    if iteration >= 0:
        return int(iteration)
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"point_cloud directory not found: {point_cloud_root}")
    candidates: List[int] = []
    for child in point_cloud_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            candidates.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found under {point_cloud_root}")
    return int(max(candidates))


def _copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src_file = src_model_path / name
        if src_file.exists():
            shutil.copy2(src_file, dst_model_path / name)


def _clone_subset_gaussians(base: GaussianModel, mask: torch.Tensor) -> GaussianModel:
    mask = mask.to(device=base.get_xyz.device, dtype=torch.bool).reshape(-1)
    count = int(mask.sum().item())
    subset = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    subset.active_sh_degree = int(base.active_sh_degree)
    subset.spatial_lr_scale = float(base.spatial_lr_scale)

    subset._xyz = nn.Parameter(base._xyz.detach()[mask].clone().requires_grad_(False))
    subset._features_dc = nn.Parameter(base._features_dc.detach()[mask].clone().requires_grad_(False))
    subset._features_rest = nn.Parameter(base._features_rest.detach()[mask].clone().requires_grad_(False))
    subset._opacity = nn.Parameter(base._opacity.detach()[mask].clone().requires_grad_(False))
    subset._scaling = nn.Parameter(base._scaling.detach()[mask].clone().requires_grad_(False))
    subset._rotation = nn.Parameter(base._rotation.detach()[mask].clone().requires_grad_(False))

    if (
        isinstance(base.filter_3D, torch.Tensor)
        and base.filter_3D.ndim > 0
        and base.filter_3D.shape[0] == base.get_xyz.shape[0]
    ):
        subset.filter_3D = base.filter_3D.detach()[mask].clone()
    else:
        subset.filter_3D = base.filter_3D.detach().clone()

    subset.max_radii2D = torch.zeros((count,), dtype=torch.float32, device="cuda")
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device="cuda")
    subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device="cuda")
    subset.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device="cuda")
    subset.denom = torch.zeros((count, 1), dtype=torch.float32, device="cuda")

    subset.init_tracking_state(count)
    if base._source_tag.shape[0] == base.get_xyz.shape[0]:
        subset._source_tag = base._source_tag.detach()[mask].clone()
    if base._seed_id.shape[0] == base.get_xyz.shape[0]:
        subset._seed_id = base._seed_id.detach()[mask].clone()
    if base._generation.shape[0] == base.get_xyz.shape[0]:
        subset._generation = base._generation.detach()[mask].clone()
    if base._edge_touched.shape[0] == base.get_xyz.shape[0]:
        subset._edge_touched = base._edge_touched.detach()[mask].clone()
    if base._edge_touch_iter.shape[0] == base.get_xyz.shape[0]:
        subset._edge_touch_iter = base._edge_touch_iter.detach()[mask].clone()
    return subset


def _canonical_group_name(name: str) -> str:
    lowered = str(name).strip().lower()
    if lowered in {"all", "full", "detail_all", "full_detail"}:
        return "full"
    return lowered


def _build_group_masks(
    route_payload: Dict[str, torch.Tensor],
    groups: Iterable[str],
    group_mode: str,
    threshold: float,
) -> Dict[str, torch.Tensor]:
    p_attach = route_payload.get("p_attach")
    p_detail = route_payload.get("p_detail")
    p_suppress = route_payload.get("p_suppress")
    if p_attach is None or p_detail is None or p_suppress is None:
        raise KeyError("Route payload must contain p_attach, p_detail, and p_suppress for group export.")

    p_attach = p_attach.reshape(-1).float()
    p_detail = p_detail.reshape(-1).float()
    p_suppress = p_suppress.reshape(-1).float()
    total = int(p_attach.shape[0])

    canonical_groups = [_canonical_group_name(name) for name in groups]
    masks: Dict[str, torch.Tensor] = {}
    if "full" in canonical_groups:
        masks["full"] = torch.ones((total,), dtype=torch.bool)

    if group_mode == "argmax":
        ownership = torch.stack((p_attach, p_detail, p_suppress), dim=1)
        owner = ownership.argmax(dim=1)
        candidate_masks = {
            "attach": owner == 0,
            "detail": owner == 1,
            "suppress": owner == 2,
        }
    elif group_mode == "threshold":
        threshold = float(threshold)
        candidate_masks = {
            "attach": p_attach >= threshold,
            "detail": p_detail >= threshold,
            "suppress": p_suppress >= threshold,
        }
    else:
        raise ValueError(f"Unsupported group_mode: {group_mode}")

    for group_name in ("attach", "detail", "suppress"):
        if group_name in canonical_groups:
            masks[group_name] = candidate_masks[group_name]
    return masks


def _load_gaussians(detail_model_path: Path, iteration: int, sh_degree: int) -> GaussianModel:
    point_dir = detail_model_path / "point_cloud" / f"iteration_{int(iteration)}"
    ply_path = point_dir / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"point cloud not found: {ply_path}")
    tags_path = point_dir / "gaussian_tags.pt"

    gaussians = GaussianModel(int(sh_degree))
    gaussians.load_ply(str(ply_path))
    if tags_path.is_file():
        gaussians.load_tracking_metadata(str(tags_path))
    return gaussians


def _write_group_model(
    subset: GaussianModel,
    src_model_path: Path,
    output_root: Path,
    group_name: str,
    iteration: int,
    source_count: int,
) -> Dict[str, str | int]:
    model_root = output_root / group_name
    _copy_render_config(src_model_path, model_root)
    point_dir = model_root / "point_cloud" / f"iteration_{int(iteration)}"
    point_dir.mkdir(parents=True, exist_ok=True)

    ply_path = point_dir / "point_cloud.ply"
    tags_path = point_dir / "gaussian_tags.pt"
    count = int(subset.get_xyz.shape[0])

    subset.save_ply(str(ply_path))
    subset.save_tracking_metadata(str(tags_path))
    (point_dir / "num_gaussians.json").write_text(
        json.dumps(
            {
                "num_gaussians": count,
                "source_group": group_name,
                "source_count": int(source_count),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "model_root": str(model_root),
        "point_dir": str(point_dir),
        "ply_path": str(ply_path),
        "tags_path": str(tags_path),
        "count": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export route-selected Gaussian groups as renderable model-style PLY dirs.")
    parser.add_argument("--detail_model_path", required=True)
    parser.add_argument("--route_payload", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--groups", default="detail,suppress")
    parser.add_argument("--group_mode", choices=["argmax", "threshold"], default="threshold")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    detail_model_path = Path(args.detail_model_path).expanduser().resolve()
    route_payload_path = Path(args.route_payload).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration = _resolve_model_iteration(detail_model_path, int(args.iteration))
    gaussians = _load_gaussians(detail_model_path, iteration=iteration, sh_degree=int(args.sh_degree))
    route_payload = load_route_payload(route_payload_path, total_gaussians=int(gaussians.get_xyz.shape[0]))
    if route_payload is None:
        raise RuntimeError(f"Failed to load route payload: {route_payload_path}")

    group_names = [name.strip() for name in str(args.groups).split(",") if name.strip()]
    group_masks = _build_group_masks(
        route_payload=route_payload,
        groups=group_names,
        group_mode=str(args.group_mode),
        threshold=float(args.threshold),
    )
    if not group_masks:
        hint = ""
        if group_names == ["0"]:
            hint = " Detected shell GROUPS='0'; use ROUTE_GROUPS=detail,suppress with the wrapper script."
        raise RuntimeError(f"No route groups were constructed from groups={group_names!r}.{hint}")

    summary = {
        "detail_model_path": str(detail_model_path),
        "route_payload": str(route_payload_path),
        "output_dir": str(output_dir),
        "iteration": int(iteration),
        "group_mode": str(args.group_mode),
        "threshold": float(args.threshold),
        "groups_requested": group_names,
        "route_stats": {},
        "group_exports": {},
    }

    print(
        f"[export-route-ply] detail model  : {detail_model_path}\n"
        f"[export-route-ply] route payload : {route_payload_path}\n"
        f"[export-route-ply] output dir    : {output_dir}\n"
        f"[export-route-ply] group mode    : {args.group_mode}\n"
        f"[export-route-ply] threshold     : {float(args.threshold):.4f}\n"
        f"[export-route-ply] iteration     : {iteration}\n"
        f"[export-route-ply] groups        : {group_names}"
    )

    for key in ("p_attach", "p_detail", "p_suppress"):
        if key in route_payload:
            stats = vector_stats(route_payload[key])
            summary["route_stats"][key] = stats
            print(f"[export-route-ply] {key} stats: {stats}", flush=True)

    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for group_name, mask in group_masks.items():
        selected = int(mask.sum().item())
        total = int(mask.numel())
        print(f"[export-route-ply] group={group_name} selected={selected}/{total}", flush=True)
        summary["group_exports"][group_name] = {
            "selected": selected,
            "total": total,
            "exported": False,
        }
        if selected <= 0:
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            continue

        subset = _clone_subset_gaussians(gaussians, mask)
        export_info = _write_group_model(
            subset=subset,
            src_model_path=detail_model_path,
            output_root=output_dir,
            group_name=group_name,
            iteration=iteration,
            source_count=total,
        )
        summary["group_exports"][group_name].update(export_info)
        summary["group_exports"][group_name]["exported"] = True
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("[export-route-ply] done", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.gaussian_model import GaussianModel
from utils.gaussian_route_features import HeuristicGaussianRouteConfig, build_heuristic_route_payload
from utils.route_executor import save_route_payload


def _resolve_model_iteration(model_path: Path, iteration: int) -> int:
    if iteration >= 0:
        return int(iteration)
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"point_cloud directory not found: {point_cloud_root}")
    candidates = []
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


def main() -> None:
    parser = ArgumentParser(description="Export heuristic Gaussian route payload for HRGS/route-aware SOF v0.")
    parser.add_argument("--detail_model_path", required=True)
    parser.add_argument("--carrier_payload", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--action_payload", default=None)
    parser.add_argument("--detail_iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--max_views", type=int, default=32)
    parser.add_argument("--radius_ref_px", type=float, default=96.0)
    parser.add_argument("--radius_temperature_px", type=float, default=24.0)
    parser.add_argument("--radius_gate_min", type=float, default=0.1)
    parser.add_argument("--surface_confidence_floor", type=float, default=0.05)
    parser.add_argument("--surface_distance_scale", type=float, default=2.0)
    parser.add_argument("--opacity_center", type=float, default=0.35)
    parser.add_argument("--opacity_temperature", type=float, default=0.15)
    parser.add_argument("--suppress_update_floor", type=float, default=0.15)
    parser.add_argument("--detail_boost", type=float, default=1.0)
    args = parser.parse_args()

    detail_model_path = Path(args.detail_model_path).expanduser().resolve()
    detail_iteration = _resolve_model_iteration(detail_model_path, int(args.detail_iteration))
    detail_ply = detail_model_path / "point_cloud" / f"iteration_{detail_iteration}" / "point_cloud.ply"
    if not detail_ply.is_file():
        raise FileNotFoundError(f"Detail point cloud not found: {detail_ply}")

    detail = GaussianModel(int(args.sh_degree), use_SBs=False)
    detail.load_ply(str(detail_ply))

    route_cfg = HeuristicGaussianRouteConfig(
        images_subdir=str(args.images_subdir),
        max_views=int(args.max_views),
        radius_ref_px=float(args.radius_ref_px),
        radius_temperature_px=float(args.radius_temperature_px),
        radius_gate_min=float(args.radius_gate_min),
        surface_confidence_floor=float(args.surface_confidence_floor),
        surface_distance_scale=float(args.surface_distance_scale),
        opacity_center=float(args.opacity_center),
        opacity_temperature=float(args.opacity_temperature),
        suppress_update_floor=float(args.suppress_update_floor),
        detail_boost=float(args.detail_boost),
    )
    payload = build_heuristic_route_payload(
        detail=detail,
        carrier_payload_path=args.carrier_payload,
        scene_root=args.scene_root,
        model_path=str(detail_model_path),
        action_payload_path=args.action_payload,
        cfg=route_cfg,
    )

    output_path = save_route_payload(args.output_path, payload)
    summary = {
        "detail_model_path": str(detail_model_path),
        "detail_iteration": int(detail_iteration),
        "detail_ply": str(detail_ply),
        "carrier_payload": str(Path(args.carrier_payload).expanduser().resolve()),
        "action_payload": str(Path(args.action_payload).expanduser().resolve()) if args.action_payload else None,
        "output_path": str(output_path),
        "num_gaussians": int(payload["num_gaussians"]),
        "stats": payload["stats"],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

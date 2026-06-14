from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams
from gaussian_renderer import render_simple
from models.hrgs_refiner import HRGSRefiner
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.gs_action_aggregator import aggregate_gs_actions, save_gs_action_payload
from utils.hrgs_scene_layout import (
    build_hrgs_manifest_payload,
    build_hrgs_view_records,
    resolve_hrgs_scene_layout,
    save_hrgs_manifest,
)
from utils.prior_injection import load_rgb_image, normalize_image_name
from utils.surface_payload_lifter import (
    SurfacePayloadLifterConfig,
    lift_surface_payload,
    save_surface_payload_npz,
)
from utils.vggt_adapter import FrozenVGGTAdapter, VGGTAdapterConfig
from utils.visibility_records import VisibilityRecordConfig, build_coarse_visibility_records


def _log(message: str) -> None:
    print(message, flush=True)


def _resize_image_hwc(image_hwc: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    target_h, target_w = target_hw
    image = image_hwc.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    resized = F.interpolate(image, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).contiguous()


def _load_image_for_view(path: str, target_hw: Tuple[int, int]) -> torch.Tensor:
    image = load_rgb_image(Path(path))
    if tuple(image.shape[:2]) != tuple(target_hw):
        image = _resize_image_hwc(image, target_hw)
    return image


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=-1,
        white_background=False,
        data_device="cuda",
        eval=False,
        alpha_mask=False,
        init_type="sfm",
    )


def _build_camera_index(cameras: Sequence[object]) -> Dict[str, object]:
    return {normalize_image_name(cam.image_name): cam for cam in cameras}


def _select_uniform_records(records: Sequence[Dict[str, object]], max_views: int) -> List[Dict[str, object]]:
    if max_views <= 0 or len(records) <= max_views:
        return list(records)
    indices = np.linspace(0, len(records) - 1, num=max_views, dtype=np.int64)
    indices = np.unique(indices)
    return [records[int(idx)] for idx in indices.tolist()]


def _build_camera_bundle(cameras: Sequence[object], device: str) -> Dict[str, torch.Tensor]:
    intrinsics = []
    world_to_view = []
    cam_to_world = []
    for cam in cameras:
        K = torch.tensor(
            [
                [float(cam.focal_x), 0.0, float(cam.image_width) / 2.0],
                [0.0, float(cam.focal_y), float(cam.image_height) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        w2v = cam.world_view_transform.transpose(0, 1).detach().to(dtype=torch.float32).cpu()
        c2w = torch.linalg.inv(w2v)
        intrinsics.append(K)
        world_to_view.append(w2v)
        cam_to_world.append(c2w)
    return {
        "intrinsics": torch.stack(intrinsics, dim=0).unsqueeze(0).to(device=device),
        "world_to_view": torch.stack(world_to_view, dim=0).unsqueeze(0).to(device=device),
        "cam_to_world": torch.stack(cam_to_world, dim=0).unsqueeze(0).to(device=device),
    }


def _downsample_bvchw(tensor: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    b, v, c, _, _ = tensor.shape
    out = F.interpolate(
        tensor.flatten(0, 1),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return out.unflatten(0, (b, v))


def _nested_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nested_to_cpu(item) for item in value]
    return value


def _maybe_load_refiner_checkpoint(model: HRGSRefiner, checkpoint_path: str | None) -> str | None:
    if not checkpoint_path:
        return None
    blob = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(blob, dict):
        state_dict = blob.get("state_dict", blob.get("model", blob))
    else:
        state_dict = blob
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported HRGSRefiner checkpoint type: {type(blob)!r}")
    result = model.load_state_dict(state_dict, strict=False)
    return f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HRGSRefiner-v1 on a real scene with VGGT priors.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--gs_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vggt_root", default="/root/autodl-tmp/vggt")
    parser.add_argument("--source_images_subdir", default="images_8")
    parser.add_argument("--target_images_subdir", default="images_2")
    parser.add_argument("--priors_dir", default=None)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--require_priors", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refiner_checkpoint", default=None)
    parser.add_argument("--vggt_cache", default=None)
    parser.add_argument("--visibility_downsample", type=int, default=8)
    parser.add_argument("--visibility_topk", type=int, default=4)
    parser.add_argument("--visibility_max_visible", type=int, default=30000)
    parser.add_argument("--visibility_max_patch_radius", type=int, default=1)
    parser.add_argument("--surface_min_confidence", type=float, default=None)
    parser.add_argument("--surface_max_disagreement", type=float, default=None)
    parser.add_argument("--surface_min_views_per_cluster", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = resolve_hrgs_scene_layout(
        args.scene_root,
        source_images_subdir=args.source_images_subdir,
        target_images_subdir=args.target_images_subdir,
        priors_dir=args.priors_dir,
        vggt_root=args.vggt_root,
        repo_root=str(REPO_ROOT),
    )
    records = build_hrgs_view_records(layout, require_priors=bool(args.require_priors))

    dataset_args = _build_dataset_args(layout.scene_root, args.gs_model_path, layout.target_images_subdir)
    dataset = ModelParams(None).extract(dataset_args)
    gaussians = GaussianModel(dataset.sh_degree, use_SBs=False)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.load_iteration,
        shuffle=False,
        skip_train=False,
        skip_test=True,
    )
    cameras = scene.getTrainCameras().copy()
    camera_index = _build_camera_index(cameras)

    matched = [record for record in records if str(record["image_name"]) in camera_index]
    selected = _select_uniform_records(matched, int(args.max_views))
    if not selected:
        raise RuntimeError("No matched views found between manifest records and scene cameras.")

    selected_cameras = [camera_index[str(record["image_name"])] for record in selected]
    target_hw = (int(selected_cameras[0].image_height), int(selected_cameras[0].image_width))

    _log(f"[hrgs-scene] selected views        : {len(selected)}")
    _log(f"[hrgs-scene] target resolution    : {target_hw[0]}x{target_hw[1]}")
    _log("[hrgs-scene] computing 3D filter ...")
    gaussians.compute_3D_filter(selected_cameras, CUDA=False)
    _log("[hrgs-scene] 3D filter done")
    background = torch.zeros((3,), dtype=torch.float32, device=device)

    render_pkgs = []
    render_rgb = []
    render_depth = []
    render_normal = []
    render_alpha = []
    render_diag = []
    sr_images = []
    lr_up_images = []
    lr_images = []

    for view_idx, (record, camera) in enumerate(zip(selected, selected_cameras), start=1):
        _log(f"[hrgs-scene] render view {view_idx}/{len(selected)} : {record['image_name']}")
        render_pkg = render_simple(camera, gaussians, background)
        render_pkgs.append(render_pkg)
        render_rgb.append(render_pkg["render"].detach().to(device=device))
        render_depth.append(render_pkg["depth"].detach().to(device=device))
        render_normal.append(render_pkg["normal"].detach().to(device=device))
        render_alpha.append(render_pkg["alpha"].detach().to(device=device))
        diag = torch.cat(
            [
                render_pkg["distortion"].detach(),
                render_pkg["alpha"].detach(),
                torch.clamp(render_pkg["depth"].detach(), min=0.0) / torch.clamp(render_pkg["depth"].detach().amax(), min=1.0),
            ],
            dim=0,
        )
        render_diag.append(diag.to(device=device))

        view_hw = (int(camera.image_height), int(camera.image_width))
        sr_path = str(record["prior_path"] or record["target_path"])
        sr_image = _load_image_for_view(sr_path, view_hw)
        lr_image = _load_image_for_view(str(record["lr_path"]), view_hw)
        lr_source_native = load_rgb_image(Path(str(record["lr_path"]))).permute(2, 0, 1).contiguous()
        sr_images.append(sr_image.permute(2, 0, 1).contiguous())
        lr_up_images.append(lr_image.permute(2, 0, 1).contiguous())
        lr_images.append(lr_source_native)

    sr_images_t = torch.stack(sr_images, dim=0).unsqueeze(0).to(device=device)
    lr_up_images_t = torch.stack(lr_up_images, dim=0).unsqueeze(0).to(device=device)
    lr_images_t = torch.stack(lr_images, dim=0).unsqueeze(0).to(device=device)

    camera_bundle = _build_camera_bundle(selected_cameras, device=device)
    vggt_cache_path = args.vggt_cache or str(output_dir / "vggt_prior.pt")
    _log(f"[hrgs-scene] loading VGGT prior ... cache={vggt_cache_path}")
    vggt_adapter = FrozenVGGTAdapter(
        VGGTAdapterConfig(
            vggt_root=layout.vggt_root or args.vggt_root,
            device=device,
        )
    )
    vggt_prior = vggt_adapter.run(
        lr_images_t,
        target_hw=target_hw,
        image_names=[str(record["image_name"]) for record in selected],
        cache_path=vggt_cache_path,
    )
    _log("[hrgs-scene] VGGT prior ready")

    refiner = HRGSRefiner(gs_diag_channels=render_diag[0].shape[0]).to(device)
    load_summary = _maybe_load_refiner_checkpoint(refiner, args.refiner_checkpoint)
    refiner.eval()
    _log("[hrgs-scene] running HRGSRefiner ...")

    gs_buffers = {
        "render_rgb": torch.stack(render_rgb, dim=0).unsqueeze(0),
        "depth": torch.stack(render_depth, dim=0).unsqueeze(0),
        "normal": torch.stack(render_normal, dim=0).unsqueeze(0),
        "alpha": torch.stack(render_alpha, dim=0).unsqueeze(0),
        "diagnostics": torch.stack(render_diag, dim=0).unsqueeze(0),
    }

    outputs = refiner(
        sr_images=sr_images_t,
        lr_up_images=lr_up_images_t,
        gs_buffers=gs_buffers,
        cameras=camera_bundle,
        vggt_prior=vggt_prior,
    )
    _log("[hrgs-scene] HRGSRefiner done")

    _log("[hrgs-scene] lifting surface payload ...")
    if args.surface_min_confidence is None:
        surface_min_confidence = 0.20 if args.refiner_checkpoint is None else 0.35
    else:
        surface_min_confidence = float(args.surface_min_confidence)
    if args.surface_max_disagreement is None:
        surface_max_disagreement = 1.0 if args.refiner_checkpoint is None else 0.10
    else:
        surface_max_disagreement = float(args.surface_max_disagreement)
    if args.surface_min_views_per_cluster is None:
        surface_min_views_per_cluster = 1
    else:
        surface_min_views_per_cluster = int(args.surface_min_views_per_cluster)
    surface_cfg = SurfacePayloadLifterConfig(
        min_confidence=surface_min_confidence,
        max_disagreement=surface_max_disagreement,
        min_views_per_cluster=surface_min_views_per_cluster,
    )
    carrier_payload = lift_surface_payload(
        depth_surf=outputs["surface_2d"]["depth_surf"],
        normal_surf=outputs["surface_2d"]["normal_surf"],
        conf_geo=outputs["surface_2d"]["conf_geo"],
        mask_surface=outputs["surface_2d"]["mask_surface"],
        sr_images=sr_images_t,
        cameras=camera_bundle,
        cfg=surface_cfg,
    )
    _log("[hrgs-scene] surface payload ready")

    _log("[hrgs-scene] building coarse visibility records ...")
    visibility_records = build_coarse_visibility_records(
        gaussians,
        selected_cameras,
        render_pkgs,
        image_hw=target_hw,
        cfg=VisibilityRecordConfig(
            downsample=int(args.visibility_downsample),
            topk=int(args.visibility_topk),
            max_visible_per_view=int(args.visibility_max_visible),
            max_patch_radius=int(args.visibility_max_patch_radius),
        ),
    )
    _log("[hrgs-scene] aggregating Gaussian actions ...")
    coarse_h, coarse_w = [int(x) for x in visibility_records["coarse_hw"].tolist()]
    masks_2d = {
        "mask_update2d": _downsample_bvchw(outputs["surface_2d"]["mask_update2d"], (coarse_h, coarse_w)),
        "mask_surface": _downsample_bvchw(outputs["surface_2d"]["mask_surface"], (coarse_h, coarse_w)),
        "mask_detail": _downsample_bvchw(outputs["surface_2d"]["mask_detail"], (coarse_h, coarse_w)),
        "prior_color_weight2d": _downsample_bvchw(outputs["update_2d"]["prior_color_weight2d"], (coarse_h, coarse_w)),
    }
    action_features_2d = _downsample_bvchw(outputs["update_2d"]["action_features2d"], (coarse_h, coarse_w))
    visibility_records = {
        "gaussian_ids": visibility_records["gaussian_ids"].to(device=device),
        "weights": visibility_records["weights"].to(device=device, dtype=sr_images_t.dtype),
    }
    gs_action_payload = aggregate_gs_actions(
        masks_2d=masks_2d,
        action_features_2d=action_features_2d,
        visibility_records=visibility_records,
        num_gaussians=int(gaussians.get_xyz.shape[0]),
    )
    _log("[hrgs-scene] Gaussian actions ready")

    hrgs_outputs = {
        "surface_2d": outputs["surface_2d"],
        "update_2d": outputs["update_2d"],
        "carrier_payload": carrier_payload,
        "gs_action_payload": gs_action_payload,
        "meta": {
            "selected_image_names": [str(record["image_name"]) for record in selected],
            "target_hw": list(target_hw),
            "load_iteration": int(scene.loaded_iter or -1),
            "refiner_checkpoint": args.refiner_checkpoint,
            "refiner_load_summary": load_summary,
            "vggt_cache": str(vggt_cache_path),
        },
    }

    manifest_payload = build_hrgs_manifest_payload(layout, selected)
    save_hrgs_manifest(output_dir / "selected_views_manifest.json", manifest_payload)
    torch.save(_nested_to_cpu(hrgs_outputs), output_dir / "hrgs_outputs.pt")
    save_surface_payload_npz(output_dir / "carrier_payload.npz", carrier_payload)
    save_gs_action_payload(output_dir / "gs_action_payload.pt", gs_action_payload)

    summary = {
        "num_selected_views": len(selected),
        "num_carriers": int(carrier_payload["centers"].shape[0]),
        "num_valid_carriers": int(carrier_payload["valid_mask"].sum().item()),
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "mean_update_strength": float(gs_action_payload["update_strength"].mean().item()),
        "mean_attach_strength": float(gs_action_payload["attach_strength"].mean().item()),
        "mean_detail_weight": float(gs_action_payload["detail_weight"].mean().item()),
        "refiner_load_summary": load_summary,
        "surface_lifter_config": {
            "min_confidence": surface_min_confidence,
            "max_disagreement": surface_max_disagreement,
            "min_views_per_cluster": surface_min_views_per_cluster,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[hrgs-scene] selected views        : {len(selected)}")
    print(f"[hrgs-scene] carriers             : {summary['num_carriers']}")
    print(f"[hrgs-scene] valid carriers       : {summary['num_valid_carriers']}")
    print(f"[hrgs-scene] gaussians            : {summary['num_gaussians']}")
    print(f"[hrgs-scene] vggt prior cache     : {vggt_cache_path}")
    print(f"[hrgs-scene] outputs              : {output_dir}")
    if load_summary is None:
        print("[hrgs-scene] refiner checkpoint   : random-init (zero-biased heads)")
    else:
        print(f"[hrgs-scene] refiner checkpoint   : {load_summary}")


if __name__ == "__main__":
    main()

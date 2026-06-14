import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import safe_state
from utils.sh_utils import RGB2SH


def parse_rgb(value: str):
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB triplet like '1,0,0', got: {value}")
    return torch.tensor(parts, dtype=torch.float32, device="cuda")


def build_highlight_mask(source_tag: torch.Tensor, highlight_mode: str, highlight_tag: int) -> torch.Tensor:
    if highlight_mode == "added":
        return source_tag != int(GaussianSourceTag.ORIGINAL)
    if highlight_mode == "prior":
        return source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
    if highlight_mode == "probe":
        return source_tag == int(GaussianSourceTag.EXTENSION_PROBE)
    if highlight_mode == "tag":
        return source_tag == int(highlight_tag)
    raise ValueError(f"Unsupported highlight_mode: {highlight_mode}")


def load_external_highlight_mask(mask_payload_path: str, mask_key: str, expected_count: int) -> torch.Tensor:
    payload = torch.load(mask_payload_path, map_location="cpu")
    if mask_key not in payload:
        raise KeyError(f"Mask key '{mask_key}' not found in {mask_payload_path}")
    mask = payload[mask_key]
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.to(dtype=torch.bool)
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D mask tensor for key '{mask_key}', got shape {tuple(mask.shape)}")
    if int(mask.shape[0]) != int(expected_count):
        raise ValueError(
            f"Mask length mismatch for key '{mask_key}': expected {expected_count}, got {int(mask.shape[0])}"
        )
    return mask.to(device="cuda")


def recolor_gaussians_in_place(
    gaussians: GaussianModel,
    highlight_mask: torch.Tensor,
    highlight_rgb: torch.Tensor,
    highlight_opacity: float,
):
    if not torch.any(highlight_mask):
        return

    highlight_sh = RGB2SH(highlight_rgb).to(device="cuda", dtype=torch.float32)
    opacity_value = float(max(1.0e-6, min(1.0 - 1.0e-6, highlight_opacity)))

    with torch.no_grad():
        if gaussians._features_dc.ndim == 3:
            if gaussians._features_dc.shape[1] == 1 and gaussians._features_dc.shape[2] == 3:
                gaussians._features_dc[highlight_mask, 0, :] = highlight_sh[None, :]
            elif gaussians._features_dc.shape[1] == 3 and gaussians._features_dc.shape[2] == 1:
                gaussians._features_dc[highlight_mask, :, 0] = highlight_sh[None, :]
            else:
                raise RuntimeError(f"Unsupported _features_dc shape: {tuple(gaussians._features_dc.shape)}")
        elif gaussians._features_dc.ndim == 2 and gaussians._features_dc.shape[1] == 3:
            gaussians._features_dc[highlight_mask, :] = highlight_sh[None, :]
        else:
            raise RuntimeError(f"Unsupported _features_dc shape: {tuple(gaussians._features_dc.shape)}")

        gaussians._features_rest[highlight_mask] = 0.0
        gaussians._opacity[highlight_mask] = gaussians.inverse_opacity_activation(
            torch.full_like(gaussians._opacity[highlight_mask], opacity_value)
        )


def export_recolored_cloud(dataset, iteration: int, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            skip_train=True,
            skip_test=True,
        )

        if args.highlight_mode == "mask_payload":
            if not args.highlight_mask_pt:
                raise ValueError("--highlight_mask_pt is required when --highlight_mode mask_payload")
            highlight_mask = load_external_highlight_mask(
                mask_payload_path=args.highlight_mask_pt,
                mask_key=args.highlight_mask_key,
                expected_count=int(gaussians.get_xyz.shape[0]),
            )
        else:
            highlight_mask = build_highlight_mask(
                gaussians._source_tag,
                highlight_mode=args.highlight_mode,
                highlight_tag=args.highlight_tag,
            )
        highlight_count = int(highlight_mask.sum().item())

        output_root = Path(args.output_dir) if args.output_dir else Path(dataset.model_path) / "viewer_exports"
        output_root.mkdir(parents=True, exist_ok=True)

        suffix = f"{args.highlight_mode}_{scene.loaded_iter}"
        recolored_path = output_root / f"recolored_gaussians_{suffix}.ply"
        recolored_tags_path = output_root / f"recolored_gaussians_{suffix}_tags.pt"

        highlight_rgb = parse_rgb(args.highlight_color)
        recolor_gaussians_in_place(
            gaussians,
            highlight_mask,
            highlight_rgb,
            highlight_opacity=args.highlight_opacity,
        )
        gaussians.save_ply(str(recolored_path))
        gaussians.save_tracking_metadata(str(recolored_tags_path))

        original_tags_src = Path(dataset.model_path) / "point_cloud" / f"iteration_{scene.loaded_iter}" / "gaussian_tags.pt"
        original_tags_copy = output_root / f"original_gaussian_tags_{suffix}.pt"
        if original_tags_src.exists():
            shutil.copy2(original_tags_src, original_tags_copy)

        summary = {
            "mode": "recolored_gaussian_ply",
            "model_path": str(Path(dataset.model_path).resolve()),
            "loaded_iteration": int(scene.loaded_iter),
            "highlight_mode": args.highlight_mode,
            "highlight_tag": int(args.highlight_tag),
            "highlight_mask_pt": None if not args.highlight_mask_pt else str(Path(args.highlight_mask_pt).resolve()),
            "highlight_mask_key": args.highlight_mask_key,
            "highlight_color_rgb": [float(x) for x in highlight_rgb.detach().cpu().tolist()],
            "highlight_opacity": float(args.highlight_opacity),
            "counts": {
                "total_gaussians": int(gaussians.get_xyz.shape[0]),
                "highlight_gaussians": highlight_count,
            },
            "paths": {
                "recolored_ply": str(recolored_path.resolve()),
                "recolored_tags": str(recolored_tags_path.resolve()),
                "original_tags_copy": str(original_tags_copy.resolve()) if original_tags_src.exists() else None,
            },
            "note": "This PLY keeps the original gaussian field format and only changes highlighted gaussians by overwriting their SH DC color, zeroing SH rest terms, and forcing their opacity near 1.",
        }
        summary_path = output_root / f"recolored_gaussians_{suffix}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"saved summary to: {summary_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export a gaussian PLY in the original format, recoloring selected gaussians.")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--highlight_mode",
        choices=["added", "probe", "prior", "tag", "mask_payload"],
        default="added",
        help="Which gaussians to recolor in the exported PLY.",
    )
    parser.add_argument("--highlight_tag", type=int, default=int(GaussianSourceTag.EXTENSION_PROBE))
    parser.add_argument("--highlight_mask_pt", type=str, default=None)
    parser.add_argument("--highlight_mask_key", type=str, default="prune_mask")
    parser.add_argument("--highlight_color", type=str, default="1.0,0.0,0.0")
    parser.add_argument("--highlight_opacity", type=float, default=0.999999)
    args = get_combined_args(parser)

    safe_state(args.quiet)
    export_recolored_cloud(model.extract(args), args.iteration, args)

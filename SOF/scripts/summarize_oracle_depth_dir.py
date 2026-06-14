from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a per-frame oracle depth directory for HRGSRefiner training.")
    parser.add_argument("--oracle_root", required=True, help="Oracle root directory. Supports root/*.npy, root/depth/*.npy, or root/train/ours_*/depth/*.npy")
    parser.add_argument("--scene_root", required=True, help="Scene root containing the image directory used for frame stems")
    parser.add_argument("--images_subdir", default="images_2", help="Image subdir used to enumerate expected frame names")
    parser.add_argument("--output_dir", default=None, help="Where to save summary json and preview image; defaults to oracle_root")
    parser.add_argument("--max_preview", type=int, default=9, help="Maximum number of frames to show in the preview grid")
    return parser.parse_args()


def _list_image_stems(images_dir: Path) -> List[str]:
    stems = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        stems.append(path.stem)
    return stems


def _resolve_depth_path(oracle_root: Path, stem: str) -> Path | None:
    direct = oracle_root / f"{stem}.npy"
    if direct.exists():
        return direct
    nested = oracle_root / "depth" / f"{stem}.npy"
    if nested.exists():
        return nested
    train_dirs = sorted((oracle_root / "train").glob("ours_*/depth"))
    for depth_dir in train_dirs:
        candidate = depth_dir / f"{stem}.npy"
        if candidate.exists():
            return candidate
    return None


def _load_depth(path: Path) -> np.ndarray:
    depth = np.load(str(path)).astype(np.float32)
    if depth.ndim != 2:
        depth = np.squeeze(depth)
    return depth


def _depth_to_preview(depth: np.ndarray, size: Tuple[int, int] = (320, 240)) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 1e-6)
    vis = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        values = depth[valid]
        lo = float(np.percentile(values, 5))
        hi = float(np.percentile(values, 95))
        vis = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = Image.fromarray((vis * 255.0).astype(np.uint8), mode="L").convert("RGB")
    return img.resize(size, Image.Resampling.BILINEAR)


def _make_preview_grid(items: List[Tuple[str, np.ndarray]], max_preview: int) -> Image.Image | None:
    if not items:
        return None
    tile_w, tile_h = 320, 240
    label_h = 28
    count = min(max_preview, len(items))
    ncols = 3
    nrows = int(np.ceil(count / ncols))
    canvas = Image.new("RGB", (ncols * tile_w, nrows * (tile_h + label_h)), color=(12, 12, 12))
    draw = ImageDraw.Draw(canvas)

    for idx, (stem, depth) in enumerate(items[:count]):
        r = idx // ncols
        c = idx % ncols
        x0 = c * tile_w
        y0 = r * (tile_h + label_h)
        preview = _depth_to_preview(depth, size=(tile_w, tile_h))
        canvas.paste(preview, (x0, y0))
        draw.text((x0 + 8, y0 + tile_h + 6), stem, fill=(230, 230, 230))
    return canvas


def main() -> None:
    args = parse_args()
    oracle_root = Path(args.oracle_root).expanduser().resolve()
    scene_root = Path(args.scene_root).expanduser().resolve()
    images_dir = scene_root / args.images_subdir
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else oracle_root
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = _list_image_stems(images_dir)
    if not stems:
        raise RuntimeError(f"No image files found under {images_dir}")

    found = []
    missing = []
    preview_items: List[Tuple[str, np.ndarray]] = []

    global_valid_values = []
    valid_ratios = []
    medians = []

    for stem in stems:
        depth_path = _resolve_depth_path(oracle_root, stem)
        if depth_path is None:
            missing.append(stem)
            continue
        depth = _load_depth(depth_path)
        valid = np.isfinite(depth) & (depth > 1e-6)
        valid_ratio = float(valid.mean())
        median = float(np.median(depth[valid])) if valid.any() else 0.0
        valid_ratios.append(valid_ratio)
        medians.append(median)
        if valid.any():
            global_valid_values.append(depth[valid])
        found.append(
            {
                "stem": stem,
                "path": str(depth_path),
                "shape": [int(depth.shape[0]), int(depth.shape[1])],
                "valid_ratio": valid_ratio,
                "median_depth": median,
                "min_depth": float(depth[valid].min()) if valid.any() else 0.0,
                "max_depth": float(depth[valid].max()) if valid.any() else 0.0,
            }
        )
        if len(preview_items) < int(args.max_preview):
            preview_items.append((stem, depth))

    global_values = np.concatenate(global_valid_values, axis=0) if global_valid_values else np.zeros((0,), dtype=np.float32)

    summary = {
        "scene_root": str(scene_root),
        "oracle_root": str(oracle_root),
        "images_subdir": args.images_subdir,
        "expected_frames": len(stems),
        "found_frames": len(found),
        "missing_frames": len(missing),
        "coverage_ratio": float(len(found) / max(len(stems), 1)),
        "mean_valid_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "median_valid_ratio": float(np.median(valid_ratios)) if valid_ratios else 0.0,
        "mean_median_depth": float(np.mean(medians)) if medians else 0.0,
        "median_median_depth": float(np.median(medians)) if medians else 0.0,
        "global_min_depth": float(global_values.min()) if global_values.size else 0.0,
        "global_max_depth": float(global_values.max()) if global_values.size else 0.0,
        "global_median_depth": float(np.median(global_values)) if global_values.size else 0.0,
        "missing_stems": missing,
    }

    meta = {
        "depth_convention": "camera_z_depth",
        "normal_space": "not_stored_depth_only; training derives normals from oracle depth",
        "resolution_reference": args.images_subdir,
        "camera_source": "same scene COLMAP sparse directory used by SOF runner/trainer",
        "valid_mask_rule": "valid if finite and depth > 1e-6",
        "oracle_source": "mip-splatting model trained on HR images and rendered with depth-as-color export",
    }

    (output_dir / "formal_oracle_v0_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "formal_oracle_v0_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (output_dir / "formal_oracle_v0_frames.json").write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")

    preview = _make_preview_grid(preview_items, max_preview=int(args.max_preview))
    if preview is not None:
        preview.save(output_dir / "formal_oracle_v0_preview.png")

    print(f"[oracle-summary] expected={len(stems)} found={len(found)} missing={len(missing)} coverage={summary['coverage_ratio']:.4f}")
    print(f"[oracle-summary] valid_ratio(mean/median)={summary['mean_valid_ratio']:.4f}/{summary['median_valid_ratio']:.4f}")
    print(f"[oracle-summary] depth median(global)={summary['global_median_depth']:.6f}")
    print(f"[oracle-summary] wrote summary to {output_dir}")


if __name__ == "__main__":
    main()

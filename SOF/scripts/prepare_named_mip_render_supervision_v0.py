from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_mip_to_sof_surface_v0 import load_cameras_for_split, normalize_image_name


def list_render_images(render_root: Path, recursive: bool = False) -> List[Path]:
    candidates = render_root.rglob("*") if recursive else render_root.iterdir()
    image_paths = [
        path
        for path in sorted(candidates)
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not image_paths:
        raise FileNotFoundError(f"No render images found under {render_root}")
    return image_paths


def select_uniform_indices(num_items: int, max_items: int) -> List[int]:
    if max_items <= 0 or num_items <= max_items:
        return list(range(num_items))
    ids = np.unique(np.linspace(0, num_items - 1, num=max_items, dtype=np.int64))
    return [int(idx) for idx in ids.tolist()]


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare camera-name-indexed mip render supervision directory.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--render_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--recursive_render_scan",
        action="store_true",
        help="Scan render_root recursively. The default expects the flat render.py output directory.",
    )
    parser.add_argument(
        "--allow_extra_renders",
        action="store_true",
        help="Allow extra render images and ignore sorted images beyond the camera count.",
    )
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    render_root = Path(args.render_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras_for_split(scene_root, model_path, str(args.images_subdir), str(args.split))
    selected_indices = select_uniform_indices(len(cameras), int(args.max_views))
    selected_cameras = [cameras[idx] for idx in selected_indices]
    render_paths = list_render_images(render_root, recursive=bool(args.recursive_render_scan))
    unused_render_paths: list[Path] = []
    if len(render_paths) == len(cameras):
        selected_render_paths = [render_paths[idx] for idx in selected_indices]
    elif len(render_paths) == len(selected_cameras):
        selected_render_paths = render_paths
    elif len(render_paths) > len(cameras) and bool(args.allow_extra_renders):
        usable_render_paths = render_paths[: len(cameras)]
        selected_render_paths = [usable_render_paths[idx] for idx in selected_indices]
        unused_render_paths = render_paths[len(cameras) :]
        print(
            "[prepare-named-mip-render-v0] extra render images ignored: "
            f"using {len(usable_render_paths)}/{len(render_paths)} from {render_root}",
            file=sys.stderr,
        )
    else:
        raise RuntimeError(
            f"Render count mismatch under {render_root}: got {len(render_paths)} images "
            f"for {len(cameras)} cameras ({len(selected_cameras)} selected)."
        )

    manifest = []
    for camera, render_path in zip(selected_cameras, selected_render_paths):
        stem = normalize_image_name(str(camera.image_name))
        dst = output_dir / f"{stem}{render_path.suffix.lower()}"
        safe_symlink(render_path, dst)
        manifest.append(
            {
                "camera_image_name": str(camera.image_name),
                "stem": stem,
                "render_path": str(render_path),
                "linked_path": str(dst),
            }
        )

    summary = {
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "camera_count": int(len(cameras)),
        "selected_count": int(len(selected_cameras)),
        "render_count": int(len(render_paths)),
        "used_render_count": int(len(selected_render_paths)),
        "unused_render_count": int(len(unused_render_paths)),
        "unused_render_examples": [str(path) for path in unused_render_paths[:20]],
        "recursive_render_scan": bool(args.recursive_render_scan),
        "allow_extra_renders": bool(args.allow_extra_renders),
        "render_root": str(render_root),
        "output_dir": str(output_dir),
        "manifest": manifest,
    }
    with open(output_dir / "prepare_named_mip_render_supervision_v0_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from utils.hrgs_scene_layout import (
    build_hrgs_manifest_payload,
    build_hrgs_view_records,
    resolve_hrgs_scene_layout,
    save_hrgs_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an HRGSRefiner scene manifest for images_8 -> images_2 style projects."
    )
    parser.add_argument("--scene_root", required=True, help="Scene root containing sparse/, images_8, images_2, and priors/.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    parser.add_argument("--source_images_subdir", default="images_8")
    parser.add_argument("--target_images_subdir", default="images_2")
    parser.add_argument("--priors_dir", default=None, help="Default: <scene_root>/priors")
    parser.add_argument("--vggt_root", default="/root/autodl-tmp/vggt")
    parser.add_argument("--repo_root", default="/root/autodl-tmp/SRtestrepo")
    parser.add_argument("--require_priors", action="store_true", help="Drop views that do not have matched priors.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of matched views.")
    return parser.parse_args()


def main():
    args = parse_args()
    layout = resolve_hrgs_scene_layout(
        args.scene_root,
        source_images_subdir=args.source_images_subdir,
        target_images_subdir=args.target_images_subdir,
        priors_dir=args.priors_dir,
        vggt_root=args.vggt_root,
        repo_root=args.repo_root,
    )
    records = build_hrgs_view_records(
        layout,
        require_priors=bool(args.require_priors),
        limit=int(args.limit),
    )
    payload = build_hrgs_manifest_payload(layout, records)
    manifest_path = save_hrgs_manifest(args.output, payload)

    counts = payload["counts"]
    print(f"[hrgs-manifest] scene_root          : {layout.scene_root}")
    print(f"[hrgs-manifest] source_image_dir   : {layout.source_image_dir}")
    print(f"[hrgs-manifest] target_image_dir   : {layout.target_image_dir}")
    print(f"[hrgs-manifest] priors_dir         : {layout.priors_dir or '(missing)'}")
    print(f"[hrgs-manifest] vggt_root          : {layout.vggt_root or '(unset)'}")
    print(f"[hrgs-manifest] repo_root          : {layout.repo_root or '(unset)'}")
    print(f"[hrgs-manifest] num_views          : {counts['num_views']}")
    print(f"[hrgs-manifest] num_views_with_prior: {counts['num_views_with_prior']}")
    print(f"[hrgs-manifest] manifest           : {manifest_path}")


if __name__ == "__main__":
    main()

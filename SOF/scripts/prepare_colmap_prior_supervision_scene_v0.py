#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_DIR_NAMES = {"images"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a COLMAP scene alias whose sparse assets come from an existing scene "
            "while training images are replaced by prepared SR priors. Missing prior "
            "views can be filled from a reference image directory so COLMAP loaders can "
            "still read held-out cameras under --eval."
        )
    )
    parser.add_argument("--scene_root", type=Path, required=True)
    parser.add_argument("--scene_alias_dir", type=Path, required=True)
    parser.add_argument("--prior_dir", type=Path, required=True)
    parser.add_argument("--reference_images_subdir", type=str, default="images_2")
    parser.add_argument("--fallback_images_subdir", type=str, default="images_2")
    parser.add_argument("--output_images_subdir", type=str, default="images")
    parser.add_argument(
        "--missing_policy",
        type=str,
        default="fallback",
        choices=["fallback", "error"],
        help="What to do when a COLMAP/reference image has no matching prior.",
    )
    parser.add_argument(
        "--link_mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="How to populate alias images.",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def collect_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if is_image(path))


def symlink_or_copy(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def index_by_stem(paths: list[Path], label: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in paths:
        key = path.stem
        if key in index:
            duplicates.setdefault(key, [str(index[key])]).append(str(path))
            continue
        index[key] = path
    if duplicates:
        examples = {key: vals[:4] for key, vals in list(duplicates.items())[:8]}
        raise ValueError(f"Duplicate {label} stems are ambiguous: {examples}")
    return index


def copy_scene_metadata(scene_root: Path, scene_alias_dir: Path, output_images_subdir: str) -> list[str]:
    copied: list[str] = []
    for entry in sorted(scene_root.iterdir()):
        if entry.name == output_images_subdir or entry.name in IMAGE_DIR_NAMES or entry.name.startswith("images_"):
            continue
        dst = scene_alias_dir / entry.name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.symlink_to(entry.resolve())
        copied.append(entry.name)
    return copied


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.expanduser().resolve()
    scene_alias_dir = args.scene_alias_dir.expanduser().resolve()
    prior_dir = args.prior_dir.expanduser().resolve()
    reference_dir = (scene_root / args.reference_images_subdir).resolve()
    fallback_dir = (scene_root / args.fallback_images_subdir).resolve()
    output_images_dir = scene_alias_dir / args.output_images_subdir

    for label, path in {
        "scene_root": scene_root,
        "prior_dir": prior_dir,
        "reference_dir": reference_dir,
        "fallback_dir": fallback_dir,
        "sparse/0": scene_root / "sparse" / "0",
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    prior_paths = collect_images(prior_dir)
    reference_paths = collect_images(reference_dir)
    fallback_paths = collect_images(fallback_dir)
    if not prior_paths:
        raise FileNotFoundError(f"No prior images found under: {prior_dir}")
    if not reference_paths:
        raise FileNotFoundError(f"No reference images found under: {reference_dir}")

    prior_by_stem = index_by_stem(prior_paths, "prior")
    fallback_by_name = {path.name: path for path in fallback_paths}
    fallback_by_stem = index_by_stem(fallback_paths, "fallback")

    scene_alias_dir.mkdir(parents=True, exist_ok=True)
    if output_images_dir.exists() and not output_images_dir.is_symlink():
        shutil.rmtree(output_images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    copied_metadata = copy_scene_metadata(
        scene_root=scene_root,
        scene_alias_dir=scene_alias_dir,
        output_images_subdir=args.output_images_subdir,
    )

    entries: list[dict[str, object]] = []
    prior_count = 0
    fallback_count = 0
    missing: list[str] = []

    for ref in reference_paths:
        rel = ref.relative_to(reference_dir)
        dst = output_images_dir / rel
        prior = prior_by_stem.get(ref.stem)
        source_kind = "prior"
        src = prior
        if src is None:
            source_kind = "fallback"
            src = fallback_by_name.get(ref.name) or fallback_by_stem.get(ref.stem)
        if src is None:
            missing.append(str(rel))
            continue
        if source_kind == "fallback" and args.missing_policy == "error":
            missing.append(str(rel))
            continue
        symlink_or_copy(src, dst, args.link_mode)
        prior_count += int(source_kind == "prior")
        fallback_count += int(source_kind == "fallback")
        entries.append(
            {
                "relative_path": str(rel),
                "output_image": str(dst),
                "source_kind": source_kind,
                "source_image": str(src.resolve()),
                "reference_image": str(ref.resolve()),
            }
        )

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} reference images have no matching prior/fallback. "
            f"Examples: {missing[:12]}"
        )
    if prior_count <= 0:
        raise RuntimeError("No priors were matched into the alias scene.")

    summary = {
        "version": "prepare_colmap_prior_supervision_scene_v0",
        "scene_root": str(scene_root),
        "scene_alias_dir": str(scene_alias_dir),
        "prior_dir": str(prior_dir),
        "reference_images_subdir": args.reference_images_subdir,
        "fallback_images_subdir": args.fallback_images_subdir,
        "output_images_subdir": args.output_images_subdir,
        "link_mode": args.link_mode,
        "missing_policy": args.missing_policy,
        "copied_metadata": copied_metadata,
        "num_reference_images": len(reference_paths),
        "num_prior_images_available": len(prior_paths),
        "num_prior_supervised_images": prior_count,
        "num_fallback_images": fallback_count,
        "entries": entries,
        "notes": [
            "Fallback images are only meant to make held-out COLMAP cameras load under --eval.",
            "Training supervision is prior-only for views whose stems exist in prior_dir.",
        ],
    }
    summary_path = scene_alias_dir / "prior_supervision_scene_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[prepare-colmap-prior-supervision-v0] done")
    print(f"  scene_root        : {scene_root}")
    print(f"  alias             : {scene_alias_dir}")
    print(f"  prior_dir         : {prior_dir}")
    print(f"  output_images     : {output_images_dir}")
    print(f"  prior supervised  : {prior_count}/{len(reference_paths)}")
    print(f"  fallback images   : {fallback_count}/{len(reference_paths)}")
    print(f"  summary           : {summary_path}")


if __name__ == "__main__":
    main()

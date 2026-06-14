#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
ARTIFACT_DIR_NAMES = {".ipynb_checkpoints", "__macosx"}


def _is_notebook_or_checkpoint_artifact(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts
    for part in relative_parts:
        part_lower = part.lower()
        if part.startswith("."):
            return True
        if part_lower in ARTIFACT_DIR_NAMES or "ipynb_checkpoints" in part_lower:
            return True
    if path.name.startswith("."):
        return True
    stem_lower = path.stem.lower()
    name_lower = path.name.lower()
    return "-checkpoint" in stem_lower or "-checkpoint." in name_lower


@dataclass
class AxisMapping:
    source_len: int
    target_len: int
    source_start: int
    target_start: int
    copy_len: int
    delta: int


@dataclass
class FrameSummary:
    image_name: str
    stem: str
    prior_path: str
    reference_path: Optional[str]
    output_path: str
    mask_path: str
    prior_width: int
    prior_height: int
    target_width: int
    target_height: int
    delta_width: int
    delta_height: int
    source_x0: int
    source_y0: int
    target_x0: int
    target_y0: int
    copy_width: int
    copy_height: int
    valid_fraction: float
    fill_mode_used: str
    action: str


def _list_image_paths(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths: List[Path] = []
    skipped_artifacts: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if _is_notebook_or_checkpoint_artifact(path, root):
            skipped_artifacts.append(path)
            continue
        paths.append(path)
    if skipped_artifacts:
        examples = ", ".join(str(path.relative_to(root)) for path in skipped_artifacts[:4])
        print(
            "[align-vosr-prior-v0] ignored notebook/checkpoint image artifacts: "
            f"{len(skipped_artifacts)} under {root}"
            + (f" ({examples})" if examples else ""),
            flush=True,
        )
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


def _candidate_keys(path: Path) -> List[str]:
    stem = path.stem
    name = path.name
    keys = [
        name,
        stem,
        name.lower(),
        stem.lower(),
    ]
    if stem.isdigit():
        value = int(stem)
        keys.extend(
            [
                str(value),
                f"{value:04d}",
                f"{value:05d}",
                f"{value:06d}",
                f"{value:04d}{path.suffix.lower()}",
                f"{value:05d}{path.suffix.lower()}",
                f"{value:06d}{path.suffix.lower()}",
            ]
        )
    seen: set[str] = set()
    ordered: List[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _build_lookup(paths: Iterable[Path], role: str) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in paths:
        for key in _candidate_keys(path):
            existing = lookup.get(key)
            if existing is not None and existing != path:
                raise ValueError(
                    f"Duplicate {role} lookup key '{key}' for {existing} and {path}. "
                    "Pass a more specific directory, or disable ambiguous naming upstream."
                )
            lookup[key] = path
    return lookup


def _import_colmap_reader():
    repo_root = Path(__file__).resolve().parents[2]
    mip_root = repo_root / "mip-splatting"
    if not mip_root.is_dir():
        raise FileNotFoundError(
            f"mip-splatting repo not found next to SOF: {mip_root}. "
            "Set --reference_split all, or place the repo at the expected sibling path."
        )
    mip_root_str = str(mip_root)
    if mip_root_str not in sys.path:
        sys.path.insert(0, mip_root_str)
    from scene.dataset_readers import readColmapSceneInfo  # type: ignore

    return readColmapSceneInfo


def _resolve_reference_path(reference_lookup: Dict[str, Path], image_name: str, reference_dir: Path) -> Path:
    probe = Path(image_name)
    for key in _candidate_keys(probe):
        matched = reference_lookup.get(key)
        if matched is not None:
            return matched
    raise FileNotFoundError(
        f"Could not resolve camera '{image_name}' under reference_dir={reference_dir}. "
        "Expected a file whose stem or name matches the COLMAP camera name."
    )


def _load_scene_split_reference_paths(
    *,
    scene_root: Path,
    images_subdir: str,
    reference_dir: Path,
    reference_split: str,
    llffhold: int,
) -> Tuple[List[Path], Dict[str, int]]:
    read_colmap_scene_info = _import_colmap_reader()
    scene_info = read_colmap_scene_info(str(scene_root), images=images_subdir, eval=True, llffhold=llffhold)
    reference_lookup = _build_lookup(_list_image_paths(reference_dir), "reference")

    train_names = [str(camera.image_name) for camera in scene_info.train_cameras]
    test_names = [str(camera.image_name) for camera in scene_info.test_cameras]
    counts = {
        "train": int(len(train_names)),
        "test": int(len(test_names)),
        "all": int(len(train_names) + len(test_names)),
    }

    if reference_split == "train":
        image_names = train_names
    elif reference_split == "test":
        image_names = test_names
    elif reference_split == "all":
        image_names = train_names + test_names
    else:
        raise ValueError(f"Unsupported reference_split: {reference_split}")

    ordered_paths = [
        _resolve_reference_path(reference_lookup, image_name, reference_dir)
        for image_name in image_names
    ]
    return ordered_paths, counts


def _resolve_match(lookup: Dict[str, Path], reference_path: Path) -> Optional[Path]:
    for key in _candidate_keys(reference_path):
        matched = lookup.get(key)
        if matched is not None:
            return matched
    return None


def _resolve_match_with_mode(
    *,
    prior_paths: List[Path],
    prior_lookup: Dict[str, Path],
    reference_path: Path,
    index: int,
    match_mode: str,
) -> Optional[Path]:
    match_mode = str(match_mode).strip().lower()
    if match_mode == "order":
        if 0 <= int(index) < len(prior_paths):
            return prior_paths[int(index)]
        return None
    matched = _resolve_match(prior_lookup, reference_path)
    if matched is not None:
        return matched
    if match_mode == "order_fallback" and 0 <= int(index) < len(prior_paths):
        return prior_paths[int(index)]
    return None


def _count_name_matches(prior_lookup: Dict[str, Path], reference_paths: Iterable[Path]) -> int:
    count = 0
    for reference_path in reference_paths:
        if _resolve_match(prior_lookup, reference_path) is not None:
            count += 1
    return count


def _choose_match_mode(
    *,
    requested_mode: str,
    prior_paths: List[Path],
    prior_lookup: Dict[str, Path],
    reference_paths: List[Path],
) -> Tuple[str, int]:
    requested_mode = str(requested_mode).strip().lower()
    if requested_mode != "auto":
        name_match_count = (
            _count_name_matches(prior_lookup, reference_paths)
            if requested_mode in {"name", "order_fallback"}
            else 0
        )
        return requested_mode, int(name_match_count)

    name_match_count = _count_name_matches(prior_lookup, reference_paths)
    total = len(reference_paths)
    if total == 0 or name_match_count == total:
        return "name", int(name_match_count)
    if len(prior_paths) == total:
        if name_match_count == 0:
            print(
                "[align-vosr-prior-v0] no filename matches found; "
                "falling back to sorted-order pairing for all frames.",
                flush=True,
            )
            return "order", 0
        print(
            "[align-vosr-prior-v0] partial filename matches found; "
            f"using order_fallback to keep {name_match_count}/{total} direct matches.",
            flush=True,
        )
        return "order_fallback", int(name_match_count)
    raise FileNotFoundError(
        "Auto match could not align priors to references: "
        f"{name_match_count}/{total} filename matches, but prior/reference counts differ "
        f"({len(prior_paths)} vs {total}). Pass --match_mode order explicitly if sorted pairing is intended."
    )


def _find_path_index(paths: List[Path], needle: str, role: str) -> int:
    needle = str(needle).strip()
    if not needle:
        raise ValueError(f"Empty {role} resume name is not allowed.")
    normalized = needle.lower()
    matches = [
        idx for idx, path in enumerate(paths)
        if path.name.lower() == normalized or path.stem.lower() == normalized
    ]
    if not matches:
        raise FileNotFoundError(f"Could not find {role} resume name '{needle}' in sorted image list.")
    if len(matches) > 1:
        raise ValueError(f"Resume name '{needle}' is ambiguous in {role} image list.")
    return int(matches[0])


def _offset(max_offset: int, anchor: str, explicit: Optional[int]) -> int:
    if max_offset <= 0:
        return 0
    if explicit is not None:
        if explicit < 0 or explicit > max_offset:
            raise ValueError(f"Explicit offset {explicit} is outside [0, {max_offset}]")
        return int(explicit)
    if anchor == "center":
        return max_offset // 2
    if anchor in {"top_left", "left", "top"}:
        return 0
    if anchor in {"bottom_right", "right", "bottom"}:
        return max_offset
    raise ValueError(f"Unsupported anchor: {anchor}")


def _axis_mapping(
    source_len: int,
    target_len: int,
    *,
    anchor: str,
    explicit_offset: Optional[int],
) -> AxisMapping:
    source_len = int(source_len)
    target_len = int(target_len)
    if source_len <= 0 or target_len <= 0:
        raise ValueError(f"Invalid size pair source={source_len}, target={target_len}")

    if source_len <= target_len:
        copy_len = source_len
        source_start = 0
        target_start = _offset(target_len - source_len, anchor, explicit_offset)
    else:
        copy_len = target_len
        source_start = _offset(source_len - target_len, anchor, explicit_offset)
        target_start = 0

    return AxisMapping(
        source_len=source_len,
        target_len=target_len,
        source_start=int(source_start),
        target_start=int(target_start),
        copy_len=int(copy_len),
        delta=int(target_len - source_len),
    )


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def _edge_extend_canvas(cropped: Image.Image, target_size: Tuple[int, int], target_xy: Tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    x0, y0 = target_xy
    copy_w, copy_h = cropped.size
    if copy_w <= 0 or copy_h <= 0:
        return Image.new("RGB", target_size, (0, 0, 0))

    canvas = Image.new("RGB", target_size, (0, 0, 0))
    canvas.paste(cropped, (x0, y0))
    arr = np.asarray(canvas, dtype=np.uint8).copy()

    valid_x0 = int(x0)
    valid_y0 = int(y0)
    valid_x1 = int(x0 + copy_w)
    valid_y1 = int(y0 + copy_h)
    yy = np.clip(np.arange(target_h), valid_y0, valid_y1 - 1)
    xx = np.clip(np.arange(target_w), valid_x0, valid_x1 - 1)
    extended = arr[yy[:, None], xx[None, :]]
    return Image.fromarray(extended, mode="RGB")


def _align_one(
    *,
    prior: Image.Image,
    target_size: Tuple[int, int],
    reference: Optional[Image.Image],
    anchor: str,
    fill_mode: str,
    x_offset: Optional[int],
    y_offset: Optional[int],
) -> Tuple[Image.Image, Image.Image, AxisMapping, AxisMapping, str]:
    target_w, target_h = map(int, target_size)
    prior_w, prior_h = prior.size
    xmap = _axis_mapping(prior_w, target_w, anchor=anchor, explicit_offset=x_offset)
    ymap = _axis_mapping(prior_h, target_h, anchor=anchor, explicit_offset=y_offset)

    crop_box = (
        xmap.source_start,
        ymap.source_start,
        xmap.source_start + xmap.copy_len,
        ymap.source_start + ymap.copy_len,
    )
    cropped = prior.crop(crop_box)

    fill_mode_used = fill_mode
    if fill_mode == "reference" and reference is not None:
        if reference.size != target_size:
            raise ValueError(f"Reference size {reference.size} does not match target size {target_size}")
        canvas = reference.copy()
    elif fill_mode == "reference":
        fill_mode_used = "edge"
        canvas = _edge_extend_canvas(cropped, target_size, (xmap.target_start, ymap.target_start))
    elif fill_mode == "edge":
        canvas = _edge_extend_canvas(cropped, target_size, (xmap.target_start, ymap.target_start))
    elif fill_mode == "black":
        canvas = Image.new("RGB", target_size, (0, 0, 0))
    else:
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")

    canvas.paste(cropped, (xmap.target_start, ymap.target_start))
    mask = Image.new("L", target_size, 0)
    mask.paste(
        Image.new("L", (xmap.copy_len, ymap.copy_len), 255),
        (xmap.target_start, ymap.target_start),
    )
    return canvas, mask, xmap, ymap, fill_mode_used


def _stats(values: List[int]) -> Dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "median": float(np.median(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Restore SR-prior pixel correspondence when VOSR inputs were pre-cropped. "
            "The VOSR training/test dataset path uses center crop; use --anchor top_left "
            "for external modcrop-style pipelines."
        )
    )
    parser.add_argument("--prior_dir", required=True, help="Directory containing raw VOSR/SR prior images.")
    parser.add_argument("--output_root", required=True, help="Prepared prior root to write.")
    parser.add_argument("--reference_dir", default="", help="Target image directory whose size/stems define the output grid.")
    parser.add_argument("--scene_root", default="", help="Scene root with sparse/0 for split-aware reference ordering.")
    parser.add_argument("--images_subdir", default="", help="Image subdir under scene_root used by COLMAP split ordering.")
    parser.add_argument(
        "--reference_split",
        choices=["auto", "all", "train", "test"],
        default="all",
        help=(
            "Which COLMAP split ordering to use when scene_root is available. "
            "'auto' chooses the split whose count matches the prior count."
        ),
    )
    parser.add_argument("--llffhold", type=int, default=8, help="COLMAP eval split stride.")
    parser.add_argument("--target_width", type=int, default=0, help="Fallback target width when --reference_dir is not used.")
    parser.add_argument("--target_height", type=int, default=0, help="Fallback target height when --reference_dir is not used.")
    parser.add_argument(
        "--anchor",
        choices=["center", "top_left", "bottom_right"],
        default="center",
        help="How the cropped prior maps back into the target frame. VOSR dataset test crop is center.",
    )
    parser.add_argument(
        "--fill_mode",
        choices=["reference", "edge", "black"],
        default="reference",
        help="Pixels not covered by the prior are filled this way and marked invalid in usable_masks.",
    )
    parser.add_argument("--x_offset", type=int, default=None, help="Explicit x offset for padding/cropping; overrides --anchor on x.")
    parser.add_argument("--y_offset", type=int, default=None, help="Explicit y offset for padding/cropping; overrides --anchor on y.")
    parser.add_argument("--max_delta", type=int, default=64, help="Reject size deltas larger than this unless --allow_large_delta.")
    parser.add_argument("--allow_large_delta", action="store_true", help="Allow large crop/pad deltas.")
    parser.add_argument("--allow_missing", action="store_true", help="Skip missing prior matches instead of failing.")
    parser.add_argument(
        "--start_prior_name",
        default="",
        help=(
            "Resume from this prior image name/stem in the sorted prior list. "
            "With order-based matching, the same sorted index is used for references."
        ),
    )
    parser.add_argument(
        "--start_reference_name",
        default="",
        help="Resume from this reference image name/stem in the sorted reference list.",
    )
    parser.add_argument(
        "--match_mode",
        choices=["auto", "name", "order_fallback", "order"],
        default="auto",
        help=(
            "How reference frames map to prior frames. "
            "'auto' prefers filename/stem matching and falls back to sorted order when needed; "
            "'name' matches by filename/stem only; "
            "'order_fallback' uses sorted order when name matching fails; "
            "'order' always pairs by sorted order."
        ),
    )
    parser.add_argument("--order_fallback", action="store_true", help="Match by sorted order if names do not match.")
    parser.add_argument("--dry_run", action="store_true", help="Only write summary JSON; do not write aligned images.")
    parser.add_argument("--progress_every", type=int, default=50)
    args = parser.parse_args()

    prior_dir = Path(args.prior_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    reference_dir = Path(args.reference_dir).expanduser().resolve() if str(args.reference_dir).strip() else None
    scene_root = Path(args.scene_root).expanduser().resolve() if str(args.scene_root).strip() else None
    images_subdir = str(args.images_subdir).strip()
    requested_reference_split = str(args.reference_split).strip().lower()
    requested_match_mode = str(args.match_mode)
    if bool(args.order_fallback) and requested_match_mode in {"auto", "name"}:
        requested_match_mode = "order_fallback"

    prior_paths = _list_image_paths(prior_dir)
    prior_lookup = _build_lookup(prior_paths, "prior")

    if reference_dir is not None:
        selected_reference_split = "all"
        split_counts: Dict[str, int] | None = None
        if scene_root is not None and images_subdir:
            if requested_reference_split == "auto":
                _, split_counts = _load_scene_split_reference_paths(
                    scene_root=scene_root,
                    images_subdir=images_subdir,
                    reference_dir=reference_dir,
                    reference_split="all",
                    llffhold=int(args.llffhold),
                )
                candidate_splits = [
                    split_name
                    for split_name in ("train", "test", "all")
                    if int(split_counts.get(split_name, -1)) == len(prior_paths)
                ]
                if len(candidate_splits) == 1:
                    selected_reference_split = candidate_splits[0]
                    print(
                        "[align-vosr-prior-v0] auto-selected reference split: "
                        f"{selected_reference_split} ({len(prior_paths)} priors)",
                        flush=True,
                    )
                elif len(candidate_splits) > 1:
                    raise ValueError(
                        f"reference_split=auto is ambiguous for prior_count={len(prior_paths)}: "
                        f"{candidate_splits}"
                    )
                else:
                    selected_reference_split = "all"
            else:
                selected_reference_split = requested_reference_split

            reference_paths, counts_from_split = _load_scene_split_reference_paths(
                scene_root=scene_root,
                images_subdir=images_subdir,
                reference_dir=reference_dir,
                reference_split=selected_reference_split,
                llffhold=int(args.llffhold),
            )
            if split_counts is None:
                split_counts = counts_from_split
        else:
            if requested_reference_split != "all":
                raise ValueError(
                    "--reference_split requires both --scene_root and --images_subdir."
                )
            reference_paths = _list_image_paths(reference_dir)
            split_counts = None

        match_mode, name_match_count = _choose_match_mode(
            requested_mode=requested_match_mode,
            prior_paths=prior_paths,
            prior_lookup=prior_lookup,
            reference_paths=reference_paths,
        )
    else:
        if int(args.target_width) <= 0 or int(args.target_height) <= 0:
            raise ValueError("Either --reference_dir or both --target_width/--target_height must be provided.")
        reference_paths = []
        match_mode = requested_match_mode
        name_match_count = 0
        selected_reference_split = "all"
        split_counts = None

    start_prior_name = str(args.start_prior_name).strip()
    start_reference_name = str(args.start_reference_name).strip()
    resume_prior_index = _find_path_index(prior_paths, start_prior_name, "prior") if start_prior_name else None
    resume_reference_index = (
        _find_path_index(reference_paths, start_reference_name, "reference")
        if start_reference_name and reference_paths
        else None
    )

    if reference_paths and match_mode in {"order", "order_fallback"}:
        resume_index = None
        if resume_prior_index is not None and resume_reference_index is not None and resume_prior_index != resume_reference_index:
            raise ValueError(
                "When using order-based matching, start_prior_name and start_reference_name "
                f"must resolve to the same sorted index; got prior={resume_prior_index}, reference={resume_reference_index}."
            )
        if resume_prior_index is not None:
            resume_index = resume_prior_index
        elif resume_reference_index is not None:
            resume_index = resume_reference_index
        if resume_index is not None:
            prior_paths = prior_paths[resume_index:]
            reference_paths = reference_paths[resume_index:]
            prior_lookup = _build_lookup(prior_paths, "prior")
    else:
        if resume_prior_index is not None:
            if reference_paths:
                raise ValueError(
                    "--start_prior_name with --reference_dir requires --match_mode order or order_fallback."
                )
            prior_paths = prior_paths[resume_prior_index:]
            prior_lookup = _build_lookup(prior_paths, "prior")
        if resume_reference_index is not None:
            reference_paths = reference_paths[resume_reference_index:]

    fused_dir = output_root / "fused_priors"
    mask_dir = output_root / "usable_masks"
    reference_out_dir = output_root / "aligned_references"
    if not args.dry_run:
        fused_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        if reference_dir is not None:
            reference_out_dir.mkdir(parents=True, exist_ok=True)

    frames: List[FrameSummary] = []
    missing: List[str] = []
    unreadable_priors: List[Dict[str, object]] = []
    unreadable_references: List[Dict[str, object]] = []
    skipped_large_delta: List[Dict[str, object]] = []
    delta_w_values: List[int] = []
    delta_h_values: List[int] = []
    valid_fractions: List[float] = []
    order_fallback_count = 0
    start_time = time.time()

    if reference_paths:
        work_items = list(enumerate(reference_paths))
    else:
        work_items = list(enumerate(prior_paths))

    for idx, ref_or_prior_path in work_items:
        if reference_paths:
            reference_path = ref_or_prior_path
            direct_name_match = _resolve_match(prior_lookup, reference_path)
            prior_path = _resolve_match_with_mode(
                prior_paths=prior_paths,
                prior_lookup=prior_lookup,
                reference_path=reference_path,
                index=idx,
                match_mode=match_mode,
            )
            if prior_path is None:
                missing.append(str(reference_path))
                if args.allow_missing:
                    continue
                raise FileNotFoundError(f"No prior image matched reference image: {reference_path}")
            if match_mode == "order_fallback" and direct_name_match is None:
                order_fallback_count += 1
            try:
                reference = _load_rgb(reference_path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                unreadable = {
                    "reference_path": str(reference_path),
                    "prior_path": str(prior_path) if prior_path is not None else None,
                    "error": repr(exc),
                }
                unreadable_references.append(unreadable)
                if args.allow_missing:
                    print(
                        f"[align-vosr-prior-v0] skip unreadable reference image: {reference_path} ({exc})",
                        flush=True,
                    )
                    continue
                raise RuntimeError(f"Failed to read reference image: {reference_path}") from exc
            target_size = reference.size
            out_stem = reference_path.stem
            image_name = reference_path.name
        else:
            prior_path = ref_or_prior_path
            reference_path = None
            reference = None
            target_size = (int(args.target_width), int(args.target_height))
            out_stem = prior_path.stem
            image_name = prior_path.name

        try:
            prior = _load_rgb(prior_path)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            unreadable = {
                "image_name": image_name,
                "prior_path": str(prior_path),
                "reference_path": str(reference_path) if reference_path is not None else None,
                "error": repr(exc),
            }
            unreadable_priors.append(unreadable)
            if args.allow_missing:
                print(
                    f"[align-vosr-prior-v0] skip unreadable prior image: {prior_path} ({exc})",
                    flush=True,
                )
                continue
            raise RuntimeError(f"Failed to read prior image: {prior_path}") from exc
        prior_w, prior_h = prior.size
        target_w, target_h = target_size
        delta_w = int(target_w - prior_w)
        delta_h = int(target_h - prior_h)
        if not args.allow_large_delta and max(abs(delta_w), abs(delta_h)) > int(args.max_delta):
            skipped = {
                "image_name": image_name,
                "prior_path": str(prior_path),
                "reference_path": str(reference_path) if reference_path is not None else None,
                "prior_size": [int(prior_w), int(prior_h)],
                "target_size": [int(target_w), int(target_h)],
                "delta": [int(delta_w), int(delta_h)],
            }
            skipped_large_delta.append(skipped)
            raise ValueError(
                f"Size delta too large for {image_name}: prior={prior.size}, target={target_size}, "
                f"delta=({delta_w},{delta_h}). Use --allow_large_delta only if this is intentional."
            )

        aligned, mask, xmap, ymap, fill_mode_used = _align_one(
            prior=prior,
            target_size=target_size,
            reference=reference,
            anchor=str(args.anchor),
            fill_mode=str(args.fill_mode),
            x_offset=args.x_offset,
            y_offset=args.y_offset,
        )

        out_name = f"{out_stem}.png"
        output_path = fused_dir / out_name
        mask_path = mask_dir / out_name
        reference_out_path = reference_out_dir / out_name
        if not args.dry_run:
            aligned.save(output_path)
            mask.save(mask_path)
            if reference_path is not None:
                if reference_path.suffix.lower() == ".png":
                    shutil.copyfile(reference_path, reference_out_path)
                else:
                    reference.save(reference_out_path)

        valid_fraction = float((xmap.copy_len * ymap.copy_len) / max(target_w * target_h, 1))
        action_bits: List[str] = []
        if prior_w < target_w or prior_h < target_h:
            action_bits.append("pad")
        if prior_w > target_w or prior_h > target_h:
            action_bits.append("crop")
        action = "+".join(action_bits) if action_bits else "copy"

        frame = FrameSummary(
            image_name=out_name,
            stem=out_stem,
            prior_path=str(prior_path),
            reference_path=str(reference_path) if reference_path is not None else None,
            output_path=str(output_path),
            mask_path=str(mask_path),
            prior_width=int(prior_w),
            prior_height=int(prior_h),
            target_width=int(target_w),
            target_height=int(target_h),
            delta_width=int(delta_w),
            delta_height=int(delta_h),
            source_x0=int(xmap.source_start),
            source_y0=int(ymap.source_start),
            target_x0=int(xmap.target_start),
            target_y0=int(ymap.target_start),
            copy_width=int(xmap.copy_len),
            copy_height=int(ymap.copy_len),
            valid_fraction=valid_fraction,
            fill_mode_used=fill_mode_used,
            action=action,
        )
        frames.append(frame)
        delta_w_values.append(delta_w)
        delta_h_values.append(delta_h)
        valid_fractions.append(valid_fraction)

        progress_every = max(int(args.progress_every), 1)
        if len(frames) == 1 or len(frames) % progress_every == 0 or idx == len(work_items) - 1:
            elapsed = time.time() - start_time
            print(
                f"[align-vosr-prior-v0] processed {len(frames)}/{len(work_items)} "
                f"last={image_name} prior={prior_w}x{prior_h} target={target_w}x{target_h} "
                f"action={action} valid={valid_fraction:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if not frames:
        raise RuntimeError("No aligned priors were produced.")

    usable_mean = float(np.mean(valid_fractions)) if valid_fractions else None
    manifest = {
        "version": "align_vosr_prior_size_v0",
        "notes": (
            "VOSR dataset test preprocessing uses a center crop; official inference has no explicit crop. "
            "This preprocessor restores pixel correspondence by crop/pad on the selected anchor and marks "
            "non-prior-filled pixels invalid in usable_masks."
        ),
        "prior_dir": str(prior_dir),
        "reference_dir": str(reference_dir) if reference_dir is not None else None,
        "output_root": str(output_root),
        "fused_priors": str(fused_dir),
        "aligned_references": str(reference_out_dir) if reference_dir is not None else None,
        "usable_masks": str(mask_dir),
        "anchor": str(args.anchor),
        "fill_mode": str(args.fill_mode),
        "x_offset": args.x_offset,
        "y_offset": args.y_offset,
        "max_delta": int(args.max_delta),
        "allow_large_delta": bool(args.allow_large_delta),
        "match_mode": match_mode,
        "reference_split": selected_reference_split,
        "split_counts": split_counts,
        "requested_match_mode": requested_match_mode,
        "name_match_count": int(name_match_count),
        "order_fallback_count": int(order_fallback_count),
        "start_prior_name": start_prior_name or None,
        "start_reference_name": start_reference_name or None,
        "mask_mode": "valid_prior_pixels",
        "mask_threshold": 0.5,
        "save_fused_priors": True,
        "copy_raw_priors": False,
        "num_priors": int(len(prior_paths)),
        "num_reference": int(len(reference_paths)) if reference_paths else None,
        "num_matched": int(len(frames)),
        "missing_reference_count": int(len(missing)),
        "missing_reference_examples": missing[:16],
        "unreadable_prior_count": int(len(unreadable_priors)),
        "unreadable_prior_examples": unreadable_priors[:16],
        "unreadable_reference_count": int(len(unreadable_references)),
        "unreadable_reference_examples": unreadable_references[:16],
        "usable_mean": usable_mean,
        "frames": [asdict(frame) for frame in frames],
    }
    summary = {
        "version": "align_vosr_prior_size_v0",
        "prior_dir": str(prior_dir),
        "reference_dir": str(reference_dir) if reference_dir is not None else None,
        "output_root": str(output_root),
        "processed": int(len(frames)),
        "missing": int(len(missing)),
        "unreadable_priors": int(len(unreadable_priors)),
        "unreadable_references": int(len(unreadable_references)),
        "skipped_large_delta": skipped_large_delta,
        "delta_width": _stats(delta_w_values),
        "delta_height": _stats(delta_h_values),
        "valid_fraction_mean": usable_mean,
        "valid_fraction_min": float(np.min(valid_fractions)) if valid_fractions else None,
        "valid_fraction_max": float(np.max(valid_fractions)) if valid_fractions else None,
        "anchor": str(args.anchor),
        "fill_mode": str(args.fill_mode),
        "match_mode": match_mode,
        "reference_split": selected_reference_split,
        "split_counts": split_counts,
        "requested_match_mode": requested_match_mode,
        "name_match_count": int(name_match_count),
        "order_fallback_count": int(order_fallback_count),
        "start_prior_name": start_prior_name or None,
        "start_reference_name": start_reference_name or None,
        "dry_run": bool(args.dry_run),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "align_vosr_prior_size_v0_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[align-vosr-prior-v0] fused priors : {fused_dir}", flush=True)
    print(f"[align-vosr-prior-v0] masks        : {mask_dir}", flush=True)
    print(f"[align-vosr-prior-v0] manifest     : {output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from utils.prior_injection import index_image_dir, normalize_image_name


@dataclass
class HRGSSceneLayout:
    scene_root: str
    source_images_subdir: str
    target_images_subdir: str
    source_image_dir: str
    target_image_dir: str
    priors_dir: Optional[str]
    sparse_root: Optional[str]
    vggt_root: Optional[str]
    repo_root: Optional[str]
    project_scale_tag: str = "x8_to_2"


def _resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def resolve_hrgs_scene_layout(
    scene_root: str | Path,
    *,
    source_images_subdir: str = "images_8",
    target_images_subdir: str = "images_2",
    priors_dir: str | Path | None = None,
    vggt_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> HRGSSceneLayout:
    scene_root_path = Path(scene_root).expanduser().resolve()
    source_image_dir = scene_root_path / source_images_subdir
    target_image_dir = scene_root_path / target_images_subdir
    sparse_root = scene_root_path if (scene_root_path / "sparse" / "0").is_dir() else None

    if not source_image_dir.is_dir():
        raise FileNotFoundError(f"Source image dir not found: {source_image_dir}")
    if not target_image_dir.is_dir():
        raise FileNotFoundError(f"Target image dir not found: {target_image_dir}")

    resolved_priors = None
    if priors_dir is None:
        default_priors = scene_root_path / "priors"
        if default_priors.is_dir():
            resolved_priors = str(default_priors)
    else:
        priors_path = Path(priors_dir).expanduser().resolve()
        if not priors_path.is_dir():
            raise FileNotFoundError(f"Priors dir not found: {priors_path}")
        resolved_priors = str(priors_path)

    resolved_vggt = _resolve_path(vggt_root) if vggt_root is not None else None
    resolved_repo = _resolve_path(repo_root) if repo_root is not None else None

    return HRGSSceneLayout(
        scene_root=str(scene_root_path),
        source_images_subdir=source_images_subdir,
        target_images_subdir=target_images_subdir,
        source_image_dir=str(source_image_dir),
        target_image_dir=str(target_image_dir),
        priors_dir=resolved_priors,
        sparse_root=str(sparse_root) if sparse_root is not None else None,
        vggt_root=resolved_vggt,
        repo_root=resolved_repo,
    )


def build_hrgs_view_records(
    layout: HRGSSceneLayout,
    *,
    require_priors: bool = False,
    limit: int = 0,
) -> List[Dict[str, object]]:
    source_index = index_image_dir(layout.source_image_dir)
    target_index = index_image_dir(layout.target_image_dir)
    prior_index = index_image_dir(layout.priors_dir) if layout.priors_dir else {}

    shared_names = sorted(set(source_index.keys()) & set(target_index.keys()))
    if require_priors:
        shared_names = [name for name in shared_names if name in prior_index]

    if limit > 0:
        shared_names = shared_names[: int(limit)]

    records: List[Dict[str, object]] = []
    for stem in shared_names:
        source_path = source_index[stem]
        target_path = target_index[stem]
        prior_path = prior_index.get(stem)
        record = {
            "image_name": normalize_image_name(stem),
            "lr_path": str(source_path.resolve()),
            "target_path": str(target_path.resolve()),
            "prior_path": str(prior_path.resolve()) if prior_path is not None else None,
            "has_prior": prior_path is not None,
        }
        records.append(record)
    return records


def build_hrgs_manifest_payload(
    layout: HRGSSceneLayout,
    records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    total_records = len(records)
    with_prior = sum(1 for record in records if bool(record.get("has_prior")))
    missing_prior = total_records - with_prior
    return {
        "layout": asdict(layout),
        "counts": {
            "num_views": total_records,
            "num_views_with_prior": with_prior,
            "num_views_missing_prior": missing_prior,
        },
        "views": list(records),
    }


def save_hrgs_manifest(path: str | Path, payload: Dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

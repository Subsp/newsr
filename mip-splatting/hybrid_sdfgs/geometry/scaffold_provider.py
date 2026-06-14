from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ScaffoldLoadConfig:
    path: str
    normalize_normals: bool = True
    point_keys: tuple[str, ...] = ("points", "xyz", "vertices")
    normal_keys: tuple[str, ...] = ("normals", "normal", "n")


@dataclass
class ScaffoldData:
    points: torch.Tensor
    normals: torch.Tensor | None
    source_path: str

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def has_normals(self) -> bool:
        return self.normals is not None


def _pick_first_key(container, keys: tuple[str, ...]):
    for key in keys:
        if key in container:
            return container[key]
    return None


def _to_tensor(arr, name: str) -> torch.Tensor:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{name} must have shape [N, 3+]")
    out = torch.from_numpy(arr[:, :3].astype(np.float32, copy=False)).contiguous()
    return out


def load_scaffold_data(cfg: ScaffoldLoadConfig) -> ScaffoldData:
    path = os.path.abspath(os.path.expanduser(cfg.path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scaffold file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        payload = np.load(path, allow_pickle=False)
        points_np = _pick_first_key(payload, cfg.point_keys)
        normals_np = _pick_first_key(payload, cfg.normal_keys)
        if points_np is None:
            available = ", ".join(sorted(payload.files))
            raise KeyError(
                "No point array found in scaffold npz. "
                f"Expected one of {cfg.point_keys}, available: {available}"
            )
        points = _to_tensor(points_np, "points")
        normals = None if normals_np is None else _to_tensor(normals_np, "normals")
    elif ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] not in (3, 6):
            raise ValueError("npy scaffold must be [N,3] or [N,6]")
        points = _to_tensor(arr[:, :3], "points")
        normals = _to_tensor(arr[:, 3:6], "normals") if arr.shape[1] == 6 else None
    else:
        raise ValueError(
            f"Unsupported scaffold format: {ext}. Use .npz(points[,normals]) or .npy([N,3|6])."
        )

    if points.shape[0] == 0:
        raise ValueError("Scaffold has no points.")

    if normals is not None and cfg.normalize_normals:
        normals = F.normalize(normals, dim=-1)

    return ScaffoldData(points=points, normals=normals, source_path=path)

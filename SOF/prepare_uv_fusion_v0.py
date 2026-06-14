import importlib.util
import json
import math
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

_COLMAP_LOADER_PATH = Path(__file__).resolve().parent / "scene" / "colmap_loader.py"
_COLMAP_LOADER_SPEC = importlib.util.spec_from_file_location("prepare_uv_fusion_colmap_loader", _COLMAP_LOADER_PATH)
if _COLMAP_LOADER_SPEC is None or _COLMAP_LOADER_SPEC.loader is None:
    raise ImportError(f"Failed to load COLMAP parser module from {_COLMAP_LOADER_PATH}")
_colmap_loader = importlib.util.module_from_spec(_COLMAP_LOADER_SPEC)
_COLMAP_LOADER_SPEC.loader.exec_module(_colmap_loader)
qvec2rotmat = _colmap_loader.qvec2rotmat
read_extrinsics_binary = _colmap_loader.read_extrinsics_binary
read_extrinsics_text = _colmap_loader.read_extrinsics_text
read_intrinsics_binary = _colmap_loader.read_intrinsics_binary
read_intrinsics_text = _colmap_loader.read_intrinsics_text


@dataclass
class SimpleView:
    image_name: str
    image_width: int
    image_height: int
    focal_x: float
    focal_y: float
    R: np.ndarray
    T: np.ndarray
    camera_center: np.ndarray


@dataclass
class ProxyBuildResult:
    mesh: trimesh.Trimesh
    pitch: float
    target_faces: int
    original_face_count: int


def normalize_image_name(image_name: str) -> str:
    return Path(image_name).stem


def index_image_dir(image_dir: str) -> Dict[str, Path]:
    indexed: Dict[str, Path] = {}
    for path in sorted(Path(image_dir).rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        indexed[path.stem] = path
        indexed[path.name] = path
    return indexed


def fov2focal(fov: float, pixels: int) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def compute_camera_center(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    world_to_view = np.eye(4, dtype=np.float32)
    world_to_view[:3, :3] = np.asarray(R, dtype=np.float32).T
    world_to_view[:3, 3] = np.asarray(T, dtype=np.float32)
    camera_to_world = np.linalg.inv(world_to_view)
    return camera_to_world[:3, 3].astype(np.float32, copy=False)


def is_camera_source_root(path: Path) -> bool:
    return (path / "sparse" / "0").exists() or (path / "transforms_train.json").is_file()


def resolve_camera_source_root(args) -> Path:
    if args.camera_source:
        resolved = Path(args.camera_source).expanduser().resolve()
        return resolved
    prior_root = Path(args.prior_dir).expanduser().resolve()
    base_root = prior_root.parent
    if is_camera_source_root(base_root):
        return base_root
    child_candidates = sorted(path for path in base_root.iterdir() if path.is_dir())
    matched = [path for path in child_candidates if is_camera_source_root(path)]
    if len(matched) == 1:
        return matched[0]
    return base_root


def load_triangle_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh_obj = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    if hasattr(mesh_obj, "dump"):
        dumped = mesh_obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
    raise ValueError(f"Failed to load triangle mesh from {mesh_path}")


def simplify_mesh_vertex_clustering(
    vertices: np.ndarray,
    faces: np.ndarray,
    pitch: float,
) -> trimesh.Trimesh:
    if pitch <= 0:
        raise ValueError(f"pitch must be positive, got {pitch}")

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    bbox_min = np.min(vertices, axis=0)
    grid = np.floor((vertices - bbox_min[None, :]) / float(pitch)).astype(np.int64)
    unique_cells, inverse = np.unique(grid, axis=0, return_inverse=True)
    reduced_vertices = np.zeros((unique_cells.shape[0], 3), dtype=np.float64)
    np.add.at(reduced_vertices, inverse, vertices.astype(np.float64, copy=False))
    counts = np.bincount(inverse, minlength=unique_cells.shape[0]).astype(np.float64, copy=False)
    reduced_vertices /= np.clip(counts[:, None], 1.0, None)

    remapped_faces = inverse[faces]
    valid = (
        (remapped_faces[:, 0] != remapped_faces[:, 1])
        & (remapped_faces[:, 1] != remapped_faces[:, 2])
        & (remapped_faces[:, 0] != remapped_faces[:, 2])
    )
    remapped_faces = remapped_faces[valid]
    if remapped_faces.shape[0] == 0:
        raise RuntimeError(f"Vertex clustering with pitch={pitch} collapsed all faces.")

    face_keys = np.sort(remapped_faces, axis=1)
    _, unique_idx = np.unique(face_keys, axis=0, return_index=True)
    remapped_faces = remapped_faces[np.sort(unique_idx)].astype(np.int64, copy=False)

    proxy_mesh = trimesh.Trimesh(
        vertices=reduced_vertices.astype(np.float32, copy=False),
        faces=remapped_faces,
        process=False,
    )
    proxy_mesh.remove_unreferenced_vertices()
    return proxy_mesh


def build_proxy_mesh_if_needed(
    mesh_obj: trimesh.Trimesh,
    args,
) -> Optional[ProxyBuildResult]:
    if bool(args.disable_proxy):
        return None

    original_face_count = int(len(mesh_obj.faces))
    trigger_faces = int(args.proxy_trigger_faces)
    if trigger_faces > 0 and original_face_count < trigger_faces:
        return None

    target_faces = int(args.proxy_target_faces)
    if target_faces <= 0 or original_face_count <= target_faces:
        return None

    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    bbox_extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    bbox_diag = float(np.linalg.norm(bbox_extent))
    if bbox_diag <= 1e-8:
        raise RuntimeError("Cannot build proxy mesh from a degenerate bounding box.")

    pitch_low = max(bbox_diag / 4096.0, 1e-8)
    pitch_high = max(bbox_diag / 4.0, pitch_low * 2.0)
    best_result: Optional[Tuple[float, trimesh.Trimesh]] = None
    current_pitch = pitch_low

    for _ in range(max(int(args.proxy_search_steps), 1)):
        proxy_mesh = simplify_mesh_vertex_clustering(vertices=vertices, faces=faces, pitch=current_pitch)
        proxy_face_count = int(len(proxy_mesh.faces))
        if proxy_face_count <= target_faces:
            best_result = (current_pitch, proxy_mesh)
            break
        pitch_low = current_pitch
        current_pitch *= 2.0
        if current_pitch >= pitch_high:
            break

    if best_result is None:
        # Continue searching up to the upper bound if exponential growth wasn't enough.
        current_pitch = max(current_pitch, pitch_high)
        for _ in range(max(int(args.proxy_search_steps), 1)):
            proxy_mesh = simplify_mesh_vertex_clustering(vertices=vertices, faces=faces, pitch=current_pitch)
            proxy_face_count = int(len(proxy_mesh.faces))
            if proxy_face_count <= target_faces:
                best_result = (current_pitch, proxy_mesh)
                break
            current_pitch *= 1.5

    if best_result is None:
        raise RuntimeError(
            "Failed to build a proxy mesh under the requested face budget. "
            f"original_faces={original_face_count}, target_faces={target_faces}."
        )

    best_pitch, best_mesh = best_result
    low = pitch_low
    high = best_pitch
    refined_mesh = best_mesh
    for _ in range(max(int(args.proxy_search_steps), 1)):
        if high <= low * 1.01:
            break
        mid = 0.5 * (low + high)
        proxy_mesh = simplify_mesh_vertex_clustering(vertices=vertices, faces=faces, pitch=mid)
        proxy_face_count = int(len(proxy_mesh.faces))
        if proxy_face_count <= target_faces:
            high = mid
            refined_mesh = proxy_mesh
        else:
            low = mid

    return ProxyBuildResult(
        mesh=refined_mesh,
        pitch=float(high),
        target_faces=target_faces,
        original_face_count=original_face_count,
    )


def stats_from_array(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _build_simple_view(
    image_name: str,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    R: np.ndarray,
    T: np.ndarray,
) -> SimpleView:
    return SimpleView(
        image_name=normalize_image_name(image_name),
        image_width=int(width),
        image_height=int(height),
        focal_x=fov2focal(float(fov_x), int(width)),
        focal_y=fov2focal(float(fov_y), int(height)),
        R=np.asarray(R, dtype=np.float32),
        T=np.asarray(T, dtype=np.float32),
        camera_center=compute_camera_center(R, T),
    )


def load_colmap_train_views(args) -> List[SimpleView]:
    source_root = resolve_camera_source_root(args)
    sparse_root = source_root / "sparse" / "0"
    images_bin = sparse_root / "images.bin"
    cameras_bin = sparse_root / "cameras.bin"
    images_txt = sparse_root / "images.txt"
    cameras_txt = sparse_root / "cameras.txt"

    if images_bin.is_file() and cameras_bin.is_file():
        cam_extrinsics = read_extrinsics_binary(str(images_bin))
        cam_intrinsics = read_intrinsics_binary(str(cameras_bin))
    elif images_txt.is_file() and cameras_txt.is_file():
        cam_extrinsics = read_extrinsics_text(str(images_txt))
        cam_intrinsics = read_intrinsics_text(str(cameras_txt))
    else:
        raise FileNotFoundError(f"Could not find COLMAP cameras under {sparse_root}")

    views: List[SimpleView] = []
    for image_id in sorted(cam_extrinsics.keys(), key=lambda key: cam_extrinsics[key].name):
        extr = cam_extrinsics[image_id]
        intr = cam_intrinsics[extr.camera_id]
        width = int(intr.width)
        height = int(intr.height)
        model_name = str(intr.model)
        if model_name == "SIMPLE_PINHOLE":
            focal_x = float(intr.params[0])
            focal_y = float(intr.params[0])
        elif model_name == "PINHOLE":
            focal_x = float(intr.params[0])
            focal_y = float(intr.params[1])
        else:
            raise ValueError(
                f"Unsupported COLMAP camera model '{model_name}'. Expected SIMPLE_PINHOLE or PINHOLE."
            )

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.asarray(extr.tvec, dtype=np.float32)
        views.append(
            _build_simple_view(
                image_name=extr.name,
                width=width,
                height=height,
                fov_x=focal2fov(focal_x, width),
                fov_y=focal2fov(focal_y, height),
                R=R,
                T=T,
            )
        )

    if bool(args.eval):
        train_views = [view for idx, view in enumerate(views) if idx % max(int(args.llffhold), 1) != 0]
    else:
        train_views = views
    print(f"Loaded {len(train_views)} COLMAP training views from {source_root}")
    return train_views


def _load_blender_views_from_transforms(transforms_path: Path, blender_extension: str) -> List[SimpleView]:
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    fov_x = float(payload["camera_angle_x"])
    source_root = transforms_path.parent

    views: List[SimpleView] = []
    for frame in payload["frames"]:
        frame_path = str(frame["file_path"])
        suffix = Path(frame_path).suffix
        if not suffix:
            frame_path = f"{frame_path}{blender_extension}"
        image_path = source_root / frame_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Could not find Blender frame image: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size

        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        c2w[:3, 1:3] *= -1.0
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        views.append(
            _build_simple_view(
                image_name=image_path.name,
                width=width,
                height=height,
                fov_x=fov_x,
                fov_y=focal2fov(fov2focal(fov_x, width), height),
                R=R,
                T=T,
            )
        )
    return views


def load_blender_train_views(args) -> List[SimpleView]:
    source_root = resolve_camera_source_root(args)
    train_views = _load_blender_views_from_transforms(
        source_root / "transforms_train.json",
        blender_extension=args.blender_extension,
    )
    if not bool(args.eval):
        test_transforms_path = source_root / "transforms_test.json"
        if test_transforms_path.is_file():
            train_views.extend(
                _load_blender_views_from_transforms(
                    test_transforms_path,
                    blender_extension=args.blender_extension,
                )
            )
    print(f"Loaded {len(train_views)} Blender training views from {source_root}")
    return train_views


def load_train_cameras_only(args) -> List[SimpleView]:
    source_root = resolve_camera_source_root(args)
    if (source_root / "sparse").exists():
        return load_colmap_train_views(args)
    if (source_root / "transforms_train.json").is_file():
        return load_blender_train_views(args)
    raise RuntimeError(
        "Could not recognize scene camera source under "
        f"{source_root}. Expected either 'sparse/0' for COLMAP or 'transforms_train.json' for Blender."
    )


def lookup_indexed_image(index: Dict[str, Path], image_name: str) -> Optional[Path]:
    candidates = [
        image_name,
        normalize_image_name(image_name),
        Path(image_name).name,
        Path(image_name).stem,
    ]
    for key in candidates:
        if key in index:
            return index[key]
    lower_index = {str(key).lower(): value for key, value in index.items()}
    for key in candidates:
        value = lower_index.get(str(key).lower())
        if value is not None:
            return value
    return None


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def camera_center_numpy(cam) -> np.ndarray:
    return np.asarray(cam.camera_center, dtype=np.float32).reshape(3)


def load_face_ids_from_payload(payload_path: str, key: str, num_faces: int) -> np.ndarray:
    import torch

    payload = torch.load(payload_path, map_location="cpu")
    if key in payload:
        value = tensor_to_numpy(payload[key])
    elif "roi_face_ids" in payload:
        value = tensor_to_numpy(payload["roi_face_ids"])
    elif "roi_face_mask" in payload:
        value = tensor_to_numpy(payload["roi_face_mask"])
    else:
        raise KeyError(
            f"Face payload does not contain '{key}', 'roi_face_ids', or 'roi_face_mask': {payload_path}"
        )
    value = np.asarray(value)
    if value.dtype == np.bool_:
        value = value.reshape(-1)
        if value.shape[0] != int(num_faces):
            raise ValueError(f"Boolean face mask has {value.shape[0]} entries, expected {num_faces}.")
        face_ids = np.flatnonzero(value).astype(np.int64, copy=False)
    else:
        face_ids = value.reshape(-1).astype(np.int64, copy=False)
    if face_ids.size and np.any((face_ids < 0) | (face_ids >= int(num_faces))):
        raise ValueError(f"Face ids in payload are outside [0, {num_faces}).")
    return np.unique(face_ids).astype(np.int64, copy=False)


def select_face_ids(mesh_obj: trimesh.Trimesh, args) -> Tuple[np.ndarray, str]:
    num_faces = len(mesh_obj.faces)
    if args.face_selection == "all_faces":
        face_ids = np.arange(num_faces, dtype=np.int64)
        source = "all_faces"
    elif args.face_selection == "payload":
        if not args.face_payload:
            raise ValueError("--face_selection payload requires --face_payload")
        face_ids = load_face_ids_from_payload(args.face_payload, args.face_payload_key, num_faces)
        source = f"payload:{Path(args.face_payload).name}:{args.face_payload_key}"
    else:
        raise ValueError(f"Unsupported face_selection={args.face_selection}")

    stride = max(int(args.face_stride), 1)
    if stride > 1:
        face_ids = face_ids[::stride]
        source += f"_stride_{stride}"
    max_faces = int(args.max_faces)
    if max_faces > 0 and face_ids.size > max_faces:
        if args.face_sample_mode == "random":
            rng = np.random.default_rng(int(args.random_seed))
            face_ids = np.sort(rng.choice(face_ids, size=max_faces, replace=False)).astype(np.int64, copy=False)
            source += f"_random_{max_faces}"
        else:
            face_ids = face_ids[:max_faces]
            source += f"_prefix_{max_faces}"
    return face_ids.astype(np.int64, copy=False), source


def load_rgb_image_np(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def bilinear_sample_rgb(image_hw3: np.ndarray, projected_xy: np.ndarray) -> np.ndarray:
    h, w, _ = image_hw3.shape
    x = np.clip(projected_xy[:, 0], 0.0, max(w - 1.0, 0.0))
    y = np.clip(projected_xy[:, 1], 0.0, max(h - 1.0, 0.0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wa = (x1.astype(np.float32) - x) * (y1.astype(np.float32) - y)
    wb = (x - x0.astype(np.float32)) * (y1.astype(np.float32) - y)
    wc = (x1.astype(np.float32) - x) * (y - y0.astype(np.float32))
    wd = (x - x0.astype(np.float32)) * (y - y0.astype(np.float32))
    same_x = x0 == x1
    same_y = y0 == y1
    wa[same_x] = 1.0 - (y[same_x] - y0[same_x].astype(np.float32))
    wc[same_x] = y[same_x] - y0[same_x].astype(np.float32)
    wb[same_x] = 0.0
    wd[same_x] = 0.0
    wa[same_y] = 1.0 - (x[same_y] - x0[same_y].astype(np.float32))
    wb[same_y] = x[same_y] - x0[same_y].astype(np.float32)
    wc[same_y] = 0.0
    wd[same_y] = 0.0

    Ia = image_hw3[y0, x0]
    Ib = image_hw3[y0, x1]
    Ic = image_hw3[y1, x0]
    Id = image_hw3[y1, x1]
    return (
        Ia * wa[:, None]
        + Ib * wb[:, None]
        + Ic * wc[:, None]
        + Id * wd[:, None]
    ).astype(np.float32, copy=False)


def bilinear_sample_scalar(image_hw: np.ndarray, projected_xy: np.ndarray) -> np.ndarray:
    h, w = image_hw.shape
    x = np.clip(projected_xy[:, 0], 0.0, max(w - 1.0, 0.0))
    y = np.clip(projected_xy[:, 1], 0.0, max(h - 1.0, 0.0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wa = (x1.astype(np.float32) - x) * (y1.astype(np.float32) - y)
    wb = (x - x0.astype(np.float32)) * (y1.astype(np.float32) - y)
    wc = (x1.astype(np.float32) - x) * (y - y0.astype(np.float32))
    wd = (x - x0.astype(np.float32)) * (y - y0.astype(np.float32))
    same_x = x0 == x1
    same_y = y0 == y1
    wa[same_x] = 1.0 - (y[same_x] - y0[same_x].astype(np.float32))
    wc[same_x] = y[same_x] - y0[same_x].astype(np.float32)
    wb[same_x] = 0.0
    wd[same_x] = 0.0
    wa[same_y] = 1.0 - (x[same_y] - x0[same_y].astype(np.float32))
    wb[same_y] = x[same_y] - x0[same_y].astype(np.float32)
    wc[same_y] = 0.0
    wd[same_y] = 0.0

    Ia = image_hw[y0, x0]
    Ib = image_hw[y0, x1]
    Ic = image_hw[y1, x0]
    Id = image_hw[y1, x1]
    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).astype(np.float32, copy=False)


def project_points_camera(cam, points_xyz: np.ndarray, depth_min: float, margin: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(cam.R, dtype=np.float32)
    T = np.asarray(cam.T, dtype=np.float32)
    xyz_cam = points_xyz @ R + T[None, :]
    z = xyz_cam[:, 2]
    x = xyz_cam[:, 0] / np.clip(z, 1e-6, None) * float(cam.focal_x) + float(cam.image_width) / 2.0
    y = xyz_cam[:, 1] / np.clip(z, 1e-6, None) * float(cam.focal_y) + float(cam.image_height) / 2.0
    valid = z > float(depth_min)
    valid &= x >= float(margin)
    valid &= x < float(cam.image_width - margin)
    valid &= y >= float(margin)
    valid &= y < float(cam.image_height - margin)
    return np.stack([x, y, z], axis=1).astype(np.float32, copy=False), valid


def rasterize_projected_triangles_zbuffer(
    tri_xy: np.ndarray,
    tri_depth: np.ndarray,
    face_ids: np.ndarray,
    height: int,
    width: int,
    depth_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    face_buffer = np.full((height, width), -1, dtype=np.int64)

    for tri, z, face_id in zip(tri_xy, tri_depth, face_ids):
        min_x = max(int(np.floor(np.min(tri[:, 0]))), 0)
        max_x = min(int(np.ceil(np.max(tri[:, 0]))), width - 1)
        min_y = max(int(np.floor(np.min(tri[:, 1]))), 0)
        max_y = min(int(np.ceil(np.max(tri[:, 1]))), height - 1)
        if max_x < min_x or max_y < min_y:
            continue

        x0, y0 = tri[0]
        x1, y1 = tri[1]
        x2, y2 = tri[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) < 1e-8:
            continue

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        px = xx.astype(np.float32) + 0.5
        py = yy.astype(np.float32) + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue

        interp_z = w0 * z[0] + w1 * z[1] + w2 * z[2]
        update = inside & (interp_z > float(depth_min))
        if not np.any(update):
            continue

        sub_depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update &= interp_z < sub_depth
        if not np.any(update):
            continue
        sub_depth[update] = interp_z[update]
        sub_face = face_buffer[min_y : max_y + 1, min_x : max_x + 1]
        sub_face[update] = int(face_id)

    return depth_buffer, face_buffer


def resolve_chart_layout(
    num_faces: int,
    desired_chart_resolution: int,
    padding: int,
    max_atlas_resolution: int,
    min_chart_resolution: int,
) -> Dict[str, int]:
    if num_faces <= 0:
        raise ValueError("num_faces must be positive")
    cols = int(math.ceil(math.sqrt(num_faces)))
    rows = int(math.ceil(num_faces / max(cols, 1)))

    max_chart_x = max(int(max_atlas_resolution // max(cols, 1)) - 2 * int(padding), 0)
    max_chart_y = max(int(max_atlas_resolution // max(rows, 1)) - 2 * int(padding), 0)
    chart_resolution = min(int(desired_chart_resolution), max_chart_x, max_chart_y)
    if chart_resolution < int(min_chart_resolution):
        raise RuntimeError(
            "Atlas would be too large for the requested face count. "
            f"num_faces={num_faces}, max_atlas_resolution={max_atlas_resolution}, "
            f"resolved_chart_resolution={chart_resolution}, min_chart_resolution={min_chart_resolution}."
        )

    stride = int(chart_resolution) + 2 * int(padding)
    atlas_width = cols * stride
    atlas_height = rows * stride
    return {
        "cols": cols,
        "rows": rows,
        "chart_resolution": int(chart_resolution),
        "padding": int(padding),
        "stride": int(stride),
        "atlas_width": int(atlas_width),
        "atlas_height": int(atlas_height),
    }


def build_per_face_uv_atlas(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    desired_chart_resolution: int,
    padding: int,
    max_atlas_resolution: int,
    min_chart_resolution: int,
) -> Dict[str, np.ndarray | int]:
    layout = resolve_chart_layout(
        num_faces=int(face_ids.shape[0]),
        desired_chart_resolution=int(desired_chart_resolution),
        padding=int(padding),
        max_atlas_resolution=int(max_atlas_resolution),
        min_chart_resolution=int(min_chart_resolution),
    )
    face_ids = np.asarray(face_ids, dtype=np.int64)
    chart_resolution = int(layout["chart_resolution"])
    stride = int(layout["stride"])
    pad = int(layout["padding"])
    cols = int(layout["cols"])
    atlas_width = int(layout["atlas_width"])
    atlas_height = int(layout["atlas_height"])

    face_uv_pixels = np.zeros((face_ids.shape[0], 3, 2), dtype=np.float32)
    for local_face_idx in range(face_ids.shape[0]):
        row = local_face_idx // cols
        col = local_face_idx % cols
        x0 = col * stride + pad
        y0 = row * stride + pad
        x1 = x0 + chart_resolution - 1
        y1 = y0 + chart_resolution - 1
        face_uv_pixels[local_face_idx] = np.asarray(
            [
                [x0, y0],
                [x1, y0],
                [x0, y1],
            ],
            dtype=np.float32,
        )

    face_uv_norm = face_uv_pixels.copy()
    face_uv_norm[..., 0] = (face_uv_norm[..., 0] + 0.5) / max(float(atlas_width), 1.0)
    face_uv_norm[..., 1] = 1.0 - (face_uv_norm[..., 1] + 0.5) / max(float(atlas_height), 1.0)
    return {
        "face_ids": face_ids,
        "face_uv_pixels": face_uv_pixels,
        "face_uv_norm": face_uv_norm.astype(np.float32, copy=False),
        "chart_resolution": chart_resolution,
        "padding": pad,
        "stride": stride,
        "cols": cols,
        "rows": int(layout["rows"]),
        "atlas_width": atlas_width,
        "atlas_height": atlas_height,
    }


def rasterize_atlas_texels(face_uv_pixels: np.ndarray) -> Dict[str, np.ndarray]:
    texel_x = []
    texel_y = []
    texel_bary = []
    texel_face_local = []
    face_texel_count = np.zeros((face_uv_pixels.shape[0],), dtype=np.int32)

    for local_face_idx, tri in enumerate(tqdm(face_uv_pixels, desc="Rasterizing UV atlas", dynamic_ncols=True)):
        min_x = int(np.floor(np.min(tri[:, 0])))
        max_x = int(np.ceil(np.max(tri[:, 0])))
        min_y = int(np.floor(np.min(tri[:, 1])))
        max_y = int(np.ceil(np.max(tri[:, 1])))

        x0, y0 = tri[0]
        x1, y1 = tri[1]
        x2, y2 = tri[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) < 1e-8:
            continue

        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                sample_x = float(px) + 0.5
                sample_y = float(py) + 0.5
                w0 = ((y1 - y2) * (sample_x - x2) + (x2 - x1) * (sample_y - y2)) / denom
                w1 = ((y2 - y0) * (sample_x - x2) + (x0 - x2) * (sample_y - y2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 < -1e-5 or w1 < -1e-5 or w2 < -1e-5:
                    continue
                texel_x.append(px)
                texel_y.append(py)
                texel_bary.append([w0, w1, w2])
                texel_face_local.append(local_face_idx)
                face_texel_count[local_face_idx] += 1

    if not texel_x:
        raise RuntimeError("UV atlas rasterization produced no occupied texels.")

    return {
        "texel_x": np.asarray(texel_x, dtype=np.int32),
        "texel_y": np.asarray(texel_y, dtype=np.int32),
        "texel_bary": np.asarray(texel_bary, dtype=np.float32),
        "texel_face_local": np.asarray(texel_face_local, dtype=np.int32),
        "face_texel_count": face_texel_count,
    }


def build_view_face_buffers(
    mesh: trimesh.Trimesh,
    triangles: np.ndarray,
    face_ids: np.ndarray,
    face_texel_count: np.ndarray,
    view,
    depth_min: float,
    front_facing_only: bool,
    density_weight_min: float,
    density_weight_max: float,
    visibility_mode: str = "triangle_raster",
) -> Dict[str, np.ndarray]:
    normals = np.asarray(mesh.face_normals, dtype=np.float32)[face_ids]
    face_centers = triangles.mean(axis=1)
    cam_center = camera_center_numpy(view)
    view_dir = cam_center[None, :] - face_centers
    view_dir_norm = view_dir / np.clip(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-6, None)
    frontality = np.sum(normals * view_dir_norm, axis=1)
    if bool(front_facing_only):
        front_mask = frontality > 0.0
    else:
        front_mask = np.ones((triangles.shape[0],), dtype=bool)

    R = np.asarray(view.R, dtype=np.float32)
    T = np.asarray(view.T, dtype=np.float32)
    tri_cam = triangles @ R[None, :, :] + T[None, None, :]
    tri_depth = tri_cam[..., 2]
    depth_mask = np.all(tri_depth > float(depth_min), axis=1)

    height = int(view.image_height)
    width = int(view.image_width)
    x = tri_cam[..., 0] / np.clip(tri_depth, 1e-6, None) * float(view.focal_x) + float(width) / 2.0
    y = tri_cam[..., 1] / np.clip(tri_depth, 1e-6, None) * float(view.focal_y) + float(height) / 2.0
    tri_xy = np.stack([x, y], axis=-1)

    bbox_min = np.floor(np.min(tri_xy, axis=1))
    bbox_max = np.ceil(np.max(tri_xy, axis=1))
    overlaps = (
        (bbox_max[:, 0] >= 0)
        & (bbox_min[:, 0] < width)
        & (bbox_max[:, 1] >= 0)
        & (bbox_min[:, 1] < height)
    )
    valid_faces = front_mask & depth_mask & overlaps

    face_buffer = np.full((height, width), -1, dtype=np.int32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    if np.any(valid_faces) and visibility_mode == "triangle_raster":
        kept_local_face_ids = np.flatnonzero(valid_faces).astype(np.int64, copy=False)
        depth_buffer, raw_face_buffer = rasterize_projected_triangles_zbuffer(
            tri_xy=tri_xy[valid_faces],
            tri_depth=tri_depth[valid_faces],
            face_ids=kept_local_face_ids,
            height=height,
            width=width,
            depth_min=float(depth_min),
        )
        face_buffer = raw_face_buffer.astype(np.int32, copy=False)
    elif np.any(valid_faces) and visibility_mode == "face_center":
        center_cam = face_centers @ R + T[None, :]
        center_depth = center_cam[:, 2]
        center_x = center_cam[:, 0] / np.clip(center_depth, 1e-6, None) * float(view.focal_x) + float(width) / 2.0
        center_y = center_cam[:, 1] / np.clip(center_depth, 1e-6, None) * float(view.focal_y) + float(height) / 2.0
        cx = np.round(center_x[valid_faces]).astype(np.int64)
        cy = np.round(center_y[valid_faces]).astype(np.int64)
        center_in_bounds = (
            (cx >= 0)
            & (cx < width)
            & (cy >= 0)
            & (cy < height)
        )
        kept_local_face_ids = np.flatnonzero(valid_faces).astype(np.int64, copy=False)[center_in_bounds]
        if kept_local_face_ids.size > 0:
            cx = cx[center_in_bounds]
            cy = cy[center_in_bounds]
            cz = center_depth[kept_local_face_ids].astype(np.float32, copy=False)
            order = np.lexsort((cz, cy.astype(np.int64) * width + cx.astype(np.int64)))
            linear = cy[order].astype(np.int64) * width + cx[order].astype(np.int64)
            unique_linear, unique_idx = np.unique(linear, return_index=True)
            winning = order[unique_idx]
            win_x = cx[winning]
            win_y = cy[winning]
            win_z = cz[winning]
            win_face_ids = kept_local_face_ids[winning]
            depth_buffer[win_y, win_x] = win_z
            face_buffer[win_y, win_x] = win_face_ids.astype(np.int32, copy=False)

    edge01 = tri_xy[:, 1] - tri_xy[:, 0]
    edge02 = tri_xy[:, 2] - tri_xy[:, 0]
    projected_area = 0.5 * np.abs(edge01[:, 0] * edge02[:, 1] - edge01[:, 1] * edge02[:, 0])
    density = projected_area / np.clip(face_texel_count.astype(np.float32), 1.0, None)
    density_weight = np.sqrt(np.clip(density, 1e-6, None))
    density_weight = np.clip(density_weight, float(density_weight_min), float(density_weight_max))
    face_weight = np.clip(frontality, 0.0, 1.0).astype(np.float32) * density_weight.astype(np.float32)
    face_weight[~valid_faces] = 0.0

    return {
        "face_buffer": face_buffer,
        "depth_buffer": depth_buffer,
        "face_weight": face_weight.astype(np.float32, copy=False),
        "frontality": np.clip(frontality, 0.0, 1.0).astype(np.float32, copy=False),
        "projected_area": projected_area.astype(np.float32, copy=False),
        "valid_faces": valid_faces.astype(bool, copy=False),
    }


def initialize_consensus_state(num_texels: int) -> Dict[str, np.ndarray]:
    return {
        "best_rgb": np.zeros((num_texels, 3), dtype=np.float32),
        "best_weight": np.zeros((num_texels,), dtype=np.float32),
        "best_count": np.zeros((num_texels,), dtype=np.int32),
        "best_peak_weight": np.zeros((num_texels,), dtype=np.float32),
        "best_view_id": np.full((num_texels,), -1, dtype=np.int32),
        "alt_rgb": np.zeros((num_texels, 3), dtype=np.float32),
        "alt_weight": np.zeros((num_texels,), dtype=np.float32),
        "alt_count": np.zeros((num_texels,), dtype=np.int32),
        "alt_peak_weight": np.zeros((num_texels,), dtype=np.float32),
        "alt_view_id": np.full((num_texels,), -1, dtype=np.int32),
        "total_weight": np.zeros((num_texels,), dtype=np.float32),
        "sample_count": np.zeros((num_texels,), dtype=np.int32),
    }


def _update_cluster_mean(current_rgb: np.ndarray, current_weight: np.ndarray, sample_rgb: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    return (
        current_rgb * current_weight[:, None] + sample_rgb * sample_weight[:, None]
    ) / np.clip(current_weight[:, None] + sample_weight[:, None], 1e-8, None)


def update_consensus_state(
    state: Dict[str, np.ndarray],
    texel_ids: np.ndarray,
    sample_rgb: np.ndarray,
    sample_weight: np.ndarray,
    view_id: int,
    color_threshold: float,
):
    if texel_ids.size == 0:
        return

    state["total_weight"][texel_ids] += sample_weight
    state["sample_count"][texel_ids] += 1

    uninitialized = state["best_weight"][texel_ids] <= 0.0
    if np.any(uninitialized):
        ids = texel_ids[uninitialized]
        rgb = sample_rgb[uninitialized]
        weight = sample_weight[uninitialized]
        state["best_rgb"][ids] = rgb
        state["best_weight"][ids] = weight
        state["best_count"][ids] = 1
        state["best_peak_weight"][ids] = weight
        state["best_view_id"][ids] = int(view_id)

    remaining = ~uninitialized
    if not np.any(remaining):
        return

    ids = texel_ids[remaining]
    rgb = sample_rgb[remaining]
    weight = sample_weight[remaining]

    best_rgb = state["best_rgb"][ids]
    best_weight = state["best_weight"][ids]
    alt_rgb = state["alt_rgb"][ids]
    alt_weight = state["alt_weight"][ids]

    dist_best = np.linalg.norm(rgb - best_rgb, axis=1)
    dist_alt = np.full_like(dist_best, np.inf, dtype=np.float32)
    has_alt = alt_weight > 0.0
    if np.any(has_alt):
        dist_alt[has_alt] = np.linalg.norm(rgb[has_alt] - alt_rgb[has_alt], axis=1)

    assign_best = dist_best <= float(color_threshold)
    assign_alt = (~assign_best) & has_alt & (dist_alt <= float(color_threshold))
    create_alt = (~assign_best) & (~assign_alt) & (~has_alt)
    unresolved = (~assign_best) & (~assign_alt) & (~create_alt)
    replace_alt = unresolved & (weight > alt_weight)

    if np.any(assign_best):
        ids_best = ids[assign_best]
        rgb_best = rgb[assign_best]
        weight_best = weight[assign_best]
        old_weight = state["best_weight"][ids_best]
        state["best_rgb"][ids_best] = _update_cluster_mean(state["best_rgb"][ids_best], old_weight, rgb_best, weight_best)
        state["best_weight"][ids_best] = old_weight + weight_best
        state["best_count"][ids_best] += 1
        peak_update = weight_best > state["best_peak_weight"][ids_best]
        if np.any(peak_update):
            peak_ids = ids_best[peak_update]
            state["best_peak_weight"][peak_ids] = weight_best[peak_update]
            state["best_view_id"][peak_ids] = int(view_id)

    if np.any(assign_alt):
        ids_alt = ids[assign_alt]
        rgb_alt = rgb[assign_alt]
        weight_alt = weight[assign_alt]
        old_weight = state["alt_weight"][ids_alt]
        state["alt_rgb"][ids_alt] = _update_cluster_mean(state["alt_rgb"][ids_alt], old_weight, rgb_alt, weight_alt)
        state["alt_weight"][ids_alt] = old_weight + weight_alt
        state["alt_count"][ids_alt] += 1
        peak_update = weight_alt > state["alt_peak_weight"][ids_alt]
        if np.any(peak_update):
            peak_ids = ids_alt[peak_update]
            state["alt_peak_weight"][peak_ids] = weight_alt[peak_update]
            state["alt_view_id"][peak_ids] = int(view_id)

    if np.any(create_alt):
        ids_create = ids[create_alt]
        state["alt_rgb"][ids_create] = rgb[create_alt]
        state["alt_weight"][ids_create] = weight[create_alt]
        state["alt_count"][ids_create] = 1
        state["alt_peak_weight"][ids_create] = weight[create_alt]
        state["alt_view_id"][ids_create] = int(view_id)

    if np.any(replace_alt):
        ids_replace = ids[replace_alt]
        state["alt_rgb"][ids_replace] = rgb[replace_alt]
        state["alt_weight"][ids_replace] = weight[replace_alt]
        state["alt_count"][ids_replace] = 1
        state["alt_peak_weight"][ids_replace] = weight[replace_alt]
        state["alt_view_id"][ids_replace] = int(view_id)


def finalize_consensus_state(
    state: Dict[str, np.ndarray],
    min_views: int,
) -> Dict[str, np.ndarray]:
    choose_alt = state["alt_weight"] > state["best_weight"]
    consensus_rgb = state["best_rgb"].copy()
    consensus_weight = state["best_weight"].copy()
    consensus_peak_view = state["best_view_id"].copy()
    consensus_cluster_count = state["best_count"].copy()

    if np.any(choose_alt):
        consensus_rgb[choose_alt] = state["alt_rgb"][choose_alt]
        consensus_weight[choose_alt] = state["alt_weight"][choose_alt]
        consensus_peak_view[choose_alt] = state["alt_view_id"][choose_alt]
        consensus_cluster_count[choose_alt] = state["alt_count"][choose_alt]

    total_weight = np.clip(state["total_weight"], 1e-8, None)
    support_ratio = consensus_weight / total_weight
    coverage_factor = np.clip(state["sample_count"].astype(np.float32) / max(float(min_views), 1.0), 0.0, 1.0)
    confidence = support_ratio * coverage_factor
    disagreement = 1.0 - support_ratio
    valid_mask = (state["sample_count"] >= int(min_views)) & (consensus_cluster_count > 0) & (confidence > 0.0)
    mode_count = np.zeros_like(state["sample_count"], dtype=np.int32)
    mode_count[state["best_weight"] > 0.0] += 1
    mode_count[state["alt_weight"] > 0.0] += 1

    return {
        "consensus_rgb": consensus_rgb.astype(np.float32, copy=False),
        "confidence": confidence.astype(np.float32, copy=False),
        "disagreement": disagreement.astype(np.float32, copy=False),
        "support_count": state["sample_count"].astype(np.int32, copy=False),
        "cluster_support_count": consensus_cluster_count.astype(np.int32, copy=False),
        "support_ratio": support_ratio.astype(np.float32, copy=False),
        "mode_count": mode_count.astype(np.int32, copy=False),
        "exemplar_view_id": consensus_peak_view.astype(np.int32, copy=False),
        "valid_mask": valid_mask.astype(bool, copy=False),
    }


def compute_face_preview_scores(
    texel_face_local: np.ndarray,
    consensus: Dict[str, np.ndarray],
    num_faces: int,
) -> Dict[str, np.ndarray]:
    face_valid_texels = np.zeros((num_faces,), dtype=np.int32)
    face_confidence_sum = np.zeros((num_faces,), dtype=np.float32)
    face_support_ratio_sum = np.zeros((num_faces,), dtype=np.float32)
    face_disagreement_sum = np.zeros((num_faces,), dtype=np.float32)

    valid_mask = np.asarray(consensus["valid_mask"], dtype=bool)
    if np.any(valid_mask):
        valid_face_ids = texel_face_local[valid_mask].astype(np.int64, copy=False)
        np.add.at(face_valid_texels, valid_face_ids, 1)
        np.add.at(face_confidence_sum, valid_face_ids, consensus["confidence"][valid_mask].astype(np.float32, copy=False))
        np.add.at(face_support_ratio_sum, valid_face_ids, consensus["support_ratio"][valid_mask].astype(np.float32, copy=False))
        np.add.at(face_disagreement_sum, valid_face_ids, consensus["disagreement"][valid_mask].astype(np.float32, copy=False))

    denom = np.clip(face_valid_texels.astype(np.float32), 1.0, None)
    face_confidence_mean = face_confidence_sum / denom
    face_support_ratio_mean = face_support_ratio_sum / denom
    face_disagreement_mean = face_disagreement_sum / denom
    face_score = face_confidence_mean * (0.25 + 0.75 * face_support_ratio_mean) * np.log1p(face_valid_texels.astype(np.float32))
    face_score[face_valid_texels <= 0] = 0.0

    return {
        "face_valid_texels": face_valid_texels.astype(np.int32, copy=False),
        "face_confidence_mean": face_confidence_mean.astype(np.float32, copy=False),
        "face_support_ratio_mean": face_support_ratio_mean.astype(np.float32, copy=False),
        "face_disagreement_mean": face_disagreement_mean.astype(np.float32, copy=False),
        "face_score": face_score.astype(np.float32, copy=False),
    }


def select_preview_face_ids(
    face_metrics: Dict[str, np.ndarray],
    max_preview_faces: int,
) -> np.ndarray:
    if max_preview_faces <= 0:
        return np.zeros((0,), dtype=np.int64)
    valid_face_ids = np.flatnonzero(face_metrics["face_valid_texels"] > 0).astype(np.int64, copy=False)
    if valid_face_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    scores = face_metrics["face_score"][valid_face_ids]
    order = np.argsort(scores)[::-1]
    keep = valid_face_ids[order[: min(int(max_preview_faces), int(valid_face_ids.size))]]
    return np.sort(keep).astype(np.int64, copy=False)


def assemble_atlas_maps(
    atlas_width: int,
    atlas_height: int,
    texel_x: np.ndarray,
    texel_y: np.ndarray,
    consensus: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    rgb = np.zeros((atlas_height, atlas_width, 3), dtype=np.float32)
    confidence = np.zeros((atlas_height, atlas_width), dtype=np.float32)
    disagreement = np.zeros((atlas_height, atlas_width), dtype=np.float32)
    support_count = np.zeros((atlas_height, atlas_width), dtype=np.int32)
    support_ratio = np.zeros((atlas_height, atlas_width), dtype=np.float32)
    mode_count = np.zeros((atlas_height, atlas_width), dtype=np.int32)
    exemplar_view_id = np.full((atlas_height, atlas_width), -1, dtype=np.int32)
    valid_mask = np.zeros((atlas_height, atlas_width), dtype=bool)

    rgb[texel_y, texel_x] = consensus["consensus_rgb"]
    confidence[texel_y, texel_x] = consensus["confidence"]
    disagreement[texel_y, texel_x] = consensus["disagreement"]
    support_count[texel_y, texel_x] = consensus["support_count"]
    support_ratio[texel_y, texel_x] = consensus["support_ratio"]
    mode_count[texel_y, texel_x] = consensus["mode_count"]
    exemplar_view_id[texel_y, texel_x] = consensus["exemplar_view_id"]
    valid_mask[texel_y, texel_x] = consensus["valid_mask"]

    rgba = np.concatenate(
        [
            np.clip(rgb, 0.0, 1.0),
            confidence[..., None],
        ],
        axis=2,
    )
    return {
        "rgb": rgb,
        "rgba": rgba,
        "confidence": confidence,
        "disagreement": disagreement,
        "support_count": support_count,
        "support_ratio": support_ratio,
        "mode_count": mode_count,
        "exemplar_view_id": exemplar_view_id,
        "valid_mask": valid_mask,
    }


def compute_barycentric_coordinates_3d(points_xyz: np.ndarray, triangles_xyz: np.ndarray) -> np.ndarray:
    a = triangles_xyz[:, 0]
    b = triangles_xyz[:, 1]
    c = triangles_xyz[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = points_xyz - a
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=1).astype(np.float32, copy=False)
    return np.clip(bary, -2.0, 3.0)


def transfer_proxy_consensus_to_full_faces(
    original_mesh: trimesh.Trimesh,
    proxy_mesh: trimesh.Trimesh,
    proxy_face_uv_pixels: np.ndarray,
    atlas_maps: Dict[str, np.ndarray],
    chunk_size: int = 250000,
) -> Dict[str, np.ndarray]:
    original_vertices = np.asarray(original_mesh.vertices, dtype=np.float32)
    original_faces = np.asarray(original_mesh.faces, dtype=np.int64)
    proxy_vertices = np.asarray(proxy_mesh.vertices, dtype=np.float32)
    proxy_faces = np.asarray(proxy_mesh.faces, dtype=np.int64)
    proxy_triangles = proxy_vertices[proxy_faces]
    proxy_centroids = proxy_triangles.mean(axis=1).astype(np.float32, copy=False)
    proxy_normals = np.asarray(proxy_mesh.face_normals, dtype=np.float32)
    proxy_tree = cKDTree(proxy_centroids.astype(np.float64, copy=False))
    bbox_diag = float(np.linalg.norm(np.max(proxy_vertices, axis=0) - np.min(proxy_vertices, axis=0)))
    bbox_diag = max(bbox_diag, 1e-6)

    num_faces = int(original_faces.shape[0])
    rgb = np.zeros((num_faces, 3), dtype=np.float32)
    confidence = np.zeros((num_faces,), dtype=np.float32)
    disagreement = np.zeros((num_faces,), dtype=np.float32)
    support_ratio = np.zeros((num_faces,), dtype=np.float32)
    support_count = np.zeros((num_faces,), dtype=np.float32)
    proxy_face_id = np.full((num_faces,), -1, dtype=np.int32)

    atlas_rgb = atlas_maps["rgb"]
    atlas_confidence = atlas_maps["confidence"]
    atlas_disagreement = atlas_maps["disagreement"]
    atlas_support_ratio = atlas_maps["support_ratio"]
    atlas_support_count = atlas_maps["support_count"].astype(np.float32, copy=False)

    for start in tqdm(range(0, num_faces, chunk_size), desc="Transfer proxy consensus", dynamic_ncols=True):
        end = min(start + chunk_size, num_faces)
        chunk_faces = original_faces[start:end]
        chunk_triangles = original_vertices[chunk_faces]
        chunk_centroids = chunk_triangles.mean(axis=1).astype(np.float32, copy=False)
        chunk_normals = np.cross(
            chunk_triangles[:, 1] - chunk_triangles[:, 0],
            chunk_triangles[:, 2] - chunk_triangles[:, 0],
        ).astype(np.float32, copy=False)
        normal_norm = np.linalg.norm(chunk_normals, axis=1, keepdims=True)
        chunk_normals = chunk_normals / np.clip(normal_norm, 1e-6, None)

        k = min(4, proxy_centroids.shape[0])
        distances, candidate_ids = proxy_tree.query(chunk_centroids.astype(np.float64, copy=False), k=k, workers=1)
        if k == 1:
            distances = distances[:, None]
            candidate_ids = candidate_ids[:, None]
        candidate_normals = proxy_normals[candidate_ids]
        normal_score = np.sum(candidate_normals * chunk_normals[:, None, :], axis=2)
        distance_score = distances.astype(np.float32, copy=False) / bbox_diag
        scores = normal_score - distance_score
        best_local = np.argmax(scores, axis=1)
        best_proxy_face_ids = candidate_ids[np.arange(candidate_ids.shape[0]), best_local].astype(np.int64, copy=False)
        best_proxy_triangles = proxy_triangles[best_proxy_face_ids]
        bary = compute_barycentric_coordinates_3d(chunk_centroids, best_proxy_triangles)
        uv_pixels = np.sum(proxy_face_uv_pixels[best_proxy_face_ids] * bary[:, :, None], axis=1).astype(np.float32, copy=False)

        rgb[start:end] = bilinear_sample_rgb(atlas_rgb, uv_pixels)
        confidence[start:end] = bilinear_sample_scalar(atlas_confidence, uv_pixels)
        disagreement[start:end] = bilinear_sample_scalar(atlas_disagreement, uv_pixels)
        support_ratio[start:end] = bilinear_sample_scalar(atlas_support_ratio, uv_pixels)
        support_count[start:end] = bilinear_sample_scalar(atlas_support_count, uv_pixels)
        proxy_face_id[start:end] = best_proxy_face_ids.astype(np.int32, copy=False)

    return {
        "rgb": rgb,
        "confidence": confidence,
        "disagreement": disagreement,
        "support_ratio": support_ratio,
        "support_count": support_count,
        "proxy_face_id": proxy_face_id,
    }


def build_preview_atlas_from_full_atlas(
    source_face_ids: np.ndarray,
    source_face_uv_pixels: np.ndarray,
    source_faces: np.ndarray,
    source_vertices: np.ndarray,
    atlas_maps: Dict[str, np.ndarray],
    chart_resolution: int,
    padding: int,
    max_atlas_resolution: int,
    min_chart_resolution: int,
) -> Optional[Dict[str, object]]:
    source_face_ids = np.asarray(source_face_ids, dtype=np.int64)
    if source_face_ids.size == 0:
        return None

    preview_layout = build_per_face_uv_atlas(
        mesh=trimesh.Trimesh(vertices=source_vertices, faces=source_faces[source_face_ids], process=False),
        face_ids=np.arange(source_face_ids.shape[0], dtype=np.int64),
        desired_chart_resolution=int(chart_resolution),
        padding=int(padding),
        max_atlas_resolution=int(max_atlas_resolution),
        min_chart_resolution=int(min_chart_resolution),
    )
    preview_raster = rasterize_atlas_texels(preview_layout["face_uv_pixels"])
    preview_face_local = preview_raster["texel_face_local"]
    preview_bary = preview_raster["texel_bary"]
    source_tri_uv = source_face_uv_pixels[source_face_ids][preview_face_local]
    sample_xy = np.sum(source_tri_uv * preview_bary[:, :, None], axis=1).astype(np.float32, copy=False)

    preview_consensus = {
        "consensus_rgb": bilinear_sample_rgb(atlas_maps["rgb"], sample_xy),
        "confidence": bilinear_sample_scalar(atlas_maps["confidence"], sample_xy),
        "disagreement": bilinear_sample_scalar(atlas_maps["disagreement"], sample_xy),
        "support_count": np.round(bilinear_sample_scalar(atlas_maps["support_count"].astype(np.float32, copy=False), sample_xy)).astype(np.int32, copy=False),
        "support_ratio": bilinear_sample_scalar(atlas_maps["support_ratio"], sample_xy),
        "mode_count": np.round(bilinear_sample_scalar(atlas_maps["mode_count"].astype(np.float32, copy=False), sample_xy)).astype(np.int32, copy=False),
        "exemplar_view_id": np.round(bilinear_sample_scalar(atlas_maps["exemplar_view_id"].astype(np.float32, copy=False), sample_xy)).astype(np.int32, copy=False),
        "valid_mask": bilinear_sample_scalar(atlas_maps["valid_mask"].astype(np.float32, copy=False), sample_xy) > 0.5,
    }
    preview_maps = assemble_atlas_maps(
        atlas_width=int(preview_layout["atlas_width"]),
        atlas_height=int(preview_layout["atlas_height"]),
        texel_x=preview_raster["texel_x"],
        texel_y=preview_raster["texel_y"],
        consensus=preview_consensus,
    )
    return {
        "layout": preview_layout,
        "raster": preview_raster,
        "maps": preview_maps,
        "face_ids": source_face_ids,
    }


def save_rgb_png(path: Path, image_hw3: np.ndarray):
    array = np.round(np.clip(image_hw3, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def save_rgba_png(path: Path, image_hw4: np.ndarray):
    array = np.round(np.clip(image_hw4, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGBA").save(path)


def save_gray_png(path: Path, image_hw: np.ndarray, scale_max: Optional[float] = None):
    image = np.asarray(image_hw, dtype=np.float32)
    if scale_max is None or scale_max <= 0:
        finite = image[np.isfinite(image)]
        scale_max = float(np.max(finite)) if finite.size > 0 else 1.0
    scale_max = max(float(scale_max), 1e-8)
    array = np.round(np.clip(image / scale_max, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def write_textured_obj(
    obj_path: Path,
    mtl_path: Path,
    texture_name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_uv_norm: np.ndarray,
):
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl atlas_material\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.000000\n")
        f.write(f"map_Kd {texture_name}\n")

    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("o uv_fusion_mesh\n")

        obj_vertex_counter = 1
        obj_uv_counter = 1
        for local_face_idx in range(faces.shape[0]):
            tri_vertices = vertices[faces[local_face_idx]]
            tri_uv = face_uv_norm[local_face_idx]
            for vertex in tri_vertices:
                f.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
            for uv in tri_uv:
                f.write(f"vt {uv[0]:.9f} {uv[1]:.9f}\n")
            if local_face_idx == 0:
                f.write("usemtl atlas_material\n")
            f.write(
                "f "
                f"{obj_vertex_counter}/{obj_uv_counter} "
                f"{obj_vertex_counter + 1}/{obj_uv_counter + 1} "
                f"{obj_vertex_counter + 2}/{obj_uv_counter + 2}\n"
            )
            obj_vertex_counter += 3
            obj_uv_counter += 3


def build_parser():
    parser = ArgumentParser(
        description=(
            "Prepare UV-atlas multiview SR consensus assets. "
            "If the mesh has no UV, this script creates a per-face atlas fallback, "
            "projects multiview SR images into atlas texels, and keeps the dominant consensus mode. "
            "Camera parameters are read from --camera_source if provided, otherwise from the parent of --prior_dir."
        )
    )
    parser.add_argument("--camera_source", "--source_path", "-s", dest="camera_source", type=str, default=None)
    parser.add_argument("--images", "-i", type=str, default="images")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--blender_extension", type=str, default=".png")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--view_limit", type=int, default=0)
    parser.add_argument("--face_selection", choices=["all_faces", "payload"], default="all_faces")
    parser.add_argument("--face_payload", type=str, default=None)
    parser.add_argument("--face_payload_key", type=str, default="roi_face_ids")
    parser.add_argument("--face_stride", type=int, default=1)
    parser.add_argument("--max_faces", type=int, default=0)
    parser.add_argument("--face_sample_mode", choices=["prefix", "random"], default="prefix")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--proxy_trigger_faces", type=int, default=200000)
    parser.add_argument("--proxy_target_faces", type=int, default=8000)
    parser.add_argument("--proxy_search_steps", type=int, default=8)
    parser.add_argument("--disable_proxy", action="store_true")
    parser.add_argument("--proxy_visibility_mode", choices=["face_center", "triangle_raster"], default="face_center")
    parser.add_argument("--atlas_chart_resolution", type=int, default=12)
    parser.add_argument("--atlas_padding", type=int, default=2)
    parser.add_argument("--atlas_max_resolution", type=int, default=4096)
    parser.add_argument("--atlas_min_chart_resolution", type=int, default=4)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--visibility_depth_tolerance", type=float, default=0.03)
    parser.add_argument("--disable_front_facing_only", action="store_true")
    parser.add_argument("--consensus_color_threshold", type=float, default=0.08)
    parser.add_argument("--consensus_min_views", type=int, default=2)
    parser.add_argument("--density_weight_min", type=float, default=0.5)
    parser.add_argument("--density_weight_max", type=float, default=2.0)
    parser.add_argument("--debug_preview_faces", type=int, default=128)
    parser.add_argument("--debug_preview_chart_resolution", type=int, default=48)
    parser.add_argument("--export_full_uv_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    np.random.seed(int(args.random_seed))

    mesh_obj = load_triangle_mesh(args.mesh_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_source_root = resolve_camera_source_root(args)

    train_cameras = load_train_cameras_only(args)
    prior_index = index_image_dir(args.prior_dir)

    selected_face_ids, selected_face_source = select_face_ids(mesh_obj, args)
    if selected_face_ids.size == 0:
        raise RuntimeError("No mesh faces selected for UV fusion.")

    original_vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    original_faces_all = np.asarray(mesh_obj.faces, dtype=np.int64)
    selected_faces_original = original_faces_all[selected_face_ids]
    selected_mesh = trimesh.Trimesh(
        vertices=original_vertices,
        faces=selected_faces_original,
        process=False,
    )
    selected_mesh.remove_unreferenced_vertices()

    proxy_result = build_proxy_mesh_if_needed(selected_mesh, args)
    if proxy_result is not None:
        working_mesh = proxy_result.mesh
        working_face_ids = np.arange(len(working_mesh.faces), dtype=np.int64)
        working_face_source = (
            f"{selected_face_source}_proxy_vertex_cluster_target_{proxy_result.target_faces}"
        )
        visibility_mode = str(args.proxy_visibility_mode)
        print(
            "Using proxy mesh for UV fusion: "
            f"original_faces={proxy_result.original_face_count}, "
            f"proxy_faces={len(working_mesh.faces)}, "
            f"pitch={proxy_result.pitch:.6f}, "
            f"visibility_mode={visibility_mode}"
        )
    else:
        working_mesh = selected_mesh
        working_face_ids = np.arange(len(working_mesh.faces), dtype=np.int64)
        working_face_source = selected_face_source
        visibility_mode = "triangle_raster"

    atlas_layout = build_per_face_uv_atlas(
        mesh=working_mesh,
        face_ids=working_face_ids,
        desired_chart_resolution=int(args.atlas_chart_resolution),
        padding=int(args.atlas_padding),
        max_atlas_resolution=int(args.atlas_max_resolution),
        min_chart_resolution=int(args.atlas_min_chart_resolution),
    )
    atlas_raster = rasterize_atlas_texels(atlas_layout["face_uv_pixels"])

    vertices = np.asarray(working_mesh.vertices, dtype=np.float32)
    faces_all = np.asarray(working_mesh.faces, dtype=np.int64)
    selected_faces = faces_all[working_face_ids]
    selected_triangles = vertices[selected_faces]
    texel_face_local = atlas_raster["texel_face_local"]
    texel_bary = atlas_raster["texel_bary"]
    texel_world = np.sum(selected_triangles[texel_face_local] * texel_bary[:, :, None], axis=1).astype(np.float32, copy=False)
    texel_normals = np.asarray(working_mesh.face_normals, dtype=np.float32)[working_face_ids][texel_face_local]

    consensus_state = initialize_consensus_state(num_texels=int(texel_face_local.shape[0]))
    view_summaries = []
    processed_views = 0
    missing_prior_views = []

    view_iter = tqdm(train_cameras, desc="Projecting views to UV", disable=bool(args.quiet), dynamic_ncols=True)
    for view_index, view in enumerate(view_iter):
        if args.view_limit > 0 and processed_views >= int(args.view_limit):
            break
        prior_path = lookup_indexed_image(prior_index, view.image_name)
        if prior_path is None:
            if len(missing_prior_views) < 32:
                missing_prior_views.append(str(view.image_name))
            continue

        prior_image = load_rgb_image_np(prior_path)
        face_buffers = build_view_face_buffers(
            mesh=working_mesh,
            triangles=selected_triangles,
            face_ids=working_face_ids,
            face_texel_count=atlas_raster["face_texel_count"],
            view=view,
            depth_min=float(args.depth_min),
            front_facing_only=not bool(args.disable_front_facing_only),
            density_weight_min=float(args.density_weight_min),
            density_weight_max=float(args.density_weight_max),
            visibility_mode=visibility_mode,
        )

        projected, valid = project_points_camera(view, texel_world, depth_min=float(args.depth_min), margin=0)
        pix_x = np.round(projected[:, 0]).astype(np.int64)
        pix_y = np.round(projected[:, 1]).astype(np.int64)
        in_bounds = (
            (pix_x >= 0)
            & (pix_x < int(view.image_width))
            & (pix_y >= 0)
            & (pix_y < int(view.image_height))
        )
        valid &= in_bounds

        cam_center = camera_center_numpy(view)
        view_dir = cam_center[None, :] - texel_world
        view_dir_norm = view_dir / np.clip(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-6, None)
        texel_frontality = np.sum(texel_normals * view_dir_norm, axis=1).astype(np.float32, copy=False)
        valid &= texel_frontality > 0.0

        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            view_summaries.append(
                {
                    "image_name": normalize_image_name(str(view.image_name)),
                    "prior_path": str(prior_path),
                    "visible_texels": 0,
                    "valid_samples": 0,
                    "front_visible_faces": int(np.sum(face_buffers["valid_faces"])),
                }
            )
            processed_views += 1
            continue

        if visibility_mode == "triangle_raster":
            visible_face_ids = face_buffers["face_buffer"][pix_y[valid_ids], pix_x[valid_ids]]
            visible_depth = face_buffers["depth_buffer"][pix_y[valid_ids], pix_x[valid_ids]]
            face_match = visible_face_ids == texel_face_local[valid_ids]
            depth_match = np.abs(visible_depth - projected[valid_ids, 2]) <= float(args.visibility_depth_tolerance)
            valid_ids = valid_ids[face_match & depth_match]
        else:
            valid_ids = valid_ids[face_buffers["valid_faces"][texel_face_local[valid_ids]]]
        if valid_ids.size == 0:
            view_summaries.append(
                {
                    "image_name": normalize_image_name(str(view.image_name)),
                    "prior_path": str(prior_path),
                    "visible_texels": 0,
                    "valid_samples": 0,
                    "front_visible_faces": int(np.sum(face_buffers["valid_faces"])),
                }
            )
            processed_views += 1
            continue

        sampled_rgb = bilinear_sample_rgb(prior_image, projected[valid_ids, :2])
        sample_weight = texel_frontality[valid_ids] * face_buffers["face_weight"][texel_face_local[valid_ids]]
        sample_weight = np.clip(sample_weight.astype(np.float32, copy=False), 1e-4, None)
        update_consensus_state(
            state=consensus_state,
            texel_ids=valid_ids.astype(np.int64, copy=False),
            sample_rgb=sampled_rgb,
            sample_weight=sample_weight,
            view_id=int(view_index),
            color_threshold=float(args.consensus_color_threshold),
        )

        view_summaries.append(
            {
                "image_name": normalize_image_name(str(view.image_name)),
                "prior_path": str(prior_path),
                "visible_texels": int(valid_ids.size),
                "valid_samples": int(valid_ids.size),
                "front_visible_faces": int(np.sum(face_buffers["valid_faces"])),
                "projected_area_visible_mean": (
                    float(np.mean(face_buffers["projected_area"][face_buffers["valid_faces"]]))
                    if np.any(face_buffers["valid_faces"])
                    else 0.0
                ),
            }
        )
        processed_views += 1
        view_iter.set_postfix(views=processed_views, texels=int(valid_ids.size))

    consensus = finalize_consensus_state(
        state=consensus_state,
        min_views=int(args.consensus_min_views),
    )
    atlas_maps = assemble_atlas_maps(
        atlas_width=int(atlas_layout["atlas_width"]),
        atlas_height=int(atlas_layout["atlas_height"]),
        texel_x=atlas_raster["texel_x"],
        texel_y=atlas_raster["texel_y"],
        consensus=consensus,
    )
    face_metrics = compute_face_preview_scores(
        texel_face_local=atlas_raster["texel_face_local"],
        consensus=consensus,
        num_faces=int(selected_faces.shape[0]),
    )
    preview_face_ids = select_preview_face_ids(
        face_metrics=face_metrics,
        max_preview_faces=int(args.debug_preview_faces),
    )

    save_rgb_png(output_dir / "uv_consensus_rgb_v0.png", atlas_maps["rgb"])
    save_rgba_png(output_dir / "uv_consensus_rgba_v0.png", atlas_maps["rgba"])
    save_gray_png(output_dir / "uv_confidence_v0.png", atlas_maps["confidence"], scale_max=1.0)
    save_gray_png(output_dir / "uv_disagreement_v0.png", atlas_maps["disagreement"], scale_max=1.0)
    save_gray_png(
        output_dir / "uv_support_count_v0.png",
        atlas_maps["support_count"].astype(np.float32),
        scale_max=float(max(np.max(atlas_maps["support_count"]), 1)),
    )
    save_gray_png(
        output_dir / "uv_support_ratio_v0.png",
        atlas_maps["support_ratio"],
        scale_max=1.0,
    )

    atlas_payload_path = output_dir / "uv_fusion_payload_v0.npz"
    np.savez_compressed(
        atlas_payload_path,
        atlas_width=np.asarray([atlas_layout["atlas_width"]], dtype=np.int32),
        atlas_height=np.asarray([atlas_layout["atlas_height"]], dtype=np.int32),
        original_selected_face_ids=selected_face_ids.astype(np.int64, copy=False),
        working_face_ids=working_face_ids.astype(np.int64, copy=False),
        face_uv_norm=atlas_layout["face_uv_norm"].astype(np.float32, copy=False),
        face_uv_pixels=atlas_layout["face_uv_pixels"].astype(np.float32, copy=False),
        chart_resolution=np.asarray([atlas_layout["chart_resolution"]], dtype=np.int32),
        chart_padding=np.asarray([atlas_layout["padding"]], dtype=np.int32),
        texel_x=atlas_raster["texel_x"].astype(np.int32, copy=False),
        texel_y=atlas_raster["texel_y"].astype(np.int32, copy=False),
        texel_face_local=atlas_raster["texel_face_local"].astype(np.int32, copy=False),
        texel_bary=atlas_raster["texel_bary"].astype(np.float32, copy=False),
        face_texel_count=atlas_raster["face_texel_count"].astype(np.int32, copy=False),
        consensus_rgb=consensus["consensus_rgb"].astype(np.float32, copy=False),
        confidence=consensus["confidence"].astype(np.float32, copy=False),
        disagreement=consensus["disagreement"].astype(np.float32, copy=False),
        support_count=consensus["support_count"].astype(np.int32, copy=False),
        support_ratio=consensus["support_ratio"].astype(np.float32, copy=False),
        mode_count=consensus["mode_count"].astype(np.int32, copy=False),
        exemplar_view_id=consensus["exemplar_view_id"].astype(np.int32, copy=False),
        valid_mask=consensus["valid_mask"].astype(bool, copy=False),
    )

    preview_artifacts = build_preview_atlas_from_full_atlas(
        source_face_ids=preview_face_ids,
        source_face_uv_pixels=atlas_layout["face_uv_pixels"],
        source_faces=selected_faces,
        source_vertices=vertices,
        atlas_maps=atlas_maps,
        chart_resolution=int(args.debug_preview_chart_resolution),
        padding=int(args.atlas_padding),
        max_atlas_resolution=int(args.atlas_max_resolution),
        min_chart_resolution=int(args.atlas_min_chart_resolution),
    )

    obj_path = output_dir / "uv_fusion_mesh_v0.obj"
    mtl_path = output_dir / "uv_fusion_mesh_v0.mtl"
    preview_rgb_path = None
    preview_confidence_path = None
    preview_disagreement_path = None
    if preview_artifacts is not None:
        preview_rgb_path = output_dir / "uv_preview_consensus_rgb_v0.png"
        preview_confidence_path = output_dir / "uv_preview_confidence_v0.png"
        preview_disagreement_path = output_dir / "uv_preview_disagreement_v0.png"
        save_rgb_png(preview_rgb_path, preview_artifacts["maps"]["rgb"])
        save_gray_png(preview_confidence_path, preview_artifacts["maps"]["confidence"], scale_max=1.0)
        save_gray_png(preview_disagreement_path, preview_artifacts["maps"]["disagreement"], scale_max=1.0)
        write_textured_obj(
            obj_path=obj_path,
            mtl_path=mtl_path,
            texture_name=preview_rgb_path.name,
            vertices=vertices,
            faces=selected_faces[preview_face_ids],
            face_uv_norm=preview_artifacts["layout"]["face_uv_norm"],
        )
    elif bool(args.export_full_uv_mesh):
        write_textured_obj(
            obj_path=obj_path,
            mtl_path=mtl_path,
            texture_name="uv_consensus_rgb_v0.png",
            vertices=vertices,
            faces=selected_faces,
            face_uv_norm=atlas_layout["face_uv_norm"],
        )

    full_obj_path = None
    full_mtl_path = None
    if bool(args.export_full_uv_mesh) and preview_artifacts is not None:
        full_obj_path = output_dir / "uv_fusion_mesh_full_v0.obj"
        full_mtl_path = output_dir / "uv_fusion_mesh_full_v0.mtl"
        write_textured_obj(
            obj_path=full_obj_path,
            mtl_path=full_mtl_path,
            texture_name="uv_consensus_rgb_v0.png",
            vertices=vertices,
            faces=selected_faces,
            face_uv_norm=atlas_layout["face_uv_norm"],
        )

    full_face_overlay_path = None
    if proxy_result is not None:
        transferred = transfer_proxy_consensus_to_full_faces(
            original_mesh=selected_mesh,
            proxy_mesh=working_mesh,
            proxy_face_uv_pixels=atlas_layout["face_uv_pixels"],
            atlas_maps=atlas_maps,
        )
        full_face_overlay_path = output_dir / "full_mesh_face_overlay_v0.npz"
        np.savez_compressed(
            full_face_overlay_path,
            original_face_ids=selected_face_ids.astype(np.int64, copy=False),
            proxy_face_ids=transferred["proxy_face_id"].astype(np.int32, copy=False),
            rgb=transferred["rgb"].astype(np.float32, copy=False),
            confidence=transferred["confidence"].astype(np.float32, copy=False),
            disagreement=transferred["disagreement"].astype(np.float32, copy=False),
            support_ratio=transferred["support_ratio"].astype(np.float32, copy=False),
            support_count=transferred["support_count"].astype(np.float32, copy=False),
        )

    summary = {
        "mode": "prepare_uv_fusion_v0",
        "camera_source": str(camera_source_root),
        "mesh_path": str(Path(args.mesh_path).expanduser().resolve()),
        "prior_dir": str(Path(args.prior_dir).expanduser().resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_face_count": int(selected_face_ids.shape[0]),
        "selected_face_ratio_vs_mesh": float(selected_face_ids.shape[0] / max(len(mesh_obj.faces), 1)),
        "selected_face_source": selected_face_source,
        "working_face_count": int(selected_faces.shape[0]),
        "working_face_source": working_face_source,
        "views_processed": int(processed_views),
        "missing_prior_view_samples": missing_prior_views,
        "atlas": {
            "width": int(atlas_layout["atlas_width"]),
            "height": int(atlas_layout["atlas_height"]),
            "chart_resolution": int(atlas_layout["chart_resolution"]),
            "padding": int(atlas_layout["padding"]),
            "occupied_texels": int(atlas_raster["texel_x"].shape[0]),
            "occupied_texel_ratio": float(
                atlas_raster["texel_x"].shape[0] / max(int(atlas_layout["atlas_width"]) * int(atlas_layout["atlas_height"]), 1)
            ),
        },
        "outputs": {
            "consensus_rgb_png": str((output_dir / "uv_consensus_rgb_v0.png").resolve()),
            "consensus_rgba_png": str((output_dir / "uv_consensus_rgba_v0.png").resolve()),
            "confidence_png": str((output_dir / "uv_confidence_v0.png").resolve()),
            "disagreement_png": str((output_dir / "uv_disagreement_v0.png").resolve()),
            "support_count_png": str((output_dir / "uv_support_count_v0.png").resolve()),
            "support_ratio_png": str((output_dir / "uv_support_ratio_v0.png").resolve()),
            "atlas_payload_npz": str(atlas_payload_path.resolve()),
        },
        "consensus_stats": {
            "confidence": stats_from_array(consensus["confidence"][consensus["valid_mask"]]),
            "disagreement": stats_from_array(consensus["disagreement"][consensus["valid_mask"]]),
            "support_count": stats_from_array(consensus["support_count"][consensus["valid_mask"]].astype(np.float32)),
            "support_ratio": stats_from_array(consensus["support_ratio"][consensus["valid_mask"]]),
            "mode_count": stats_from_array(consensus["mode_count"][consensus["valid_mask"]].astype(np.float32)),
        },
        "parameters": {
            "depth_min": float(args.depth_min),
            "visibility_depth_tolerance": float(args.visibility_depth_tolerance),
            "front_facing_only": not bool(args.disable_front_facing_only),
            "consensus_color_threshold": float(args.consensus_color_threshold),
            "consensus_min_views": int(args.consensus_min_views),
            "density_weight_min": float(args.density_weight_min),
            "density_weight_max": float(args.density_weight_max),
            "visibility_mode": visibility_mode,
            "debug_preview_faces": int(args.debug_preview_faces),
            "debug_preview_chart_resolution": int(args.debug_preview_chart_resolution),
            "export_full_uv_mesh": bool(args.export_full_uv_mesh),
        },
        "view_summaries": view_summaries,
        "note": (
            "This v0 pipeline uses a per-face UV atlas fallback because the input mesh has no UV. "
            "Each atlas texel keeps the dominant multiview SR color mode instead of naive averaging. "
            "The default OBJ preview is now a sparse high-confidence subset for clearer debugging; use --export_full_uv_mesh to additionally export the full working UV mesh."
        ),
    }
    if preview_artifacts is not None and preview_rgb_path is not None and preview_confidence_path is not None and preview_disagreement_path is not None:
        summary["preview"] = {
            "face_count": int(preview_face_ids.shape[0]),
            "chart_resolution": int(preview_artifacts["layout"]["chart_resolution"]),
            "atlas_width": int(preview_artifacts["layout"]["atlas_width"]),
            "atlas_height": int(preview_artifacts["layout"]["atlas_height"]),
        }
        summary["outputs"]["uv_mesh_obj"] = str(obj_path.resolve())
        summary["outputs"]["uv_mesh_mtl"] = str(mtl_path.resolve())
        summary["outputs"]["preview_consensus_rgb_png"] = str(preview_rgb_path.resolve())
        summary["outputs"]["preview_confidence_png"] = str(preview_confidence_path.resolve())
        summary["outputs"]["preview_disagreement_png"] = str(preview_disagreement_path.resolve())
    else:
        summary["preview"] = {"face_count": 0}
        if bool(args.export_full_uv_mesh) and obj_path.exists() and mtl_path.exists():
            summary["outputs"]["uv_mesh_obj"] = str(obj_path.resolve())
            summary["outputs"]["uv_mesh_mtl"] = str(mtl_path.resolve())
    if proxy_result is not None:
        summary["proxy"] = {
            "enabled": True,
            "original_face_count": int(proxy_result.original_face_count),
            "proxy_face_count": int(len(working_mesh.faces)),
            "proxy_vertex_count": int(len(working_mesh.vertices)),
            "proxy_pitch": float(proxy_result.pitch),
            "target_faces": int(proxy_result.target_faces),
        }
        if full_face_overlay_path is not None:
            summary["outputs"]["full_mesh_face_overlay_npz"] = str(full_face_overlay_path.resolve())
    else:
        summary["proxy"] = {"enabled": False}
    if full_obj_path is not None and full_mtl_path is not None:
        summary["outputs"]["uv_mesh_full_obj"] = str(full_obj_path.resolve())
        summary["outputs"]["uv_mesh_full_mtl"] = str(full_mtl_path.resolve())
    summary_path = output_dir / "prepare_uv_fusion_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

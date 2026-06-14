import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from scene.dataset_readers import (
    CameraInfo,
    SceneInfo,
    fetchPly,
    getNerfppNorm,
    storePly,
)
from utils.graphics_utils import focal2fov, fov2focal
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


def _resolve_image_path(root_path, file_path, extension=".png"):
    candidates = []
    file_path = file_path.replace("\\", "/")

    if os.path.isabs(file_path):
        candidates.append(file_path)
    else:
        candidates.append(os.path.join(root_path, file_path))

    if not Path(candidates[-1]).suffix:
        candidates.append(candidates[-1] + extension)

    images_candidate = os.path.join(root_path, "images", os.path.basename(file_path))
    candidates.append(images_candidate)
    if not Path(images_candidate).suffix:
        candidates.append(images_candidate + extension)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not locate image for frame path '{file_path}'")


def _compose_rgb_with_background(image, white_background):
    im_data = np.array(image.convert("RGBA"), dtype=np.float32) / 255.0
    bg = np.array([1.0, 1.0, 1.0], dtype=np.float32) if white_background else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rgb = im_data[:, :, :3] * im_data[:, :, 3:4] + bg * (1 - im_data[:, :, 3:4])
    return Image.fromarray(np.array(rgb * 255.0, dtype=np.uint8), "RGB")


def _infer_fovs(frame, meta, width, height):
    fx = frame.get("fl_x", meta.get("fl_x"))
    fy = frame.get("fl_y", meta.get("fl_y", fx))

    if fx is not None and fy is not None:
        return focal2fov(float(fx), width), focal2fov(float(fy), height)

    fovx = frame.get("camera_angle_x", meta.get("camera_angle_x"))
    if fovx is None:
        raise ValueError("transforms.json must provide fl_x/fl_y or camera_angle_x")

    fovx = float(fovx)
    fovy = focal2fov(fov2focal(fovx, width), height)
    return fovx, fovy


def _build_cam_infos(root_path, meta, white_background, extension=".png"):
    cam_infos = []
    frames = meta.get("frames", [])

    for idx, frame in enumerate(frames):
        image_path = _resolve_image_path(root_path, frame["file_path"], extension=extension)
        image_name = Path(image_path).stem
        image = Image.open(image_path)
        image = _compose_rgb_with_background(image, white_background=white_background)

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w = c2w.copy()
        # OpenGL/Blender to COLMAP convention.
        c2w[:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        fovx, fovy = _infer_fovs(frame, meta, image.size[0], image.size[1])

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=fovy,
                FovX=fovx,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        )

    return cam_infos


def _ensure_point_cloud(root_path, meta):
    ply_path = os.path.join(root_path, "points3d.ply")
    if os.path.exists(ply_path):
        try:
            return fetchPly(ply_path), ply_path
        except Exception:
            pass

    num_pts = 100_000
    radius = float(meta.get("sphere_radius", 1.3))
    center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)

    xyz = np.random.random((num_pts, 3)).astype(np.float32) * (2.0 * radius) - radius
    xyz += center[None, :]
    shs = np.random.random((num_pts, 3)).astype(np.float32) / 255.0

    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3), dtype=np.float32))
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    return pcd, ply_path


def read_transforms_json_scene(path, white_background, eval_mode, llffhold=8, extension=".png"):
    transforms_path = os.path.join(path, "transforms.json")
    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cam_infos = _build_cam_infos(path, meta, white_background=white_background, extension=extension)

    if eval_mode:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if len(train_cam_infos) == 0:
        raise RuntimeError("No training cameras found after split for transforms.json")

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd, ply_path = _ensure_point_cloud(path, meta)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

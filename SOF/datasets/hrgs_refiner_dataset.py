from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams
from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.hrgs_scene_layout import build_hrgs_view_records, resolve_hrgs_scene_layout
from utils.prior_injection import load_rgb_image, normalize_image_name
from utils.vggt_adapter import FrozenVGGTAdapter, VGGTAdapterConfig
from utils.visibility_records import VisibilityRecordConfig, build_coarse_visibility_records


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, resolution: int | float) -> Any:
    class _Args:
        pass

    args = _Args()
    args.sh_degree = 3
    args.source_path = scene_root
    args.model_path = model_path
    args.images = images_subdir
    args.resolution = resolution
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    args.alpha_mask = False
    args.init_type = "sfm"
    return args


def _build_camera_index(cameras: Sequence[object]) -> Dict[str, object]:
    return {normalize_image_name(cam.image_name): cam for cam in cameras}


def _resize_image_hwc(image_hwc: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    target_h, target_w = target_hw
    image = image_hwc.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    resized = F.interpolate(image, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).contiguous()


def _load_image_for_view(path: str, target_hw: Tuple[int, int]) -> torch.Tensor:
    image = load_rgb_image(Path(path))
    if tuple(image.shape[:2]) != tuple(target_hw):
        image = _resize_image_hwc(image, target_hw)
    return image


def _build_camera_bundle(cameras: Sequence[object], device: str) -> Dict[str, torch.Tensor]:
    intrinsics = []
    world_to_view = []
    cam_to_world = []
    for cam in cameras:
        K = torch.tensor(
            [
                [float(cam.focal_x), 0.0, float(cam.image_width) / 2.0],
                [0.0, float(cam.focal_y), float(cam.image_height) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        w2v = cam.world_view_transform.transpose(0, 1).detach().to(dtype=torch.float32).cpu()
        c2w = torch.linalg.inv(w2v)
        intrinsics.append(K)
        world_to_view.append(w2v)
        cam_to_world.append(c2w)
    return {
        "intrinsics": torch.stack(intrinsics, dim=0),
        "world_to_view": torch.stack(world_to_view, dim=0),
        "cam_to_world": torch.stack(cam_to_world, dim=0),
    }


def _nested_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nested_to_cpu(item) for item in value]
    return value


def _nested_to_device(value, device: str):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _nested_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_nested_to_device(item, device) for item in value]
    return value


def load_oracle_depth(oracle_scene_dir: str, frame_name: str) -> np.ndarray | None:
    candidates = [
        Path(oracle_scene_dir) / f"{frame_name}.npy",
        Path(oracle_scene_dir) / "train" / "ours_30000" / "depth" / f"{frame_name}.npy",
        Path(oracle_scene_dir) / "depth" / f"{frame_name}.npy",
    ]
    for npy_path in candidates:
        if npy_path.exists():
            return np.load(str(npy_path)).astype(np.float32)

    try:
        import OpenEXR
        import Imath

        exr_path = Path(oracle_scene_dir) / "train" / "ours_30000" / "depth" / f"{frame_name}.exr"
        if exr_path.exists():
            f = OpenEXR.InputFile(str(exr_path))
            dw = f.header()["dataWindow"]
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            ch = f.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
            return np.frombuffer(ch, dtype=np.float32).reshape(size[1], size[0])
    except ImportError:
        pass
    return None


def oracle_depth_exists(oracle_scene_dir: str, frame_name: str) -> bool:
    candidates = [
        Path(oracle_scene_dir) / f"{frame_name}.npy",
        Path(oracle_scene_dir) / "train" / "ours_30000" / "depth" / f"{frame_name}.npy",
        Path(oracle_scene_dir) / "depth" / f"{frame_name}.npy",
        Path(oracle_scene_dir) / "train" / "ours_30000" / "depth" / f"{frame_name}.exr",
    ]
    return any(path.exists() for path in candidates)


def _load_resized_oracle_depth(oracle_scene_dir: str, frame_name: str, target_hw: Tuple[int, int]) -> torch.Tensor:
    depth_np = load_oracle_depth(oracle_scene_dir, frame_name)
    if depth_np is None:
        raise FileNotFoundError(f"Oracle depth not found for view '{frame_name}' under {oracle_scene_dir}")
    if depth_np.ndim != 2:
        depth_np = np.squeeze(depth_np)
    depth = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0)
    depth = F.interpolate(depth, size=target_hw, mode="bilinear", align_corners=False)
    return depth[0]


@dataclass
class HRGSRefinerSceneSpec:
    scene_root: str
    gs_model_path: str
    oracle_root: str
    scene_id: str | None = None
    source_images_subdir: str = "images_8"
    target_images_subdir: str = "images_2"
    priors_dir: str | None = None
    load_iteration: int = -1


@dataclass
class HRGSRefinerDatasetConfig:
    vggt_root: str
    device: str = "cuda"
    num_views: int = 2
    samples_per_scene: int = 16
    camera_resolution: int | float = 2
    require_priors: bool = True
    seed: int = 0
    cache_samples: bool = True
    cache_dir: str | None = None
    visibility_downsample: int = 8
    visibility_topk: int = 4
    visibility_max_visible: int = 30000
    visibility_max_patch_radius: int = 1


@dataclass
class _SceneContext:
    spec: HRGSRefinerSceneSpec
    scene: Scene
    gaussians: GaussianModel
    layout: Any
    records: List[Dict[str, object]]
    camera_index: Dict[str, object]
    all_cameras: List[object]
    target_hw: Tuple[int, int]
    oracle_scene_dir: str


def load_scene_specs_from_json(path: str | Path) -> List[HRGSRefinerSceneSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Scene spec JSON must be a list of objects.")
    specs = []
    for item in payload:
        specs.append(HRGSRefinerSceneSpec(**item))
    return specs


class HRGSRefinerDataset(Dataset):
    def __init__(
        self,
        scene_specs: Sequence[HRGSRefinerSceneSpec],
        config: HRGSRefinerDatasetConfig,
    ) -> None:
        if not scene_specs:
            raise ValueError("scene_specs must not be empty.")
        self.scene_specs = list(scene_specs)
        self.config = config
        self.device = config.device
        self.vggt_adapter = FrozenVGGTAdapter(
            VGGTAdapterConfig(
                vggt_root=config.vggt_root,
                device=config.device,
            )
        )
        self.contexts = [self._build_context(spec) for spec in self.scene_specs]
        self.sample_index = [
            (scene_idx, sample_idx)
            for scene_idx in range(len(self.contexts))
            for sample_idx in range(int(self.config.samples_per_scene))
        ]
        self.sample_cache: Dict[int, Dict[str, Any]] = {}

    def _build_context(self, spec: HRGSRefinerSceneSpec) -> _SceneContext:
        scene_id = spec.scene_id or Path(spec.scene_root).name
        layout = resolve_hrgs_scene_layout(
            spec.scene_root,
            source_images_subdir=spec.source_images_subdir,
            target_images_subdir=spec.target_images_subdir,
            priors_dir=spec.priors_dir,
            vggt_root=self.config.vggt_root,
            repo_root=str(REPO_ROOT),
        )
        records = build_hrgs_view_records(layout, require_priors=bool(self.config.require_priors))
        if not records:
            raise RuntimeError(f"No matched view records for scene: {scene_id}")

        dataset_args = _build_dataset_args(
            scene_root=layout.scene_root,
            model_path=str(Path(spec.gs_model_path).expanduser().resolve()),
            images_subdir=layout.target_images_subdir,
            resolution=self.config.camera_resolution,
        )
        dataset = ModelParams(None).extract(dataset_args)
        gaussians = GaussianModel(dataset.sh_degree, use_SBs=False)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=spec.load_iteration,
            shuffle=False,
            skip_train=False,
            skip_test=True,
        )
        all_cameras = scene.getTrainCameras().copy()
        camera_index = _build_camera_index(all_cameras)
        oracle_scene_dir = str(Path(spec.oracle_root).expanduser().resolve())

        matched = []
        for record in records:
            image_name = normalize_image_name(str(record["image_name"]))
            if image_name not in camera_index:
                continue
            if not oracle_depth_exists(oracle_scene_dir, image_name):
                continue
            matched.append(record)
        if len(matched) < int(self.config.num_views):
            raise RuntimeError(
                f"Scene '{scene_id}' only has {len(matched)} views with priors+oracle, "
                f"need at least {self.config.num_views}."
            )

        gaussians.compute_3D_filter(all_cameras, CUDA=False)
        any_camera = camera_index[str(matched[0]["image_name"])]
        target_hw = (int(any_camera.image_height), int(any_camera.image_width))

        spec.scene_id = scene_id
        return _SceneContext(
            spec=spec,
            scene=scene,
            gaussians=gaussians,
            layout=layout,
            records=matched,
            camera_index=camera_index,
            all_cameras=all_cameras,
            target_hw=target_hw,
            oracle_scene_dir=oracle_scene_dir,
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def _choose_records(self, context: _SceneContext, sample_idx: int) -> List[Dict[str, object]]:
        if len(context.records) <= int(self.config.num_views):
            return list(context.records[: int(self.config.num_views)])
        stable_scene_seed = sum(ord(ch) for ch in str(context.spec.scene_id))
        rng = np.random.default_rng(int(self.config.seed) + sample_idx + 1000 * stable_scene_seed)
        chosen = rng.choice(len(context.records), size=int(self.config.num_views), replace=False)
        selected = [context.records[int(idx)] for idx in chosen.tolist()]
        selected.sort(key=lambda item: str(item["image_name"]))
        return selected

    @torch.no_grad()
    def _build_sample(self, index: int) -> Dict[str, Any]:
        scene_idx, sample_idx = self.sample_index[index]
        context = self.contexts[scene_idx]
        selected = self._choose_records(context, sample_idx)
        selected_cameras = [context.camera_index[normalize_image_name(str(record["image_name"]))] for record in selected]
        target_hw = context.target_hw
        background = torch.zeros((3,), dtype=torch.float32, device=self.device)

        render_pkgs = []
        render_rgb = []
        render_depth = []
        render_normal = []
        render_alpha = []
        render_diag = []
        sr_images = []
        lr_up_images = []
        lr_native = []
        oracle_depths = []
        valid_depths = []

        for record, camera in zip(selected, selected_cameras):
            render_pkg = render_simple(camera, context.gaussians, background)
            render_pkgs.append(render_pkg)
            render_rgb.append(render_pkg["render"].detach().to(device=self.device))
            render_depth.append(render_pkg["depth"].detach().to(device=self.device))
            render_normal.append(render_pkg["normal"].detach().to(device=self.device))
            render_alpha.append(render_pkg["alpha"].detach().to(device=self.device))
            diag = torch.cat(
                [
                    render_pkg["distortion"].detach(),
                    render_pkg["alpha"].detach(),
                    torch.clamp(render_pkg["depth"].detach(), min=0.0)
                    / torch.clamp(render_pkg["depth"].detach().amax(), min=1.0),
                ],
                dim=0,
            )
            render_diag.append(diag.to(device=self.device))

            view_hw = (int(camera.image_height), int(camera.image_width))
            sr_path = str(record["prior_path"] or record["target_path"])
            sr_image = _load_image_for_view(sr_path, view_hw)
            lr_up_image = _load_image_for_view(str(record["lr_path"]), view_hw)
            lr_native_image = load_rgb_image(Path(str(record["lr_path"]))).permute(2, 0, 1).contiguous()
            sr_images.append(sr_image.permute(2, 0, 1).contiguous())
            lr_up_images.append(lr_up_image.permute(2, 0, 1).contiguous())
            lr_native.append(lr_native_image)

            oracle_depth = _load_resized_oracle_depth(context.oracle_scene_dir, str(record["image_name"]), view_hw)
            oracle_depths.append(oracle_depth)
            valid_depths.append(torch.isfinite(oracle_depth) & (oracle_depth > 1e-5))

        sr_images_t = torch.stack(sr_images, dim=0).to(device=self.device)
        lr_up_images_t = torch.stack(lr_up_images, dim=0).to(device=self.device)
        lr_native_t = torch.stack(lr_native, dim=0).to(device=self.device)
        oracle_depth_t = torch.stack(oracle_depths, dim=0).to(device=self.device)
        valid_depth_t = torch.stack(valid_depths, dim=0).to(device=self.device)

        camera_bundle = _build_camera_bundle(selected_cameras, device=self.device)
        cache_root = (
            Path(self.config.cache_dir).expanduser().resolve()
            if self.config.cache_dir
            else Path(context.spec.gs_model_path).expanduser().resolve() / "hrgs_refiner_cache"
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        vggt_cache_path = cache_root / f"{context.spec.scene_id}_sample{sample_idx:04d}_v{len(selected)}.pt"
        vggt_prior = self.vggt_adapter.run(
            lr_native_t.unsqueeze(0),
            target_hw=target_hw,
            image_names=[str(record["image_name"]) for record in selected],
            cache_path=vggt_cache_path,
        )

        visibility_records = build_coarse_visibility_records(
            context.gaussians,
            selected_cameras,
            render_pkgs,
            image_hw=target_hw,
            cfg=VisibilityRecordConfig(
                downsample=int(self.config.visibility_downsample),
                topk=int(self.config.visibility_topk),
                max_visible_per_view=int(self.config.visibility_max_visible),
                max_patch_radius=int(self.config.visibility_max_patch_radius),
            ),
        )

        sample = {
            "scene_id": context.spec.scene_id,
            "view_names": [str(record["image_name"]) for record in selected],
            "images": {
                "images_sr": sr_images_t,
                "images_lr_up": lr_up_images_t,
                "images_lr_native": lr_native_t,
            },
            "cameras": camera_bundle,
            "vggt_prior": vggt_prior,
            "lr_gs_buffers": {
                "render_rgb": torch.stack(render_rgb, dim=0),
                "depth": torch.stack(render_depth, dim=0),
                "normal": torch.stack(render_normal, dim=0),
                "alpha": torch.stack(render_alpha, dim=0),
                "diagnostics": torch.stack(render_diag, dim=0),
            },
            "visibility_records": visibility_records,
            "targets": {
                "oracle_depth": oracle_depth_t,
                "valid_depth": valid_depth_t.to(dtype=torch.bool),
                "surface_mask": valid_depth_t.to(dtype=torch.float32),
            },
            "meta": {
                "target_hw": list(target_hw),
                "num_gaussians": int(context.gaussians.get_xyz.shape[0]),
                "camera_resolution": self.config.camera_resolution,
                "cache_path": str(vggt_cache_path),
                "scene_spec": asdict(context.spec),
            },
        }
        return _nested_to_cpu(sample)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.config.cache_samples and index in self.sample_cache:
            return self.sample_cache[index]
        sample = self._build_sample(index)
        if self.config.cache_samples:
            self.sample_cache[index] = sample
        return sample


def collate_hrgs_refiner_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(items) != 1:
        raise ValueError("HRGSRefiner v0 trainer currently expects batch_size=1.")
    return items[0]

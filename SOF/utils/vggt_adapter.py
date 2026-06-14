from __future__ import annotations

import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class VGGTAdapterConfig:
    vggt_root: str
    device: str = "cuda"
    vggt_resolution: int = 518
    feature_down_ratio: int = 4
    frames_chunk_size: int = 8
    model_cache_name: str = "model.pt"


def _ensure_vggt_import_path(vggt_root: str | Path) -> Path:
    root = Path(vggt_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _resolve_autocast_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def _resize_intrinsics_to_hw(
    intrinsics: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> torch.Tensor:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    out = intrinsics.clone()
    out[..., 0, 0] *= float(dst_w) / float(src_w)
    out[..., 1, 1] *= float(dst_h) / float(src_h)
    out[..., 0, 2] *= float(dst_w) / float(src_w)
    out[..., 1, 2] *= float(dst_h) / float(src_h)
    return out


def _lift_extrinsic_to_world_to_view(extrinsic: torch.Tensor) -> torch.Tensor:
    if extrinsic.ndim != 4 or extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape [B, V, 3, 4], got {tuple(extrinsic.shape)}")
    b, v = extrinsic.shape[:2]
    world_to_view = torch.eye(4, device=extrinsic.device, dtype=extrinsic.dtype).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    world_to_view[..., :3, :4] = extrinsic
    return world_to_view


def _move_nested_to_device(bundle: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in bundle.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _cached_bundle_matches_request(
    cached: Dict[str, torch.Tensor],
    *,
    batch_size: int,
    num_views: int,
    target_hw: Tuple[int, int],
    image_names: Optional[Sequence[str]],
) -> bool:
    depth_hr = cached.get("depth_hr")
    conf_hr = cached.get("conf_hr")
    feat_s4 = cached.get("feat_s4")
    if not torch.is_tensor(depth_hr) or not torch.is_tensor(conf_hr) or not torch.is_tensor(feat_s4):
        return False

    target_h, target_w = target_hw
    if tuple(depth_hr.shape) != (batch_size, num_views, 1, target_h, target_w):
        return False
    if tuple(conf_hr.shape) != (batch_size, num_views, 1, target_h, target_w):
        return False
    if feat_s4.ndim != 5 or feat_s4.shape[0] != batch_size or feat_s4.shape[1] != num_views:
        return False

    if image_names is not None:
        cached_names = cached.get("image_names")
        if cached_names is None:
            return False
        if list(cached_names) != list(image_names):
            return False

    return True


def _canonicalize_dense_prediction(
    tensor: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    num_views: int,
) -> torch.Tensor:
    if tensor.ndim == 5 and tensor.shape[2] == 1:
        return tensor
    if tensor.ndim == 4 and tensor.shape[:2] == (batch_size, num_views):
        return tensor.unsqueeze(2)
    if tensor.ndim == 5 and tensor.shape[-1] == 1 and tensor.shape[:2] == (batch_size, num_views):
        return tensor.permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"Unsupported {name} shape from VGGT: {tuple(tensor.shape)}")


def _filter_matching_state_dict(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    current = module.state_dict()
    matched = {}
    for key, value in state_dict.items():
        if key not in current:
            continue
        if current[key].shape != value.shape:
            continue
        matched[key] = value
    return matched


class FrozenVGGTAdapter:
    def __init__(self, config: VGGTAdapterConfig) -> None:
        self.config = config
        self._model = None
        self._pose_encoding_to_extri_intri = None
        self._feature_head = None

    def _setup(self):
        if self._model is not None:
            return self._model, self._pose_encoding_to_extri_intri, self._feature_head

        vggt_root = _ensure_vggt_import_path(self.config.vggt_root)
        from vggt.models.vggt import VGGT  # type: ignore
        from vggt.heads.dpt_head import DPTHead  # type: ignore
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

        model = VGGT()
        ckpt_cache = vggt_root / self.config.model_cache_name
        if ckpt_cache.exists():
            state = torch.load(ckpt_cache, map_location="cpu")
        else:
            url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            print("[VGGT] Downloading model weights ...")
            urllib.request.urlretrieve(url, ckpt_cache)
            state = torch.load(ckpt_cache, map_location="cpu")

        model.load_state_dict(state)
        model.eval().to(self.config.device)

        feature_head = DPTHead(
            dim_in=model.depth_head.norm.normalized_shape[0],
            patch_size=model.depth_head.patch_size,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            intermediate_layer_idx=model.depth_head.intermediate_layer_idx,
            pos_embed=model.depth_head.pos_embed,
            feature_only=True,
            down_ratio=self.config.feature_down_ratio,
        )
        matched = _filter_matching_state_dict(feature_head, model.depth_head.state_dict())
        feature_head.load_state_dict(matched, strict=False)
        feature_head.eval().to(self.config.device)

        self._model = model
        self._pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
        self._feature_head = feature_head
        return self._model, self._pose_encoding_to_extri_intri, self._feature_head

    @torch.no_grad()
    def run(
        self,
        lr_images: torch.Tensor,
        *,
        target_hw: Tuple[int, int],
        image_names: Optional[Sequence[str]] = None,
        cache_path: str | Path | None = None,
    ) -> Dict[str, torch.Tensor]:
        if lr_images.ndim == 4:
            lr_images = lr_images.unsqueeze(0)
        if lr_images.ndim != 5:
            raise ValueError(f"Expected lr_images [B, V, 3, H, W], got {tuple(lr_images.shape)}")

        if cache_path is not None:
            cache_path = Path(cache_path).expanduser().resolve()
            if cache_path.is_file():
                cached = torch.load(cache_path, map_location="cpu")
                if not isinstance(cached, dict):
                    raise TypeError(f"Unsupported VGGT cache type: {type(cached)!r}")
                if _cached_bundle_matches_request(
                    cached,
                    batch_size=lr_images.shape[0],
                    num_views=lr_images.shape[1],
                    target_hw=target_hw,
                    image_names=image_names,
                ):
                    return _move_nested_to_device(cached, self.config.device)
                print(f"[VGGT] cache mismatch, recomputing prior: {cache_path}", flush=True)

        model, pose_encoding_to_extri_intri, feature_head = self._setup()

        b, v, _, src_h, src_w = lr_images.shape
        target_h, target_w = target_hw
        device = self.config.device
        dtype = _resolve_autocast_dtype(device)

        lr_images = lr_images.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        images_vggt = F.interpolate(
            lr_images.flatten(0, 1),
            size=(self.config.vggt_resolution, self.config.vggt_resolution),
            mode="bilinear",
            align_corners=False,
        ).unflatten(0, (b, v))

        with torch.cuda.amp.autocast(dtype=dtype, enabled=(device != "cpu")):
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_vggt)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            depth_raw, conf_raw = model.depth_head(
                aggregated_tokens_list,
                images=images_vggt,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=self.config.frames_chunk_size,
            )
            feat_s4 = feature_head(
                aggregated_tokens_list,
                images=images_vggt,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=self.config.frames_chunk_size,
            )

        depth_raw = _canonicalize_dense_prediction(
            depth_raw.float(),
            name="depth_raw",
            batch_size=b,
            num_views=v,
        )
        conf_raw = _canonicalize_dense_prediction(
            conf_raw.float(),
            name="conf_raw",
            batch_size=b,
            num_views=v,
        )

        pred_extrinsic, pred_intrinsic = pose_encoding_to_extri_intri(
            pose_enc,
            images_vggt.shape[-2:],
        )
        pred_world_to_view = _lift_extrinsic_to_world_to_view(pred_extrinsic.float())
        pred_cam_to_world = torch.linalg.inv(pred_world_to_view)

        depth_hr = F.interpolate(
            depth_raw.flatten(0, 1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).unflatten(0, (b, v)).float()
        conf_hr = F.interpolate(
            conf_raw.flatten(0, 1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).unflatten(0, (b, v)).float()

        feat_h = max(int(target_h) // int(self.config.feature_down_ratio), 1)
        feat_w = max(int(target_w) // int(self.config.feature_down_ratio), 1)
        feat_s4 = F.interpolate(
            feat_s4.flatten(0, 1),
            size=(feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        ).unflatten(0, (b, v)).float()

        pred_intrinsic_hr = _resize_intrinsics_to_hw(
            pred_intrinsic.float(),
            (self.config.vggt_resolution, self.config.vggt_resolution),
            (target_h, target_w),
        )

        bundle: Dict[str, torch.Tensor] = {
            "depth_hr": depth_hr,
            "conf_hr": conf_hr,
            "feat_s4": feat_s4,
            "pred_intrinsics": pred_intrinsic_hr,
            "pred_world_to_view": pred_world_to_view.float(),
            "pred_cam_to_world": pred_cam_to_world.float(),
        }
        if image_names is not None:
            bundle["image_names"] = list(image_names)  # type: ignore[assignment]
        bundle["meta"] = {  # type: ignore[assignment]
            "config": asdict(self.config),
            "source_hw": [src_h, src_w],
            "target_hw": [target_h, target_w],
        }

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {}
            for key, value in bundle.items():
                if torch.is_tensor(value):
                    serializable[key] = value.detach().cpu()
                else:
                    serializable[key] = value
            torch.save(serializable, cache_path)

        return bundle

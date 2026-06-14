import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.depth_utils import depth_to_normal
from utils.general_utils import build_scaling_rotation
from utils.sh_utils import RGB2SH


@dataclass
class PriorSeedSpec:
    image_name: str
    x: float
    y: float
    confidence: float = 1.0
    patch_radius: int = 1


@dataclass
class ViewInjectionSummary:
    image_name: str
    requested_seeds: int
    injected_seeds: int
    injected_gaussians: int
    skipped_invalid_anchor: int
    skipped_missing_prior: int
    skipped_weak_support: int = 0
    anchor_pixels: int = 0
    near_pixels: int = 0
    reliable_pixels: int = 0
    good_pixels: int = 0
    need_pixels: int = 0
    geom_pixels: int = 0
    rep_pixels: int = 0
    thin_pixels: int = 0
    inject_pixels: int = 0


@dataclass
class InjectionSummary:
    baseline_checkpoint: str
    output_checkpoint: str
    prior_dir: str
    candidate_mode: str
    auto_mask_variant: str
    injection_preset: str
    bundle_size: int
    bundle_spacing_scale: float
    center_opacity: float
    min_injected_opacity: float
    side_opacity_scale: float
    surface_support_lock_enabled: bool
    surface_support_min_neighbors: int
    surface_support_neighbor_soft_target: int
    surface_support_max_normal_ratio: float
    surface_support_min_strength: float
    surface_support_confidence_power: float
    total_requested_seeds: int
    total_injected_seeds: int
    total_injected_gaussians: int
    views: List[ViewInjectionSummary]


@dataclass
class RenderedViewState:
    render_rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    normal_world: torch.Tensor
    points_world: torch.Tensor


def index_image_dir(image_dir: str) -> Dict[str, Path]:
    indexed: Dict[str, Path] = {}
    for path in sorted(Path(image_dir).rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        indexed[path.stem] = path
    return indexed


def normalize_image_name(image_name: str) -> str:
    return Path(image_name).stem


def load_rgb_image(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image)


def load_prior_image(path: Path) -> torch.Tensor:
    return load_rgb_image(path)


def load_seed_manifest(path: str) -> List[PriorSeedSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds: List[PriorSeedSpec] = []
    if isinstance(payload, dict):
        for image_name, items in payload.items():
            for item in items:
                seeds.append(
                    PriorSeedSpec(
                        image_name=normalize_image_name(image_name),
                        x=float(item["x"]),
                        y=float(item["y"]),
                        confidence=float(item.get("confidence", 1.0)),
                        patch_radius=int(item.get("patch_radius", 1)),
                    )
                )
        return seeds

    for item in payload:
        seeds.append(
            PriorSeedSpec(
                image_name=normalize_image_name(str(item["image_name"])),
                x=float(item["x"]),
                y=float(item["y"]),
                confidence=float(item.get("confidence", 1.0)),
                patch_radius=int(item.get("patch_radius", 1)),
            )
        )
    return seeds


def load_mask(path: Path) -> torch.Tensor:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(mask)


def generate_grid_candidates(image_name: str, width: int, height: int, stride: int, border: int, max_seeds: int = 0) -> List[PriorSeedSpec]:
    seeds: List[PriorSeedSpec] = []
    for y in range(border, max(border, height - border), stride):
        for x in range(border, max(border, width - border), stride):
            seeds.append(PriorSeedSpec(image_name=image_name, x=float(x), y=float(y)))
            if max_seeds > 0 and len(seeds) >= max_seeds:
                return seeds
    return seeds


def generate_mask_candidates(
    image_name: str,
    mask: torch.Tensor,
    stride: int,
    border: int,
    max_seeds: int = 0,
) -> List[PriorSeedSpec]:
    height, width = mask.shape
    seeds: List[PriorSeedSpec] = []
    for y in range(border, max(border, height - border), stride):
        for x in range(border, max(border, width - border), stride):
            if float(mask[y, x]) <= 0.0:
                continue
            seeds.append(PriorSeedSpec(image_name=image_name, x=float(x), y=float(y)))
            if max_seeds > 0 and len(seeds) >= max_seeds:
                return seeds
    return seeds


def group_seeds_by_view(seeds: Sequence[PriorSeedSpec]) -> Dict[str, List[PriorSeedSpec]]:
    grouped: Dict[str, List[PriorSeedSpec]] = {}
    for seed in seeds:
        grouped.setdefault(seed.image_name, []).append(seed)
    return grouped


def render_view_state(view, gaussians: GaussianModel, pipeline, background, splat_args) -> RenderedViewState:
    with torch.no_grad():
        render_pkg = render(view, gaussians, pipeline, background, splat_args=splat_args)
        render_tensor = render_pkg["render"]
        if "depth" in render_pkg and "alpha" in render_pkg:
            render_rgb = render_tensor.permute(1, 2, 0)
            depth = render_pkg["depth"][0]
            alpha = render_pkg["alpha"][0]
        else:
            if render_tensor.shape[0] < 8:
                raise KeyError(
                    "Renderer output does not expose depth/alpha keys and render tensor has fewer than 8 channels."
                )
            render_rgb = render_tensor[:3].permute(1, 2, 0)
            depth = render_tensor[6]
            alpha = render_tensor[7]
        normal_world, points_world = depth_to_normal(view, depth[None, ...])
    return RenderedViewState(
        render_rgb=render_rgb,
        alpha=alpha,
        depth=depth,
        normal_world=normal_world,
        points_world=points_world,
    )


def infer_lr_dir_from_pseudo_scene(source_path: str) -> Optional[str]:
    summary_path = Path(source_path) / "pseudo_sr_summary.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    scene_root = payload.get("scene_root")
    source_images_subdir = payload.get("source_images_subdir")
    if scene_root is None or source_images_subdir is None:
        return None
    return str((Path(scene_root) / source_images_subdir).resolve())


def _to_nchw(image_hw3: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    return image_hw3.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)


def _to_hw(image_nchw: torch.Tensor) -> torch.Tensor:
    return image_nchw[0, 0]


def _to_hwc(image_nchw: torch.Tensor) -> torch.Tensor:
    return image_nchw[0].permute(1, 2, 0)


def blur_image(image_hw3: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return image_hw3.to(device="cuda", dtype=torch.float32)
    x = _to_nchw(image_hw3)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1)
    return _to_hwc(x)


def resize_image_hw3(image_hw3: torch.Tensor, target_height: int, target_width: int, mode: str = "bicubic") -> torch.Tensor:
    image = _to_nchw(image_hw3)
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    resized = F.interpolate(
        image,
        size=(target_height, target_width),
        mode=mode,
        align_corners=align_corners,
    )
    return _to_hwc(resized).clamp(0.0, 1.0)


def high_pass_image(image_hw3: torch.Tensor, kernel_size: int) -> torch.Tensor:
    image = image_hw3.to(device="cuda", dtype=torch.float32)
    return image - blur_image(image, kernel_size)


def gradient_magnitude_map(image_hw3: torch.Tensor) -> torch.Tensor:
    image = _to_nchw(image_hw3)
    gray = image.mean(dim=1, keepdim=True)
    gray = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    dx = gray[:, :, 1:-1, 2:] - gray[:, :, 1:-1, :-2]
    dy = gray[:, :, 2:, 1:-1] - gray[:, :, :-2, 1:-1]
    return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-12)[0, 0]


def gradient_magnitude_scalar_map(image_hw: torch.Tensor) -> torch.Tensor:
    image = image_hw.to(device="cuda", dtype=torch.float32)[None, None]
    image = F.pad(image, (1, 1, 1, 1), mode="reflect")
    dx = image[:, :, 1:-1, 2:] - image[:, :, 1:-1, :-2]
    dy = image[:, :, 2:, 1:-1] - image[:, :, :-2, 1:-1]
    return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-12)[0, 0]


def local_average_map(values_hw: torch.Tensor, kernel_size: int) -> torch.Tensor:
    values = values_hw.to(device="cuda", dtype=torch.float32)[None, None]
    if kernel_size <= 1:
        return values[0, 0]
    pad = kernel_size // 2
    values = F.pad(values, (pad, pad, pad, pad), mode="reflect")
    values = F.avg_pool2d(values, kernel_size=kernel_size, stride=1)
    return values[0, 0]


def dilate_binary_mask(mask_hw: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mask = mask_hw.to(device="cuda", dtype=torch.float32)[None, None]
    if kernel_size <= 1:
        return mask[0, 0] > 0
    pad = kernel_size // 2
    mask = F.pad(mask, (pad, pad, pad, pad), mode="replicate")
    mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1)
    return mask[0, 0] > 0


def normalize_robust_map(
    values_hw: torch.Tensor,
    valid_mask: torch.Tensor,
    low_quantile: float = 0.05,
    high_quantile: float = 0.95,
) -> torch.Tensor:
    values = values_hw.to(device="cuda", dtype=torch.float32)
    valid = valid_mask & torch.isfinite(values)
    if int(valid.sum().item()) == 0:
        return torch.zeros_like(values)

    samples = values[valid]
    lo = torch.quantile(samples, low_quantile)
    hi = torch.quantile(samples, high_quantile)
    denom = float((hi - lo).item())
    if denom < 1e-6:
        return torch.zeros_like(values)
    return ((values - lo) / (hi - lo)).clamp(0.0, 1.0)


def smooth_binary_mask(mask_hw: torch.Tensor, kernel_size: int, threshold: float) -> torch.Tensor:
    if kernel_size <= 1:
        return mask_hw > 0
    mask = mask_hw.to(device="cuda", dtype=torch.float32)[None, None]
    pad = kernel_size // 2
    mask = F.pad(mask, (pad, pad, pad, pad), mode="replicate")
    smoothed = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1)
    return smoothed[0, 0] >= threshold


def build_anchor_mask(view_state: RenderedViewState, alpha_threshold: float, normal_threshold: float) -> torch.Tensor:
    normal_norm = torch.linalg.norm(view_state.normal_world, dim=-1)
    finite_normal = torch.isfinite(normal_norm)
    finite_depth = torch.isfinite(view_state.depth)
    return (
        (view_state.alpha > alpha_threshold)
        & (view_state.depth > 0.0)
        & finite_depth
        & finite_normal
        & (normal_norm > normal_threshold)
    )


def build_near_foreground_mask(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    near_quantile: float,
) -> torch.Tensor:
    finite = valid_mask & torch.isfinite(depth)
    if int(finite.sum().item()) == 0:
        return torch.zeros_like(valid_mask)
    depth_values = depth[finite]
    q = float(min(max(near_quantile, 0.0), 1.0))
    if q <= 0.0:
        depth_cutoff = depth_values.min()
    elif q >= 1.0:
        depth_cutoff = depth_values.max()
    else:
        depth_cutoff = torch.quantile(depth_values, q)
    return finite & (depth <= depth_cutoff)


def build_reliable_prior_mask(
    prior_hr: torch.Tensor,
    lr_image: torch.Tensor,
    target_height: int,
    target_width: int,
    sigma_rgb: float,
    sigma_grad: float,
    threshold: float,
    blur_kernel: int,
) -> torch.Tensor:
    prior_lr = F.interpolate(
        _to_nchw(prior_hr),
        size=lr_image.shape[:2],
        mode="bicubic",
        align_corners=False,
    )
    prior_lr_hwc = _to_hwc(prior_lr).clamp(0.0, 1.0)
    lr_image = lr_image.to(device="cuda", dtype=torch.float32)

    rgb_diff = torch.abs(prior_lr_hwc - lr_image).mean(dim=-1)
    grad_prior = gradient_magnitude_map(prior_lr_hwc)
    grad_lr = gradient_magnitude_map(lr_image)
    grad_diff = torch.abs(grad_prior - grad_lr)

    reliability_lr = torch.exp(-rgb_diff / max(sigma_rgb, 1e-6)) * torch.exp(-grad_diff / max(sigma_grad, 1e-6))
    reliability_hr = F.interpolate(
        reliability_lr[None, None],
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    if blur_kernel > 1:
        reliability_hr = blur_image(reliability_hr[..., None].repeat(1, 1, 3), blur_kernel)[..., 0]
    return reliability_hr >= threshold


def build_need_region_mask(
    prior_hr: torch.Tensor,
    render_hr: torch.Tensor,
    valid_mask: torch.Tensor,
    pool_kernel: int,
    score_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prior = prior_hr.to(device="cuda", dtype=torch.float32)
    render = render_hr.to(device="cuda", dtype=torch.float32)
    rgb_delta = local_average_map(torch.abs(prior - render).mean(dim=-1), pool_kernel)
    grad_gain = local_average_map(
        (gradient_magnitude_map(prior) - gradient_magnitude_map(render)).clamp_min(0.0),
        pool_kernel,
    )

    rgb_score = normalize_robust_map(rgb_delta, valid_mask)
    grad_score = normalize_robust_map(grad_gain, valid_mask)
    need_score = 0.5 * (rgb_score + grad_score)
    need_mask = valid_mask & (need_score >= score_threshold)
    return need_score, need_mask


def build_good_region_mask(
    prior_hr: torch.Tensor,
    render_hr: torch.Tensor,
    alpha: torch.Tensor,
    alpha_threshold: float,
    match_threshold: float,
    highpass_threshold: float,
    highpass_kernel: int,
) -> torch.Tensor:
    prior = prior_hr.to(device="cuda", dtype=torch.float32)
    render = render_hr.to(device="cuda", dtype=torch.float32)
    match = torch.abs(prior - render).mean(dim=-1)
    hp_prior = high_pass_image(prior, highpass_kernel)
    hp_render = high_pass_image(render, highpass_kernel)
    hp_match = torch.abs(hp_prior - hp_render).mean(dim=-1)
    return (alpha > alpha_threshold) & (match < match_threshold) & (hp_match < highpass_threshold)


def build_geometry_support_mask(
    view_state: RenderedViewState,
    valid_mask: torch.Tensor,
    pool_kernel: int,
    score_threshold: float,
    band_kernel: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    depth_grad = local_average_map(gradient_magnitude_scalar_map(view_state.depth), pool_kernel)
    alpha_grad = local_average_map(gradient_magnitude_scalar_map(view_state.alpha), pool_kernel)
    normal_grad = local_average_map(gradient_magnitude_map(view_state.normal_world), pool_kernel)

    depth_score = normalize_robust_map(depth_grad, valid_mask)
    alpha_score = normalize_robust_map(alpha_grad, valid_mask)
    normal_score = normalize_robust_map(normal_grad, valid_mask)
    geom_score = torch.maximum(depth_score, torch.maximum(alpha_score, normal_score))
    geom_core = valid_mask & (geom_score >= score_threshold)
    geom_band = dilate_binary_mask(geom_core, band_kernel) & valid_mask
    return geom_score, geom_band


def build_repetitive_risk_mask(
    prior_hr: torch.Tensor,
    render_hr: torch.Tensor,
    valid_mask: torch.Tensor,
    geom_score: torch.Tensor,
    pool_kernel: int,
    score_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prior = prior_hr.to(device="cuda", dtype=torch.float32)
    render = render_hr.to(device="cuda", dtype=torch.float32)
    texture_energy = local_average_map(
        torch.maximum(gradient_magnitude_map(prior), gradient_magnitude_map(render)),
        pool_kernel,
    )
    rep_score_raw = texture_energy * (1.0 - geom_score).clamp(0.0, 1.0)
    rep_score = normalize_robust_map(rep_score_raw, valid_mask)
    rep_mask = valid_mask & (rep_score >= score_threshold)
    return rep_score, rep_mask


def build_thin_structure_mask(
    prior_hr: torch.Tensor,
    render_hr: torch.Tensor,
    valid_mask: torch.Tensor,
    geom_mask: torch.Tensor,
    rep_score: torch.Tensor,
    pool_kernel: int,
    score_threshold: float,
    edge_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prior = prior_hr.to(device="cuda", dtype=torch.float32)
    render = render_hr.to(device="cuda", dtype=torch.float32)
    edge_strength = torch.maximum(gradient_magnitude_map(prior), gradient_magnitude_map(render))
    edge_score = normalize_robust_map(edge_strength, valid_mask)
    edge_binary = valid_mask & (edge_score >= edge_threshold)

    # Thin structures tend to pack multiple edges into a small neighborhood,
    # while a single broad boundary occupies a much smaller fraction of the patch.
    edge_density = local_average_map(edge_binary.to(dtype=torch.float32), pool_kernel)
    geom_density = local_average_map(geom_mask.to(dtype=torch.float32), pool_kernel)
    thin_score_raw = edge_density * geom_density * (1.0 - rep_score).clamp(0.0, 1.0)
    thin_score = normalize_robust_map(thin_score_raw, valid_mask)
    thin_mask = valid_mask & (thin_score >= score_threshold)
    return thin_score, thin_mask


def build_auto_injection_mask(
    *,
    view_state: RenderedViewState,
    prior_hr: torch.Tensor,
    lr_image: torch.Tensor,
    auto_mask_variant: str,
    alpha_threshold: float,
    normal_threshold: float,
    reliable_sigma_rgb: float,
    reliable_sigma_grad: float,
    reliable_threshold: float,
    reliable_blur_kernel: int,
    near_depth_quantile: float,
    need_pool_kernel: int,
    need_score_threshold: float,
    geom_pool_kernel: int,
    geom_score_threshold: float,
    geom_band_kernel: int,
    rep_pool_kernel: int,
    rep_score_threshold: float,
    thin_pool_kernel: int,
    thin_score_threshold: float,
    thin_edge_threshold: float,
    good_alpha_threshold: float,
    good_match_threshold: float,
    good_highpass_threshold: float,
    good_highpass_kernel: int,
    mask_smoothing_kernel: int,
    mask_smoothing_threshold: float,
) -> Dict[str, torch.Tensor]:
    anchor_mask = build_anchor_mask(view_state, alpha_threshold, normal_threshold)
    reliable_mask = build_reliable_prior_mask(
        prior_hr=prior_hr,
        lr_image=lr_image,
        target_height=view_state.depth.shape[0],
        target_width=view_state.depth.shape[1],
        sigma_rgb=reliable_sigma_rgb,
        sigma_grad=reliable_sigma_grad,
        threshold=reliable_threshold,
        blur_kernel=reliable_blur_kernel,
    )
    zero_mask = torch.zeros_like(anchor_mask)
    zero_score = torch.zeros_like(view_state.depth, dtype=torch.float32, device="cuda")

    if auto_mask_variant == "legacy":
        near_mask = zero_mask
        need_mask = zero_mask
        geom_mask = zero_mask
        rep_mask = zero_mask
        thin_mask = zero_mask
        need_score = zero_score
        geom_score = zero_score
        rep_score = zero_score
        thin_score = zero_score
        good_mask = build_good_region_mask(
            prior_hr=prior_hr,
            render_hr=view_state.render_rgb,
            alpha=view_state.alpha,
            alpha_threshold=good_alpha_threshold,
            match_threshold=good_match_threshold,
            highpass_threshold=good_highpass_threshold,
            highpass_kernel=good_highpass_kernel,
        )
        inject_mask = anchor_mask & reliable_mask & (~good_mask)
    elif auto_mask_variant == "conservative_v1":
        near_mask = build_near_foreground_mask(
            depth=view_state.depth,
            valid_mask=anchor_mask,
            near_quantile=near_depth_quantile,
        )
        valid_mask = near_mask & reliable_mask
        need_score, need_mask = build_need_region_mask(
            prior_hr=prior_hr,
            render_hr=view_state.render_rgb,
            valid_mask=valid_mask,
            pool_kernel=need_pool_kernel,
            score_threshold=need_score_threshold,
        )
        geom_score, geom_mask = build_geometry_support_mask(
            view_state=view_state,
            valid_mask=valid_mask,
            pool_kernel=geom_pool_kernel,
            score_threshold=geom_score_threshold,
            band_kernel=geom_band_kernel,
        )
        rep_score, rep_mask = build_repetitive_risk_mask(
            prior_hr=prior_hr,
            render_hr=view_state.render_rgb,
            valid_mask=valid_mask,
            geom_score=geom_score,
            pool_kernel=rep_pool_kernel,
            score_threshold=rep_score_threshold,
        )
        good_mask = zero_mask
        inject_mask = valid_mask & need_mask & geom_mask & (~rep_mask)
        thin_mask = zero_mask
        thin_score = zero_score
    elif auto_mask_variant == "conservative_v2":
        near_mask = build_near_foreground_mask(
            depth=view_state.depth,
            valid_mask=anchor_mask,
            near_quantile=near_depth_quantile,
        )
        valid_mask = near_mask & reliable_mask
        need_score, need_mask = build_need_region_mask(
            prior_hr=prior_hr,
            render_hr=view_state.render_rgb,
            valid_mask=valid_mask,
            pool_kernel=need_pool_kernel,
            score_threshold=need_score_threshold,
        )
        geom_score, geom_mask = build_geometry_support_mask(
            view_state=view_state,
            valid_mask=valid_mask,
            pool_kernel=geom_pool_kernel,
            score_threshold=geom_score_threshold,
            band_kernel=geom_band_kernel,
        )
        rep_score, rep_mask = build_repetitive_risk_mask(
            prior_hr=prior_hr,
            render_hr=view_state.render_rgb,
            valid_mask=valid_mask,
            geom_score=geom_score,
            pool_kernel=rep_pool_kernel,
            score_threshold=rep_score_threshold,
        )
        thin_score, thin_mask = build_thin_structure_mask(
            prior_hr=prior_hr,
            render_hr=view_state.render_rgb,
            valid_mask=valid_mask,
            geom_mask=geom_mask,
            rep_score=rep_score,
            pool_kernel=thin_pool_kernel,
            score_threshold=thin_score_threshold,
            edge_threshold=thin_edge_threshold,
        )
        good_mask = zero_mask
        inject_mask = valid_mask & need_mask & geom_mask & thin_mask & (~rep_mask)
    else:
        raise ValueError(f"Unsupported auto_mask_variant: {auto_mask_variant}")

    inject_mask = smooth_binary_mask(inject_mask, mask_smoothing_kernel, mask_smoothing_threshold)
    if auto_mask_variant in {"conservative_v1", "conservative_v2"}:
        inject_mask = inject_mask & near_mask & reliable_mask
    return {
        "anchor_mask": anchor_mask,
        "near_mask": near_mask,
        "reliable_mask": reliable_mask,
        "good_mask": good_mask,
        "need_mask": need_mask,
        "geom_mask": geom_mask,
        "rep_mask": rep_mask,
        "thin_mask": thin_mask,
        "need_score": need_score,
        "geom_score": geom_score,
        "rep_score": rep_score,
        "thin_score": thin_score,
        "inject_mask": inject_mask,
    }


def sample_prior_patch_rgb(prior_image: torch.Tensor, x: float, y: float, patch_radius: int) -> torch.Tensor:
    height, width, _ = prior_image.shape
    ix = int(round(x))
    iy = int(round(y))
    x0 = max(ix - patch_radius, 0)
    x1 = min(ix + patch_radius + 1, width)
    y0 = max(iy - patch_radius, 0)
    y1 = min(iy + patch_radius + 1, height)
    patch = prior_image[y0:y1, x0:x1]
    return patch.mean(dim=(0, 1)).to(device="cuda", dtype=torch.float32)


def sample_anchor_from_view(
    view_state: RenderedViewState,
    view,
    x: float,
    y: float,
    alpha_threshold: float,
    normal_threshold: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    ix = int(round(x))
    iy = int(round(y))
    height, width = view_state.depth.shape
    if ix < 0 or iy < 0 or ix >= width or iy >= height:
        return None

    if float(view_state.alpha[iy, ix]) < alpha_threshold:
        return None
    if float(view_state.depth[iy, ix]) <= 0.0:
        return None

    point = view_state.points_world[iy, ix].to(device="cuda", dtype=torch.float32)
    normal = view_state.normal_world[iy, ix].to(device="cuda", dtype=torch.float32)
    if torch.isnan(point).any() or torch.isnan(normal).any():
        return None
    if float(torch.linalg.norm(normal)) < normal_threshold:
        return None

    normal = F.normalize(normal, dim=0)
    view_dir = F.normalize(view.camera_center - point, dim=0)
    if torch.dot(normal, view_dir) < 0:
        normal = -normal
    return point, normal


def build_local_frame(normal: torch.Tensor) -> torch.Tensor:
    helper = torch.tensor([1.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype)
    if torch.abs(torch.dot(helper, normal)) > 0.9:
        helper = torch.tensor([0.0, 1.0, 0.0], device=normal.device, dtype=normal.dtype)
    tangent_1 = F.normalize(torch.cross(helper, normal, dim=0), dim=0)
    tangent_2 = F.normalize(torch.cross(normal, tangent_1, dim=0), dim=0)
    return torch.stack([tangent_1, tangent_2, normal], dim=1)


def rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = torch.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = torch.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quat = torch.stack([qw, qx, qy, qz], dim=0)
    return F.normalize(quat, dim=0)


def estimate_local_surface_scales(
    gaussians: GaussianModel,
    anchor: torch.Tensor,
    normal: torch.Tensor,
    neighborhood_radius: float,
    neighborhood_thickness: float,
) -> Tuple[float, float, int]:
    xyz = gaussians.get_xyz.detach()
    diff = xyz - anchor[None, :]
    normal_offset = diff @ normal
    tangent_diff = diff - normal_offset[:, None] * normal[None, :]
    tangent_radius_sq = torch.sum(tangent_diff * tangent_diff, dim=1)
    mask = torch.logical_and(tangent_radius_sq <= neighborhood_radius ** 2, torch.abs(normal_offset) <= neighborhood_thickness)

    scales = gaussians.get_scaling.detach()
    rotations = gaussians._rotation.detach()
    if mask.any():
        scales = scales[mask]
        rotations = rotations[mask]
        neighbor_count = int(mask.sum().item())
    else:
        neighbor_count = 0

    if scales.shape[0] == 0:
        min_axis = gaussians.get_scaling.detach().min(dim=1).values
        max_axis = gaussians.get_scaling.detach().max(dim=1).values
        return float(max_axis.median().item()), float(min_axis.median().item()), neighbor_count

    cov = build_scaling_rotation(scales, rotations)
    cov = cov @ cov.transpose(1, 2)
    normal_var = torch.einsum("i,nij,j->n", normal, cov, normal).clamp_min(1e-12)
    tangent_var = ((torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1) - normal_var) / 2.0).clamp_min(1e-12)
    tangent_scale = torch.sqrt(tangent_var).median()
    normal_scale = torch.sqrt(normal_var).median()
    return float(tangent_scale.item()), float(normal_scale.item()), neighbor_count


def compute_surface_support_strength(
    *,
    neighbor_count: int,
    tangent_scale: float,
    normal_scale: float,
    neighbor_soft_target: int,
    max_normal_ratio: float,
) -> float:
    safe_tangent = max(float(tangent_scale), 1e-6)
    safe_ratio = max(float(max_normal_ratio), 1e-6)
    neighbor_target = max(int(neighbor_soft_target), 1)

    neighbor_score = min(max(float(neighbor_count), 0.0) / float(neighbor_target), 1.0)
    normal_ratio = float(normal_scale) / safe_tangent
    surface_score = 1.0 - min(normal_ratio / safe_ratio, 1.0)
    return max(0.0, neighbor_score * surface_score)


def build_bundle_feature_tensors(gaussians: GaussianModel, rgb: torch.Tensor, bundle_size: int):
    if gaussians.use_SBs:
        features_dc = rgb[None, :].repeat(bundle_size, 1)
        features_rest = torch.zeros(
            (bundle_size, gaussians._features_rest.shape[1]),
            device="cuda",
            dtype=gaussians._features_rest.dtype,
        )
        return features_dc, features_rest

    sh_dc = RGB2SH(rgb[None, :])
    features_dc = sh_dc[:, None, :].repeat(bundle_size, 1, 1)
    features_rest = torch.zeros(
        (bundle_size, gaussians._features_rest.shape[1], gaussians._features_rest.shape[2]),
        device="cuda",
        dtype=gaussians._features_rest.dtype,
    )
    return features_dc, features_rest


def build_normal_bundle(
    gaussians: GaussianModel,
    anchor: torch.Tensor,
    normal: torch.Tensor,
    rgb: torch.Tensor,
    confidence: float,
    tangent_scale: float,
    normal_scale: float,
    bundle_spacing_scale: float,
    center_opacity: float,
    min_opacity: float,
    side_opacity_scale: float,
    seed_id: int,
    bundle_size: int = 3,
):
    if bundle_size % 2 == 0:
        raise ValueError("bundle_size must be odd to place a center Gaussian on the surface anchor.")

    rotation = build_local_frame(normal)
    quaternion = rotation_matrix_to_quaternion(rotation)[None, :].repeat(bundle_size, 1)

    tangent_scale = max(tangent_scale, 1e-4)
    normal_scale = max(normal_scale, 1e-5)
    spacing = bundle_spacing_scale * normal_scale
    offsets = torch.linspace(
        -(bundle_size // 2),
        bundle_size // 2,
        steps=bundle_size,
        device="cuda",
        dtype=torch.float32,
    )
    xyz = anchor[None, :] + offsets[:, None] * spacing * normal[None, :]

    scale_values = torch.tensor(
        [tangent_scale, tangent_scale, normal_scale],
        device="cuda",
        dtype=torch.float32,
    )[None, :].repeat(bundle_size, 1)
    scaling = gaussians.scaling_inverse_activation(scale_values)

    effective_center_opacity = max(min_opacity, center_opacity * max(confidence, 1e-3))
    opacity_values = torch.full((bundle_size, 1), effective_center_opacity * side_opacity_scale, device="cuda")
    opacity_values[bundle_size // 2, 0] = effective_center_opacity
    opacity_values = opacity_values.clamp(min=1e-4, max=1.0 - 1e-6)
    opacities = gaussians.inverse_opacity_activation(opacity_values)

    features_dc, features_rest = build_bundle_feature_tensors(gaussians, rgb, bundle_size)
    tracking_state = gaussians._build_tracking_extension(
        bundle_size,
        source_tag=torch.full((bundle_size,), int(GaussianSourceTag.PRIOR_INJECTED), device="cuda", dtype=torch.int32),
        seed_id=torch.full((bundle_size,), seed_id, device="cuda", dtype=torch.int64),
        generation=torch.zeros((bundle_size,), device="cuda", dtype=torch.int32),
    )

    return xyz, features_dc, features_rest, opacities, scaling, quaternion, tracking_state


def save_injection_summary(summary: InjectionSummary, path: str):
    serializable = asdict(summary)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")

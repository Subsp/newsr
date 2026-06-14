#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

try:
    from diff_gaussian_rasterization import DebugVisualization, ExtendedSettings
except ImportError:
    DebugVisualization = None
    ExtendedSettings = None

try:
    from diff_gaussian_rasterization import supports_merged_features as _supports_merged_features
except ImportError:
    def _supports_merged_features() -> bool:
        return False


_RASTER_SETTINGS_FIELDS = set(getattr(GaussianRasterizationSettings, "_fields", ()))
_SOF_RASTERIZER_FIELDS = {"inv_viewprojmatrix", "settings", "debug_data"}
_VANILLA_SOF_RASTERIZER_FIELDS = {"inv_viewprojmatrix", "settings", "debug_data"}
_VANILLA_SOF_MODULE = None
_VANILLA_SOF_IMPORT_ERROR = None


def supports_merged_sof_rasterizer() -> bool:
    return (
        ExtendedSettings is not None
        and _supports_merged_features()
        and _SOF_RASTERIZER_FIELDS.issubset(_RASTER_SETTINGS_FIELDS)
    )


def _load_vanilla_sof_module():
    global _VANILLA_SOF_MODULE, _VANILLA_SOF_IMPORT_ERROR
    if _VANILLA_SOF_MODULE is not None:
        return _VANILLA_SOF_MODULE
    if _VANILLA_SOF_IMPORT_ERROR is not None:
        return None
    try:
        import diff_gaussian_rasterization_sof_vanilla as vanilla_mod
    except ImportError as exc:
        _VANILLA_SOF_IMPORT_ERROR = exc
        return None
    _VANILLA_SOF_MODULE = vanilla_mod
    return vanilla_mod


def supports_vanilla_sof_rasterizer() -> bool:
    vanilla_mod = _load_vanilla_sof_module()
    if vanilla_mod is None:
        return False
    fields = set(getattr(vanilla_mod.GaussianRasterizationSettings, "_fields", ()))
    return _VANILLA_SOF_RASTERIZER_FIELDS.issubset(fields)


def _make_subpixel_offset(viewpoint_camera, subpixel_offset):
    if subpixel_offset is not None:
        return subpixel_offset
    return torch.zeros(
        (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
        dtype=torch.float32,
        device="cuda",
    )


def _build_screenspace_points(pc: GaussianModel):
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz,
            dtype=pc.get_xyz.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    return screenspace_points


def _resolve_color_inputs(viewpoint_camera, pc: GaussianModel, pipe, override_color):
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    return shs, colors_precomp


def _base_render_output(rendered_image, screenspace_points, radii):
    output = {
        "render": rendered_image[:3] if rendered_image.ndim >= 3 and rendered_image.shape[0] > 3 else rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    if rendered_image.ndim >= 3 and rendered_image.shape[0] > 3:
        output["render_full"] = rendered_image
    return output


def _merged_render_output(rendered_image, screenspace_points, radii):
    output = {
        "render": rendered_image[:3],
        "render_full": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    if rendered_image.shape[0] >= 6:
        output["normal"] = rendered_image[3:6]
    if rendered_image.shape[0] >= 7:
        output["depth"] = rendered_image[6:7]
    if rendered_image.shape[0] >= 8:
        output["alpha"] = rendered_image[7:8]
    if rendered_image.shape[0] >= 9:
        output["distortion"] = rendered_image[8:9]
    if rendered_image.shape[0] >= 10:
        output["extent"] = rendered_image[9:10]
    return output


def _render_mip_base(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    kernel_size: float,
    scaling_modifier=1.0,
    override_color=None,
    subpixel_offset=None,
):
    screenspace_points = _build_screenspace_points(pc)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    subpixel_offset = _make_subpixel_offset(viewpoint_camera, subpixel_offset)
    splat_args = ExtendedSettings() if "settings" in _RASTER_SETTINGS_FIELDS and ExtendedSettings is not None else None
    debug_vis = DebugVisualization() if "debug_data" in _RASTER_SETTINGS_FIELDS and DebugVisualization is not None else None

    raster_settings_kwargs = {
        "image_height": int(viewpoint_camera.image_height),
        "image_width": int(viewpoint_camera.image_width),
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "bg": bg_color,
        "scale_modifier": scaling_modifier,
        "viewmatrix": viewpoint_camera.world_view_transform,
        "projmatrix": viewpoint_camera.full_proj_transform,
        "sh_degree": pc.active_sh_degree,
        "campos": viewpoint_camera.camera_center,
        "prefiltered": False,
        "debug": pipe.debug,
    }
    if "kernel_size" in _RASTER_SETTINGS_FIELDS:
        raster_settings_kwargs["kernel_size"] = kernel_size
    if "subpixel_offset" in _RASTER_SETTINGS_FIELDS:
        raster_settings_kwargs["subpixel_offset"] = subpixel_offset
    if "inv_viewprojmatrix" in _RASTER_SETTINGS_FIELDS:
        raster_settings_kwargs["inv_viewprojmatrix"] = viewpoint_camera.full_proj_transform_inverse
    if "settings" in _RASTER_SETTINGS_FIELDS:
        raster_settings_kwargs["settings"] = splat_args
    if "debug_data" in _RASTER_SETTINGS_FIELDS:
        raster_settings_kwargs["debug_data"] = debug_vis

    raster_settings = GaussianRasterizationSettings(**raster_settings_kwargs)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    shs, colors_precomp = _resolve_color_inputs(viewpoint_camera, pc, pipe, override_color)
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return _base_render_output(rendered_image, screenspace_points, radii)


def _render_mip_sof_merged(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    kernel_size: float,
    scaling_modifier=1.0,
    override_color=None,
    subpixel_offset=None,
    splat_args=None,
    debug_vis=None,
):
    if not supports_merged_sof_rasterizer():
        raise RuntimeError(
            "Merged mip+SOF rasterizer requested, but the active diff_gaussian_rasterization "
            "extension does not expose SOF-style settings/output fields yet. Rebuild the local "
            "submodule extension after wiring the merged CUDA interface."
        )

    screenspace_points = _build_screenspace_points(pc)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    subpixel_offset = _make_subpixel_offset(viewpoint_camera, subpixel_offset)
    if splat_args is None:
        splat_args = ExtendedSettings()
    if debug_vis is None and DebugVisualization is not None:
        debug_vis = DebugVisualization()

    raster_settings_kwargs = {
        "image_height": int(viewpoint_camera.image_height),
        "image_width": int(viewpoint_camera.image_width),
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "bg": bg_color,
        "scale_modifier": scaling_modifier,
        "viewmatrix": viewpoint_camera.world_view_transform,
        "projmatrix": viewpoint_camera.full_proj_transform,
        "inv_viewprojmatrix": viewpoint_camera.full_proj_transform_inverse,
        "sh_degree": pc.active_sh_degree,
        "campos": viewpoint_camera.camera_center,
        "prefiltered": False,
        "settings": splat_args,
        "debug_data": debug_vis,
        "debug": pipe.debug,
    }
    if "kernel_size" in _RASTER_SETTINGS_FIELDS:
        raster_settings_kwargs["kernel_size"] = kernel_size
    if "subpixel_offset" in _RASTER_SETTINGS_FIELDS:
        raster_settings_kwargs["subpixel_offset"] = subpixel_offset

    raster_settings = GaussianRasterizationSettings(**raster_settings_kwargs)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # Keep merged SOF aux rendering on top of the same stabilized mip inputs we
    # already trust in the plain mip path. Letting the merged CUDA branch
    # rebuild filter/view2gaussian internally introduced another source of
    # semantic drift and, on some machines, pathological binning explosions.
    filter_3d = None
    scales = pc.get_scaling_with_3D_filter
    opacity = pc.get_opacity_with_3D_filter

    cov3D_precomp = None
    rotations = pc.get_rotation

    # Always precompute the SOF view2gaussian payload in Python for the merged
    # route. This matches the 10-value packed layout consumed by the aux losses
    # and avoids depending on a second internal codepath during merged smoke
    # bring-up.
    view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    shs, colors_precomp = _resolve_color_inputs(viewpoint_camera, pc, pipe, override_color)
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp,
        filter_3d=filter_3d,
    )
    return _merged_render_output(rendered_image, screenspace_points, radii)


def _render_sof_vanilla(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    kernel_size: float,
    scaling_modifier=1.0,
    override_color=None,
    subpixel_offset=None,
    splat_args=None,
    debug_vis=None,
):
    vanilla_mod = _load_vanilla_sof_module()
    if vanilla_mod is None or not supports_vanilla_sof_rasterizer():
        detail = ""
        if _VANILLA_SOF_IMPORT_ERROR is not None:
            detail = f" Import error: {_VANILLA_SOF_IMPORT_ERROR}"
        raise RuntimeError(
            "Vanilla SOF rasterizer requested, but diff_gaussian_rasterization_sof_vanilla "
            f"is unavailable.{detail}"
        )

    screenspace_points = _build_screenspace_points(pc)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    if splat_args is None:
        splat_args = vanilla_mod.ExtendedSettings()
    if debug_vis is None:
        debug_vis = vanilla_mod.DebugVisualization()

    raster_settings = vanilla_mod.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        inv_viewprojmatrix=viewpoint_camera.full_proj_transform_inverse,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        settings=splat_args,
        debug_data=debug_vis,
        debug=pipe.debug,
    )
    rasterizer = vanilla_mod.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    if pipe.compute_filter3D_python:
        filter_3d = None
        scales = pc.get_scaling_with_3D_filter
        opacity = pc.get_opacity_with_3D_filter
    else:
        filter_3d = pc.filter_3D
        scales = pc.get_scaling
        opacity = pc.get_opacity

    cov3D_precomp = None
    rotations = pc.get_rotation

    view2gaussian_precomp = None
    if pipe.compute_view2gaussian_python:
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    shs, colors_precomp = _resolve_color_inputs(viewpoint_camera, pc, pipe, override_color)
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp,
        filter_3d=filter_3d,
    )
    return _merged_render_output(rendered_image, screenspace_points, radii)


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    kernel_size: float,
    scaling_modifier=1.0,
    override_color=None,
    subpixel_offset=None,
    splat_args=None,
    debug_vis=None,
    return_aux=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU.
    """

    wants_vanilla_path = bool(getattr(pipe, "use_vanilla_sof_rasterizer", False))
    wants_merged_path = bool(
        getattr(pipe, "use_merged_sof_rasterizer", False) or splat_args is not None or return_aux
    )
    if wants_vanilla_path and wants_merged_path and not getattr(pipe, "use_merged_sof_rasterizer", False):
        wants_merged_path = False
    if wants_vanilla_path and getattr(pipe, "use_merged_sof_rasterizer", False):
        raise RuntimeError("Vanilla SOF rasterizer and merged SOF rasterizer cannot be enabled at the same time.")
    if wants_vanilla_path:
        return _render_sof_vanilla(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            bg_color=bg_color,
            kernel_size=kernel_size,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            subpixel_offset=subpixel_offset,
            splat_args=splat_args,
            debug_vis=debug_vis,
        )
    if wants_merged_path:
        return _render_mip_sof_merged(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            bg_color=bg_color,
            kernel_size=kernel_size,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            subpixel_offset=subpixel_offset,
            splat_args=splat_args,
            debug_vis=debug_vis,
        )

    return _render_mip_base(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        kernel_size=kernel_size,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
        subpixel_offset=subpixel_offset,
    )

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

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def supports_merged_features() -> bool:
    return hasattr(_C, "compute_filter_3d") and hasattr(_C, "integrate_gaussians_to_points")


class SortMode(IntEnum):
    GLOBAL = 0
    PPX_FULL = 1
    PPX_KBUFFER = 2
    HIER = 3


class GlobalSortOrder(IntEnum):
    Z_DEPTH = 0
    DISTANCE = 1
    PTD_CENTER = 2
    PTD_MAX = 3
    MIN_Z_BOUNDING = 4


class DebugVisualizationType(IntEnum):
    DISABLED = 0
    GAUSSIANCOUNTPERTILE = 1
    GAUSSIANCOUNTPERPIXEL = 2
    TRANSMITTANCE = 3
    DEPTH = 4
    OPACITY = 5
    DISTORTION = 6
    EXTENT_LOSS = 7
    DEPTH_INDEX = 8
    DEBUG = 9


@dataclass
class SortQueueSizes:
    tile_4x4: int = 64
    tile_2x2: int = 8
    per_pixel: int = 4

    def set_value(self, key, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)


@dataclass
class SortSettings:
    queue_sizes: SortQueueSizes = field(default_factory=SortQueueSizes)
    sort_mode: SortMode = SortMode.GLOBAL
    sort_order: GlobalSortOrder = GlobalSortOrder.Z_DEPTH

    def set_value(self, key, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            self.queue_sizes.set_value(key, value)


@dataclass
class CullingSettings:
    rect_bounding: bool = False
    tight_opacity_bounding: bool = False
    tile_based_culling: bool = False
    hierarchical_4x4_culling: bool = False

    def set_value(self, key, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)


@dataclass
class MeshingSettings:
    alpha_early_stop: bool = False
    return_color: bool = False

    def set_value(self, key, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)


@dataclass
class ExtendedSettings:
    sort_settings: SortSettings = field(default_factory=SortSettings)
    culling_settings: CullingSettings = field(default_factory=CullingSettings)
    meshing_settings: MeshingSettings = field(default_factory=MeshingSettings)
    load_balancing: bool = False
    proper_ewa_scaling: bool = False
    exact_depth: bool = False
    detach_alpha: bool = False
    far_plane: float = 100.0
    detach_alpha_extent: bool = False
    include_alpha: bool = True
    render_opacity: bool = False

    def set_value(self, key, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            self.sort_settings.set_value(key, value)
            self.culling_settings.set_value(key, value)
            self.meshing_settings.set_value(key, value)

    def to_dict(self):
        return asdict(self)


@dataclass
class DebugVisualization:
    type: DebugVisualizationType = DebugVisualizationType.DISABLED
    debugX: int = 0
    debugY: int = 0
    min: float = 0.0
    max: float = 1.0
    debug_normalize: bool = False
    timing_enabled: bool = False
    printing_enabled: bool = True
    colormap: bool = True
    precision: int = 3

    def to_dict(self):
        data = asdict(self)
        data["type"] = int(self.type)
        return data


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


def rasterize_gaussians_merged(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    view2gaussian_precomp,
    filter_3d,
    raster_settings,
):
    return _RasterizeGaussiansMerged.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        view2gaussian_precomp,
        filter_3d,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.subpixel_offset,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.subpixel_offset,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
            ) = _C.rasterize_gaussians_backward(*args)

        return (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )


class _RasterizeGaussiansMerged(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        view2gaussian_precomp,
        filter_3d,
        raster_settings,
    ):
        if not supports_merged_features():
            raise RuntimeError(
                "The active diff_gaussian_rasterization extension does not support merged SOF "
                "features yet. Rebuild the local submodule after wiring the merged CUDA API."
            )

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            view2gaussian_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.inv_viewprojmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.subpixel_offset,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.settings.to_dict(),
            raster_settings.debug_data.to_dict(),
            filter_3d,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            view2gaussian_precomp,
            filter_3d,
            radii,
            sh,
            color,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            view2gaussian_precomp,
            filter_3d,
            radii,
            sh,
            color,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
            opacities,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            view2gaussian_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.inv_viewprojmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.subpixel_offset,
            color,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.settings.to_dict(),
            filter_3d,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                    grad_view2gaussian_precomp,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
                grad_view2gaussian_precomp,
            ) = _C.rasterize_gaussians_backward(*args)

        return (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_view2gaussian_precomp,
            None,
            None,
        )


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    kernel_size: float = 0.0
    subpixel_offset: Optional[torch.Tensor] = None
    bg: Optional[torch.Tensor] = None
    scale_modifier: float = 1.0
    viewmatrix: Optional[torch.Tensor] = None
    projmatrix: Optional[torch.Tensor] = None
    inv_viewprojmatrix: Optional[torch.Tensor] = None
    sh_degree: int = 0
    campos: Optional[torch.Tensor] = None
    prefiltered: bool = False
    settings: Optional[ExtendedSettings] = None
    debug_data: Optional[DebugVisualization] = None
    debug: bool = False


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
            )
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        view2gaussian_precomp=None,
        filter_3d=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception("Please provide excatly one of either SHs or precomputed colors!")

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        wants_merged = (
            view2gaussian_precomp is not None
            or filter_3d is not None
            or raster_settings.inv_viewprojmatrix is not None
            or raster_settings.settings is not None
            or raster_settings.debug_data is not None
        )
        wants_merged = wants_merged or supports_merged_features()

        if wants_merged:
            if view2gaussian_precomp is None:
                view2gaussian_precomp = torch.Tensor([])
            if filter_3d is None:
                filter_3d = torch.Tensor([])
            if raster_settings.inv_viewprojmatrix is None:
                inv_viewprojmatrix = torch.inverse(raster_settings.projmatrix)
            else:
                inv_viewprojmatrix = raster_settings.inv_viewprojmatrix
            if raster_settings.settings is None:
                settings = ExtendedSettings()
                settings.proper_ewa_scaling = True
            else:
                settings = raster_settings.settings
            if raster_settings.debug_data is None:
                debug_data = DebugVisualization()
            else:
                debug_data = raster_settings.debug_data
            merged_settings = raster_settings._replace(
                inv_viewprojmatrix=inv_viewprojmatrix,
                settings=settings,
                debug_data=debug_data,
            )
            return rasterize_gaussians_merged(
                means3D,
                means2D,
                shs,
                colors_precomp,
                opacities,
                scales,
                rotations,
                cov3D_precomp,
                view2gaussian_precomp,
                filter_3d,
                merged_settings,
            )

        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

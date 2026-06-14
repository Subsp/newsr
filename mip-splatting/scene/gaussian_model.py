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

import torch
import numpy as np
import warnings
from enum import IntEnum
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from scipy.spatial import cKDTree
try:
    from simple_knn._C import distCUDA2 as _distCUDA2_cuda
except ImportError:
    _distCUDA2_cuda = None
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    if _distCUDA2_cuda is not None:
        return _distCUDA2_cuda(points)

    warnings.warn(
        "simple_knn._C is unavailable; falling back to scipy cKDTree for "
        "Gaussian init distances. This is slower but preserves the baseline.",
        stacklevel=2,
    )

    points_np = points.detach().cpu().numpy().astype(np.float32, copy=False)
    point_count = points_np.shape[0]
    if point_count == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    if point_count == 1:
        return torch.full((1,), 1e-7, dtype=torch.float32, device=points.device)

    neighbor_count = min(4, point_count)
    tree = cKDTree(points_np)
    distances, _ = tree.query(points_np, k=neighbor_count, workers=1)
    if neighbor_count == 2:
        nearest = np.square(distances[:, 1:2]).mean(axis=1)
    else:
        nearest = np.square(distances[:, 1:neighbor_count]).mean(axis=1)
    return torch.from_numpy(nearest.astype(np.float32, copy=False)).to(points.device)


class GaussianSourceTag(IntEnum):
    ORIGINAL = 0
    PRIOR_INJECTED = 1
    EXTENSION_PROBE = 2


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._source_tag = torch.empty(0, dtype=torch.int32, device="cuda")
        self._seed_id = torch.empty(0, dtype=torch.int64, device="cuda")
        self._root_id = torch.empty(0, dtype=torch.int64, device="cuda")
        self._generation = torch.empty(0, dtype=torch.int32, device="cuda")
        self._edge_touched = torch.empty(0, dtype=torch.bool, device="cuda")
        self._edge_touch_iter = torch.empty(0, dtype=torch.int32, device="cuda")
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.get_tracking_state(),
        )
    
    def restore(self, model_args, training_args):
        tracking_state = None
        def _default_filter():
            return torch.zeros(
                (self._xyz.shape[0], 1),
                dtype=self._xyz.dtype,
                device=self._xyz.device,
            )

        if len(model_args) == 12:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            self.filter_3D = _default_filter()
        elif len(model_args) == 13:
            if isinstance(model_args[-1], dict):
                (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self.max_radii2D,
                    xyz_gradient_accum,
                    denom,
                    opt_dict,
                    self.spatial_lr_scale,
                    tracking_state,
                ) = model_args
                self.filter_3D = _default_filter()
            else:
                (
                    self.active_sh_degree,
                    self._xyz,
                    self._features_dc,
                    self._features_rest,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self.max_radii2D,
                    xyz_gradient_accum,
                    denom,
                    opt_dict,
                    self.spatial_lr_scale,
                    self.filter_3D,
                ) = model_args
        elif len(model_args) == 14:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.filter_3D,
                tracking_state,
            ) = model_args
        else:
            raise ValueError(f"Unsupported Gaussian checkpoint payload length: {len(model_args)}")
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if tracking_state is None:
            self.init_tracking_state(self._xyz.shape[0])
        else:
            self.restore_tracking_state(tracking_state)

    def _default_tracking_tensor(self, length: int, value, dtype: torch.dtype):
        if length <= 0:
            return torch.empty((0,), dtype=dtype, device="cuda")
        return torch.full((int(length),), value, dtype=dtype, device="cuda")

    def init_tracking_state(
        self,
        length: int,
        source_tag: int = 0,
        seed_id: int = -1,
        root_id: int = -1,
        generation: int = 0,
    ):
        self._source_tag = self._default_tracking_tensor(length, source_tag, torch.int32)
        self._seed_id = self._default_tracking_tensor(length, seed_id, torch.int64)
        self._root_id = self._default_tracking_tensor(length, root_id, torch.int64)
        self._generation = self._default_tracking_tensor(length, generation, torch.int32)
        self._edge_touched = self._default_tracking_tensor(length, False, torch.bool)
        self._edge_touch_iter = self._default_tracking_tensor(length, -1, torch.int32)

    def get_tracking_state(self):
        return {
            "source_tag": self._source_tag,
            "seed_id": self._seed_id,
            "root_id": self._root_id,
            "generation": self._generation,
            "edge_touched": self._edge_touched,
            "edge_touch_iter": self._edge_touch_iter,
        }

    def restore_tracking_state(self, tracking_state):
        self._source_tag = tracking_state["source_tag"].to(device="cuda", dtype=torch.int32)
        self._seed_id = tracking_state["seed_id"].to(device="cuda", dtype=torch.int64)
        self._root_id = tracking_state.get(
            "root_id",
            self._default_tracking_tensor(self._xyz.shape[0], -1, torch.int64),
        ).to(device="cuda", dtype=torch.int64)
        self._generation = tracking_state["generation"].to(device="cuda", dtype=torch.int32)
        self._edge_touched = tracking_state.get(
            "edge_touched",
            self._default_tracking_tensor(self._xyz.shape[0], False, torch.bool),
        ).to(device="cuda", dtype=torch.bool)
        self._edge_touch_iter = tracking_state.get(
            "edge_touch_iter",
            self._default_tracking_tensor(self._xyz.shape[0], -1, torch.int32),
        ).to(device="cuda", dtype=torch.int32)

    def save_tracking_metadata(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            mkdir_p(directory)
        torch.save(
            {
                "source_tag": self._source_tag.detach().cpu(),
                "seed_id": self._seed_id.detach().cpu(),
                "root_id": self._root_id.detach().cpu(),
                "generation": self._generation.detach().cpu(),
                "edge_touched": self._edge_touched.detach().cpu(),
                "edge_touch_iter": self._edge_touch_iter.detach().cpu(),
            },
            path,
        )

    def load_tracking_metadata(self, path: str):
        if not os.path.exists(path):
            self.init_tracking_state(self._xyz.shape[0])
            return
        tracking_state = torch.load(path, map_location="cpu")
        self.restore_tracking_state(tracking_state)

    def _build_tracking_extension(
        self,
        length: int,
        source_tag=None,
        seed_id=None,
        root_id=None,
        generation=None,
        edge_touched=None,
        edge_touch_iter=None,
    ):
        if source_tag is None:
            source_tag = self._default_tracking_tensor(length, int(GaussianSourceTag.ORIGINAL), torch.int32)
        if seed_id is None:
            seed_id = self._default_tracking_tensor(length, -1, torch.int64)
        if root_id is None:
            root_id = self._default_tracking_tensor(length, -1, torch.int64)
        if generation is None:
            generation = self._default_tracking_tensor(length, 0, torch.int32)
        if edge_touched is None:
            edge_touched = self._default_tracking_tensor(length, False, torch.bool)
        if edge_touch_iter is None:
            edge_touch_iter = self._default_tracking_tensor(length, -1, torch.int32)
        return {
            "source_tag": source_tag.to(device="cuda", dtype=torch.int32),
            "seed_id": seed_id.to(device="cuda", dtype=torch.int64),
            "root_id": root_id.to(device="cuda", dtype=torch.int64),
            "generation": generation.to(device="cuda", dtype=torch.int32),
            "edge_touched": edge_touched.to(device="cuda", dtype=torch.bool),
            "edge_touch_iter": edge_touch_iter.to(device="cuda", dtype=torch.int32),
        }

    def _append_tracking_extension(self, tracking_state):
        self._source_tag = torch.cat((self._source_tag, tracking_state["source_tag"]), dim=0)
        self._seed_id = torch.cat((self._seed_id, tracking_state["seed_id"]), dim=0)
        root_id = tracking_state.get(
            "root_id",
            self._default_tracking_tensor(tracking_state["source_tag"].shape[0], -1, torch.int64),
        )
        self._root_id = torch.cat((self._root_id, root_id.to(device="cuda", dtype=torch.int64)), dim=0)
        self._generation = torch.cat((self._generation, tracking_state["generation"]), dim=0)
        self._edge_touched = torch.cat((self._edge_touched, tracking_state["edge_touched"]), dim=0)
        self._edge_touch_iter = torch.cat((self._edge_touch_iter, tracking_state["edge_touch_iter"]), dim=0)

    def _inherit_tracking(self, indices: torch.Tensor, repeats: int = 1, generation_offset: int = 1):
        source_tag = self._source_tag[indices].repeat(repeats)
        seed_id = self._seed_id[indices].repeat(repeats)
        root_id = self._root_id[indices].repeat(repeats)
        generation = (self._generation[indices] + generation_offset).repeat(repeats)
        edge_touched = self._edge_touched[indices].repeat(repeats)
        edge_touch_iter = self._edge_touch_iter[indices].repeat(repeats)
        return self._build_tracking_extension(
            source_tag.shape[0],
            source_tag,
            seed_id,
            root_id,
            generation,
            edge_touched,
            edge_touch_iter,
        )

    def mark_edge_touched(self, mask, iteration: int):
        if mask is None:
            return
        if mask.ndim != 1 or mask.shape[0] != self._xyz.shape[0]:
            raise ValueError("edge touched mask must be a 1D tensor aligned with current gaussians")
        mask = mask.to(device="cuda", dtype=torch.bool)
        if not torch.any(mask):
            return
        self._edge_touched[mask] = True
        self._edge_touch_iter[mask] = int(iteration)

    def _normalize_candidate_mask(self, candidate_mask, *, label: str):
        if candidate_mask is None:
            return None
        if candidate_mask.ndim != 1 or candidate_mask.shape[0] != self._xyz.shape[0]:
            raise ValueError(f"{label} must be a 1D tensor aligned with current gaussians")
        return candidate_mask.to(device="cuda", dtype=torch.bool)

    @torch.no_grad()
    def inject_prior_hf_seeds(
        self,
        candidate_mask,
        *,
        max_points: int,
        scale_multiplier: float = 0.35,
        opacity: float = 0.02,
        jitter_scale: float = 0.15,
        original_only: bool = True,
    ) -> int:
        if candidate_mask is None or int(max_points) <= 0:
            return 0
        if candidate_mask.ndim != 1 or candidate_mask.shape[0] != self._xyz.shape[0]:
            raise ValueError("HF seed mask must be a 1D tensor aligned with current gaussians")

        candidate_mask = candidate_mask.to(device="cuda", dtype=torch.bool)
        if original_only:
            candidate_mask = candidate_mask & (self._source_tag == int(GaussianSourceTag.ORIGINAL))
        candidate_idx = candidate_mask.nonzero(as_tuple=True)[0]
        if candidate_idx.numel() == 0:
            return 0

        if candidate_idx.numel() > int(max_points):
            perm = torch.randperm(candidate_idx.numel(), device=candidate_idx.device)[: int(max_points)]
            candidate_idx = candidate_idx[perm]

        parent_scales = self.get_scaling[candidate_idx]
        seed_scales = parent_scales * max(float(scale_multiplier), 1e-4)
        if float(jitter_scale) > 0.0:
            jitter = torch.randn_like(seed_scales) * seed_scales * float(jitter_scale)
        else:
            jitter = torch.zeros_like(seed_scales)

        new_xyz = self._xyz[candidate_idx].detach() + jitter
        new_features_dc = self._features_dc[candidate_idx].detach().clone()
        new_features_rest = self._features_rest[candidate_idx].detach().clone()
        opacity = min(max(float(opacity), 1e-5), 1.0 - 1e-5)
        new_opacities = self.inverse_opacity_activation(
            torch.full((candidate_idx.shape[0], 1), opacity, dtype=self._opacity.dtype, device="cuda")
        )
        new_scaling = self.scaling_inverse_activation(seed_scales.clamp(min=1e-7))
        new_rotation = self._rotation[candidate_idx].detach().clone()

        tracking_state = self._build_tracking_extension(
            candidate_idx.shape[0],
            source_tag=self._default_tracking_tensor(
                candidate_idx.shape[0],
                int(GaussianSourceTag.PRIOR_INJECTED),
                torch.int32,
            ),
            seed_id=candidate_idx.to(device="cuda", dtype=torch.int64),
            root_id=torch.where(
                self._root_id[candidate_idx] >= 0,
                self._root_id[candidate_idx],
                candidate_idx.to(device="cuda", dtype=torch.int64),
            ),
            generation=(self._generation[candidate_idx] + 1).to(device="cuda", dtype=torch.int32),
            edge_touched=self._default_tracking_tensor(candidate_idx.shape[0], True, torch.bool),
            edge_touch_iter=self._edge_touch_iter[candidate_idx].clone(),
        )
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            tracking_state=tracking_state,
        )
        return int(candidate_idx.shape[0])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_view2gaussian(self, viewmatrix: torch.Tensor) -> torch.Tensor:
        r = self._rotation
        norm = torch.sqrt(
            r[:, 0] * r[:, 0]
            + r[:, 1] * r[:, 1]
            + r[:, 2] * r[:, 2]
            + r[:, 3] * r[:, 3]
        )
        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device=q.device, dtype=q.dtype)

        qr = q[:, 0]
        qx = q[:, 1]
        qy = q[:, 2]
        qz = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
        R[:, 0, 1] = 2 * (qx * qy - qr * qz)
        R[:, 0, 2] = 2 * (qx * qz + qr * qy)
        R[:, 1, 0] = 2 * (qx * qy + qr * qz)
        R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
        R[:, 1, 2] = 2 * (qy * qz - qr * qx)
        R[:, 2, 0] = 2 * (qx * qz - qr * qy)
        R[:, 2, 1] = 2 * (qy * qz + qr * qx)
        R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)

        xyz = self.get_xyz
        gaussian_to_world = torch.zeros((xyz.shape[0], 4, 4), device=xyz.device, dtype=xyz.dtype)
        gaussian_to_world[:, :3, :3] = R
        gaussian_to_world[:, :3, 3] = xyz
        gaussian_to_world[:, 3, 3] = 1.0

        viewmatrix = viewmatrix.transpose(0, 1)
        gaussian_to_view = viewmatrix @ gaussian_to_world

        view_rotation = gaussian_to_view[:, :3, :3]
        view_translation = gaussian_to_view[:, :3, 3]
        inv_translation = torch.bmm(
            -view_rotation.transpose(1, 2),
            view_translation[..., None],
        )[..., 0]

        view_to_gaussian = torch.zeros((xyz.shape[0], 4, 4), device=xyz.device, dtype=xyz.dtype)
        view_to_gaussian[:, :3, :3] = view_rotation.transpose(1, 2)
        view_to_gaussian[:, :3, 3] = inv_translation
        view_to_gaussian[:, 3, 3] = 1.0
        view_to_gaussian = view_to_gaussian.transpose(2, 1).contiguous()

        scales = self.get_scaling_with_3D_filter
        inv_scale_sq = 1.0 / (scales ** 2)
        view_to_gaussian_rot = view_to_gaussian[:, :3, :3].transpose(1, 2)
        view_to_gaussian_trans = view_to_gaussian[:, 3:, :3]

        C = torch.sum((view_to_gaussian_trans ** 2) * inv_scale_sq[:, None, :], dim=2)
        inv_scale_sq_rot = inv_scale_sq[:, :, None] * view_to_gaussian_rot
        B = view_to_gaussian_trans @ inv_scale_sq_rot
        Sigma = view_to_gaussian_rot.transpose(1, 2) @ inv_scale_sq_rot
        return torch.cat(
            [
                Sigma[:, :, 0],
                Sigma[:, 1:, 1],
                Sigma[:, 2:, 2],
                B.squeeze(1),
                C,
            ],
            dim=1,
        )

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.init_tracking_state(self.get_xyz.shape[0], source_tag=int(GaussianSourceTag.ORIGINAL))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, exclude_filter=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_fused_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # fuse opacity and scale
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities = inverse_sigmoid(current_opacity_with_filter).detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling_with_3D_filter).detach().cpu().numpy()
        
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = inverse_sigmoid(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree
        self.init_tracking_state(self.get_xyz.shape[0], source_tag=int(GaussianSourceTag.ORIGINAL))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._source_tag = self._source_tag[valid_points_mask]
        self._seed_id = self._seed_id[valid_points_mask]
        self._root_id = self._root_id[valid_points_mask]
        self._generation = self._generation[valid_points_mask]
        self._edge_touched = self._edge_touched[valid_points_mask]
        self._edge_touch_iter = self._edge_touch_iter[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        tracking_state=None,
    ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if tracking_state is None:
            tracking_state = self._build_tracking_extension(
                new_xyz.shape[0],
                source_tag=self._default_tracking_tensor(
                    new_xyz.shape[0],
                    int(GaussianSourceTag.EXTENSION_PROBE),
                    torch.int32,
                ),
            )
        self._append_tracking_extension(tracking_state)

        #TODO Maybe we don't need to reset the value, it's better to use moving average instead of reset the value
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        grads_abs,
        grad_abs_threshold,
        scene_extent,
        N=2,
        candidate_mask=None,
    ):
        candidate_mask = self._normalize_candidate_mask(candidate_mask, label="densify split candidate mask")
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if candidate_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, candidate_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        tracking_state = self._inherit_tracking(selected_pts_mask.nonzero(as_tuple=True)[0], repeats=N)
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            tracking_state=tracking_state,
        )

        if candidate_mask is not None:
            candidate_mask = torch.cat(
                (
                    candidate_mask,
                    torch.ones(N * int(selected_pts_mask.sum().item()), device="cuda", dtype=torch.bool),
                ),
                dim=0,
            )
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        if candidate_mask is not None:
            candidate_mask = candidate_mask[~prune_filter]
        return candidate_mask

    def densify_and_clone(
        self,
        grads,
        grad_threshold,
        grads_abs,
        grad_abs_threshold,
        scene_extent,
        candidate_mask=None,
    ):
        candidate_mask = self._normalize_candidate_mask(candidate_mask, label="densify clone candidate mask")
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if candidate_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, candidate_mask)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        tracking_state = self._inherit_tracking(selected_pts_mask.nonzero(as_tuple=True)[0], repeats=1)
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            tracking_state=tracking_state,
        )
        if candidate_mask is not None:
            candidate_mask = torch.cat(
                (
                    candidate_mask,
                    torch.ones(int(selected_pts_mask.sum().item()), device="cuda", dtype=torch.bool),
                ),
                dim=0,
            )
        return candidate_mask

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        candidate_mask=None,
        protect_source_tag=None,
        protect_recent_iters: int = 0,
        current_iteration: int = 0,
    ):
        candidate_mask = self._normalize_candidate_mask(candidate_mask, label="densify prune candidate mask")
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        before = self._xyz.shape[0]
        candidate_mask = self.densify_and_clone(
            grads,
            max_grad,
            grads_abs,
            Q,
            extent,
            candidate_mask=candidate_mask,
        )
        clone = self._xyz.shape[0]
        candidate_mask = self.densify_and_split(
            grads,
            max_grad,
            grads_abs,
            Q,
            extent,
            candidate_mask=candidate_mask,
        )
        split = self._xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        if candidate_mask is not None:
            prune_mask = torch.logical_and(prune_mask, candidate_mask)
        if protect_source_tag is not None and int(protect_recent_iters) > 0:
            protect_mask = self._source_tag == int(protect_source_tag)
            protect_mask = protect_mask & (self._edge_touch_iter >= 0)
            protect_mask = protect_mask & (
                (int(current_iteration) - self._edge_touch_iter) < int(protect_recent_iters)
            )
            prune_mask = torch.logical_and(prune_mask, ~protect_mask)
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]
        # torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #TODO maybe use max instead of average
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1

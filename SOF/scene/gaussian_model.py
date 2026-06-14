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

import os
import warnings
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np
import torch
import trimesh
from einops import einsum
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from torch import nn

from arguments import BoundingSetting, MeshingSettings
from diff_gaussian_rasterization._C import compute_filter_3d
from scene.appearance_network import AppearanceEmbedding, PGSREmbedding
from scene.cameras import Camera
from utils.system_utils import mkdir_p
try:
    from simple_knn._C import distCUDA2 as _distCUDA2_cuda
except ImportError:
    _distCUDA2_cuda = None
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.reloc_utils import compute_relocation_cuda
from utils.sh_utils import RGB2SH


class GaussianSourceTag(IntEnum):
    ORIGINAL = 0
    PRIOR_INJECTED = 1
    EXTENSION_PROBE = 2


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    if _distCUDA2_cuda is not None:
        return _distCUDA2_cuda(points)

    warnings.warn(
        "simple_knn._C is unavailable; falling back to scipy cKDTree for "
        "Gaussian init distances. This is slower but should preserve behavior.",
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

@torch.no_grad()
def get_frustum_mask_batched(points: torch.Tensor, cameras: List[Camera], near: float = 0.02, far: float = 1e6):
    
    N = 200_000
    
    mask = torch.empty(0, device='cuda', dtype=torch.bool)
    number_of_batches = np.ceil(len(points)/N).astype(int)
    for i in range(number_of_batches):        
        mask = torch.cat((mask, get_frustum_mask(points[N*i: N * (i+1)], cameras, near, far)))
    return mask
    
@torch.no_grad()
def get_frustum_mask(points: torch.Tensor, cameras: List[Camera], near: float = 0.02, far: float = 1e6):
    H, W = cameras[0].image_height, cameras[0].image_width

    intrinsics = torch.stack(
        [
            torch.Tensor(
                [[cam.focal_x, 0, W / 2],
                 [0, cam.focal_y, H / 2],
                 [0, 0, 1]]
            ) for cam in cameras
        ], 
        dim=0
    ).to(points.device)

    # full_proj_matrices: (n_view, 4, 4)
    view_matrices = torch.stack(
        [cam.world_view_transform for cam in cameras], dim=0
    ).transpose(1, 2)

    ones = torch.ones_like(points[:, 0]).unsqueeze(-1)
    # homo_points: (N, 4)
    homo_points = torch.cat([points, ones], dim=-1)

    # uv_points: (n_view, N, 4, 4)
    # Apply batch matrix multiplication to get uv_points for all cameras
    view_points = einsum(view_matrices, homo_points, "n_view b c, N c -> n_view N b")
    view_points = view_points[:, :, :3]

    uv_points = einsum(intrinsics, view_points, "n_view b c, n_view N c -> n_view N b")

    z = uv_points[:, :, -1:]
    uv_points = uv_points[:, :, :2] / z
    u, v = uv_points[:, :, 0], uv_points[:, :, 1]

    # Optionally, we can apply near-far culling
    # Apply near-far culling
    depth = view_points[:, :, -1]
    cull_near_fars = (depth >= near) & (depth <= far)

    # Apply frustum mask
    mask = torch.any(cull_near_fars & (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1), dim=0)
    return mask
class GaussianModel:

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
    
    def get_view2gaussian(self, viewmatrix):
        r = self._rotation
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]
        
        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
        rots = R
        xyz = self.get_xyz
        N = xyz.shape[0]
        G2W = torch.zeros((N, 4, 4), device='cuda')
        G2W[:, :3, :3] = rots # TODO check if we need to transpose here
        G2W[:, :3, 3] = xyz
        G2W[:, 3, 3] = 1.0
        
        viewmatrix = viewmatrix.transpose(0, 1)
        G2V = viewmatrix @ G2W
        
        R = G2V[:, :3, :3]
        t = G2V[:, :3, 3]
        
        t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
        V2G = torch.zeros((N, 4, 4), device='cuda')
        V2G[:, :3, :3] = R.transpose(1, 2)
        V2G[:, :3, 3] = t2
        V2G[:, 3, 3] = 1.0
        
        # transpose view2gaussian to match glm in CUDA code
        V2G = V2G.transpose(2, 1).contiguous()
        
        # precompute results to reduce computation and IO
        scales = self.get_scaling_with_3D_filter
        S_inv_square = 1.0 / (scales ** 2)
        R = V2G[:, :3, :3].transpose(1, 2)
        t2 = V2G[:, 3:, :3]
        
        C = torch.sum((t2 ** 2) * S_inv_square[:, None, :], dim=2)
        S_inv_square_R = S_inv_square[:, :, None] * R
        B = t2 @ S_inv_square_R
        Sigma = R.transpose(1, 2) @ S_inv_square_R
        merged = torch.cat([Sigma[:, :, 0], Sigma[:, 1:, 1], Sigma[:, 2:, 2], B.squeeze(), C], dim=1)
        
        return merged

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


    def __init__(self, sh_degree : int, use_SBs : bool = False):
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
        self._source_tag = torch.empty(0, dtype=torch.int32, device="cuda")
        self._seed_id = torch.empty(0, dtype=torch.int64, device="cuda")
        self._generation = torch.empty(0, dtype=torch.int32, device="cuda")
        self._edge_touched = torch.empty(0, dtype=torch.bool, device="cuda")
        self._edge_touch_iter = torch.empty(0, dtype=torch.int32, device="cuda")
        self.optimizer = None
        self.use_SBs = use_SBs
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.filter_3D = torch.tensor([0.003,0.003,0.003]).cuda()
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
            self.filter_3D,
            self.capture_tracking_state(),
        )
    
    def restore(self, model_args, training_args, mesh_args, appearance_net):
        tracking_state = None
        def _default_filter():
            return torch.zeros(
                (self._xyz.shape[0], 1),
                dtype=self._xyz.dtype,
                device=self._xyz.device,
            )
        if len(model_args) == 12:
            (self.active_sh_degree,
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
            self.spatial_lr_scale) = model_args
            self.filter_3D = _default_filter()
        elif len(model_args) == 13:
            if isinstance(model_args[-1], dict):
                (self.active_sh_degree,
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
                tracking_state) = model_args
                self.filter_3D = _default_filter()
            else:
                (self.active_sh_degree,
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
                self.filter_3D) = model_args
        elif len(model_args) == 14:
            (self.active_sh_degree,
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
            tracking_state) = model_args
        else:
            raise RuntimeError(f"Unsupported GaussianModel checkpoint payload length: {len(model_args)}")
        self.training_setup(training_args, mesh_args, appearance_net)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.restore_tracking_state(tracking_state)

    def _default_tracking_tensor(self, length: int, fill_value: int, dtype: torch.dtype):
        return torch.full((length,), fill_value=fill_value, dtype=dtype, device="cuda")

    def init_tracking_state(self, length: int, source_tag: int = int(GaussianSourceTag.ORIGINAL), seed_id: int = -1, generation: int = 0):
        self._source_tag = self._default_tracking_tensor(length, source_tag, torch.int32)
        self._seed_id = self._default_tracking_tensor(length, seed_id, torch.int64)
        self._generation = self._default_tracking_tensor(length, generation, torch.int32)
        self._edge_touched = self._default_tracking_tensor(length, False, torch.bool)
        self._edge_touch_iter = self._default_tracking_tensor(length, -1, torch.int32)

    def capture_tracking_state(self):
        return {
            "source_tag": self._source_tag,
            "seed_id": self._seed_id,
            "generation": self._generation,
            "edge_touched": self._edge_touched,
            "edge_touch_iter": self._edge_touch_iter,
        }

    def restore_tracking_state(self, tracking_state: Optional[Dict[str, torch.Tensor]]):
        if tracking_state is None:
            self.init_tracking_state(self._xyz.shape[0])
            return
        self._source_tag = tracking_state["source_tag"].to(device="cuda", dtype=torch.int32)
        self._seed_id = tracking_state["seed_id"].to(device="cuda", dtype=torch.int64)
        self._generation = tracking_state["generation"].to(device="cuda", dtype=torch.int32)
        self._edge_touched = tracking_state.get(
            "edge_touched",
            self._default_tracking_tensor(self._xyz.shape[0], False, torch.bool).cpu(),
        ).to(device="cuda", dtype=torch.bool)
        self._edge_touch_iter = tracking_state.get(
            "edge_touch_iter",
            self._default_tracking_tensor(self._xyz.shape[0], -1, torch.int32).cpu(),
        ).to(device="cuda", dtype=torch.int32)

    def save_tracking_metadata(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            mkdir_p(directory)
        torch.save(
            {
                "source_tag": self._source_tag.detach().cpu(),
                "seed_id": self._seed_id.detach().cpu(),
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
        source_tag: Optional[torch.Tensor] = None,
        seed_id: Optional[torch.Tensor] = None,
        generation: Optional[torch.Tensor] = None,
        edge_touched: Optional[torch.Tensor] = None,
        edge_touch_iter: Optional[torch.Tensor] = None,
    ):
        if source_tag is None:
            source_tag = self._default_tracking_tensor(length, int(GaussianSourceTag.ORIGINAL), torch.int32)
        if seed_id is None:
            seed_id = self._default_tracking_tensor(length, -1, torch.int64)
        if generation is None:
            generation = self._default_tracking_tensor(length, 0, torch.int32)
        if edge_touched is None:
            edge_touched = self._default_tracking_tensor(length, False, torch.bool)
        if edge_touch_iter is None:
            edge_touch_iter = self._default_tracking_tensor(length, -1, torch.int32)
        return {
            "source_tag": source_tag.to(device="cuda", dtype=torch.int32),
            "seed_id": seed_id.to(device="cuda", dtype=torch.int64),
            "generation": generation.to(device="cuda", dtype=torch.int32),
            "edge_touched": edge_touched.to(device="cuda", dtype=torch.bool),
            "edge_touch_iter": edge_touch_iter.to(device="cuda", dtype=torch.int32),
        }

    def _append_tracking_extension(self, tracking_state: Dict[str, torch.Tensor]):
        self._source_tag = torch.cat((self._source_tag, tracking_state["source_tag"]), dim=0)
        self._seed_id = torch.cat((self._seed_id, tracking_state["seed_id"]), dim=0)
        self._generation = torch.cat((self._generation, tracking_state["generation"]), dim=0)
        self._edge_touched = torch.cat((self._edge_touched, tracking_state["edge_touched"]), dim=0)
        self._edge_touch_iter = torch.cat((self._edge_touch_iter, tracking_state["edge_touch_iter"]), dim=0)

    def _inherit_tracking(self, indices: torch.Tensor, repeats: int = 1, generation_offset: int = 1):
        source_tag = self._source_tag[indices].repeat(repeats)
        seed_id = self._seed_id[indices].repeat(repeats)
        generation = (self._generation[indices] + generation_offset).repeat(repeats)
        edge_touched = self._edge_touched[indices].repeat(repeats)
        edge_touch_iter = self._edge_touch_iter[indices].repeat(repeats)
        return self._build_tracking_extension(
            source_tag.shape[0],
            source_tag,
            seed_id,
            generation,
            edge_touched,
            edge_touch_iter,
        )

    def mark_edge_touched(self, mask: Optional[torch.Tensor], iteration: int):
        if mask is None:
            return
        if mask.ndim != 1 or mask.shape[0] != self._xyz.shape[0]:
            raise ValueError("edge touched mask must be a 1D tensor aligned with current gaussians")
        mask = mask.to(device="cuda", dtype=torch.bool)
        if not torch.any(mask):
            return
        self._edge_touched[mask] = True
        self._edge_touch_iter[mask] = int(iteration)

    # setter for scaling
    def set_scaling(self, new_scales):
        self._scaling = self.scaling_inverse_activation(new_scales)

    # setter for opacity
    def set_opacity(self, new_opacity):
        self._opacity = self.inverse_opacity_activation(new_opacity)

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
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @torch.no_grad()
    def compute_3D_filter(self, cameras, CUDA=True):
        print("Computing 3D filter")
        if not CUDA:
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
                valid_depth = xyz_cam[:, 2] > 0.2 # TODO remove hard coded value
                
                
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
        else:
            viewmatrices_torch = torch.stack([c.world_view_transform for c in cameras]).cuda()
            # initialize to (-1), if the filter is negative in the end, we know the point was never observed
            filter_3d_cuda = torch.ones_like(self._opacity) * -1
            compute_filter_3d(
                self.get_xyz,
                viewmatrices_torch,
                cameras[0].image_width, cameras[0].image_height,
                cameras[0].focal_x.item(), cameras[0].focal_y.item(),
                filter_3d_cuda
            )
            
            filter_3d_cuda[filter_3d_cuda < -0.2] = filter_3d_cuda.max()
            self.filter_3D = filter_3d_cuda

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, MCMC_init : bool):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        if self.use_SBs:
            pcd_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()

            spherical_betas_paramscount = 3 + self.max_sh_degree * 6
            features = torch.zeros((pcd_color.shape[0], spherical_betas_paramscount)).float().cuda()
            features[:, :3] = pcd_color
            features[:, 3:] = 0.0
        else:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        
        if MCMC_init:
            scales = torch.log(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)
        else:
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) 
            
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        
        if MCMC_init:
            opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        if self.use_SBs:
            self._features_dc = nn.Parameter(features[:,0:3].contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,3:spherical_betas_paramscount].contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.init_tracking_state(self.get_xyz.shape[0])

    def training_setup(self, training_args, mesh_args, appearance_net):
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
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

        ]
        if appearance_net is not None:
            if isinstance(appearance_net, AppearanceEmbedding):
                l += [
                    {'params': [appearance_net._appearance_embeddings], 'lr': mesh_args.appearance_lr_init, "name": "appearance embedding"},
                    {'params': appearance_net.appearance_network.parameters(), 'lr': mesh_args.appearance_lr_init, "name": "appearance net"}
                ]
            elif isinstance(appearance_net, PGSREmbedding):
                l += [
                    {'params': [appearance_net._As], 'lr': mesh_args.appearance_lr_init, "name": "appearance embedding"},
                    {'params': [appearance_net._Bs], 'lr': mesh_args.appearance_lr_init, "name": "appearance net"}
                ]


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.appearance_scheduler_args = get_expon_lr_func(
            lr_init=mesh_args.appearance_lr_init,
            lr_final=mesh_args.appearance_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        pos_lr = 0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                pos_lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = pos_lr
            if "appearance" in param_group["name"]:
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
        return pos_lr 
        
        

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        if self.use_SBs:
            for i in range(self._features_dc.shape[1]):
                l.append('f_dc_{}'.format(i)) 
            for i in range(self._features_rest.shape[1]):
                l.append('f_rest_{}'.format(i)) 
        else:
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('filter_3D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        if self.use_SBs:
            f_dc = self._features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        else:
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

    def decay_opacity(self, val=0.999):
        opacities_new = inverse_sigmoid(self.get_opacity * val)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity(self):
        # reset opacity by considering 3D filter
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
        opacities_new = self.inverse_opacity_activation(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    @torch.no_grad()
    def get_tetra_points(self, views: List[Camera], meshing_settings : MeshingSettings):
        M = trimesh.creation.box()
        M.vertices *= 2
        
        rots = build_rotation(self._rotation)
        xyz = self.get_xyz
        
        # tight opacity bounding, as in StopThePop (in comment)
        match meshing_settings.bounding:
            case BoundingSetting.SIGMA_3:
                scale = self.get_scaling_with_3D_filter * 3.
            case BoundingSetting.SIGMA_333:
                scale = self.get_scaling_with_3D_filter * 3.33
            case BoundingSetting.STP:
                scale = self.get_scaling_with_3D_filter * torch.sqrt(2. * torch.log(255. * self.get_opacity_with_3D_filter))
            #torch.sqrt(2 * torch.log(255 * self.get_opacity_with_3D_filter))
        # filter points with small opacity (as done for bicycle in GOF)
        if meshing_settings.opacity_cutoff_tetra > 0.:
            opacity = self.get_opacity_with_3D_filter
            mask = (opacity > meshing_settings.opacity_cutoff_tetra).squeeze(-1)
            xyz = xyz[mask]
            scale = scale[mask]
            rots = rots[mask]
        
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        
        # Mask out vertices outside of context views
        if meshing_settings.near_far_culling:
            vertex_mask = get_frustum_mask_batched(vertices, views, meshing_settings.near, meshing_settings.far)
            return vertices[vertex_mask], vertices_scale[vertex_mask]
        else:
            return vertices, vertices_scale
  

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        self.use_SBs = len(extra_f_names) in {12, 18, 24, 30}  

        filter_3D = None
        if "filter_3D" in plydata.elements[0]:
            filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

        if self.use_SBs:
            features_dc = np.zeros((xyz.shape[0], 3))        
            features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
        else:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])


        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        if self.use_SBs:
            features_extra = features_extra.reshape((features_extra.shape[0], len(extra_f_names)))
        else:
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
        if self.use_SBs:
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if filter_3D is not None:
            self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")
        else:
            warnings.warn("3D Filter was not loaded (wasn't in ply file), and should be precomputed with training cameras")
        
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0],), dtype=torch.float32, device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")
        self.init_tracking_state(self._xyz.shape[0])

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "appearance" in group["name"]:
                continue
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
            if "appearance" in group["name"]:
                continue
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
        self._generation = self._generation[valid_points_mask]
        self._edge_touched = self._edge_touched[valid_points_mask]
        self._edge_touch_iter = self._edge_touch_iter[valid_points_mask]
        if isinstance(self.filter_3D, torch.Tensor) and self.filter_3D.ndim > 0 and self.filter_3D.shape[0] == valid_points_mask.shape[0]:
            self.filter_3D = self.filter_3D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "appearance" in group["name"]:
                continue
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True, tracking_state: Optional[Dict[str, torch.Tensor]] = None):
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
        self._append_tracking_extension(self._build_tracking_extension(new_xyz.shape[0]) if tracking_state is None else tracking_state)

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2):
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

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        if self.use_SBs:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1)
        else:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)

        
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        selected_indices = selected_pts_mask.nonzero(as_tuple=True)[0]
        tracking_state = self._inherit_tracking(selected_indices, repeats=N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, tracking_state=tracking_state)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

# The following code is based on Gaussian Opacity Fields (https://github.com/autonomousvision/gaussian-opacity-fields):
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/5245b20e5d11acd6d1ff5af4b890dc2bedd99693/scene/gaussian_model.py#L631
    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, clone_with_sampling=False):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        if clone_with_sampling:
            # sample a new gaussian instead of fixing position
            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        selected_indices = selected_pts_mask.nonzero(as_tuple=True)[0]
        tracking_state = self._inherit_tracking(selected_indices)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, tracking_state=tracking_state)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, abs_grad_for_densification=False, clone_with_sampling=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        # if this value is absurdly high (as it is, we effectively will not use absolute gradients)
        if not abs_grad_for_densification:
            Q = torch.tensor(1e4, device=grads_abs.device, dtype=grads_abs.dtype)
        if (Q == 0).item():
            assert(False)

        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent, clone_with_sampling)
        clone = self._xyz.shape[0]
        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]


        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # print(f"Prune {torch.sum(prune_mask)} points due to low opacity")
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # print(f"Prune {torch.sum(big_points_vs)} points due to big screen size")
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # print(f"Prune {torch.sum(big_points_ws)} points due to big scale")
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        
        prune = self._xyz.shape[0]        
        torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune
    

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1
        
# The following code is based on 3DGS-MCMC (https://github.com/ubc-vision/3dgs-mcmc):
# https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/scene/gaussian_model.py#L411
    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # handle params for the appearance embedding
            if 'appearance' in group['name']:
                continue

            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 
        torch.cuda.empty_cache()
        return optimizable_tensors
    
    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0].to(torch.int32) + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))
        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]
    
    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    
    def relocate_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return
        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        self._source_tag[dead_indices] = self._source_tag[reinit_idx]
        self._seed_id[dead_indices] = self._seed_id[reinit_idx]
        self._generation[dead_indices] = self._generation[reinit_idx] + 1
        self._edge_touched[dead_indices] = self._edge_touched[reinit_idx]
        self._edge_touch_iter[dead_indices] = self._edge_touch_iter[reinit_idx]
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]
        self.replace_tensors_to_optimizer(inds=reinit_idx) 
        
    def reclone_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return
        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        
        selected_pts_mask = torch.zeros((self._opacity.shape[0],)).bool().cuda()
        selected_pts_mask[reinit_idx] = True
        
        new_xyz = self._xyz[selected_pts_mask]

        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        selected_indices = selected_pts_mask.nonzero(as_tuple=True)[0]
        tracking_state = self._inherit_tracking(selected_indices)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, tracking_state=tracking_state)
        
    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = self.get_opacity.squeeze(-1) 
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)
        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling
        tracking_state = self._inherit_tracking(add_idx)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False, tracking_state=tracking_state)
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

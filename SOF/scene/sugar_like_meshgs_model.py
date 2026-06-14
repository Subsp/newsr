from pathlib import Path
from typing import Dict

import numpy as np
import torch
import trimesh
from torch import nn

from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import inverse_sigmoid
from utils.prior_fusion import BOUND_BARYCENTRIC_TEMPLATES
from utils.sh_utils import RGB2SH


def _normalize_tensor(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def _normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)


def _quaternion_from_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert row-major rotation matrices to GraphDeco quaternion order wxyz."""
    m = matrix
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
                    1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                    1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                    1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
                ],
                dim=-1,
            ),
            min=1e-8,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[:, 0] ** 2, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], dim=-1),
            torch.stack([m[:, 2, 1] - m[:, 1, 2], q_abs[:, 1] ** 2, m[:, 1, 0] + m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0]], dim=-1),
            torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] + m[:, 0, 1], q_abs[:, 2] ** 2, m[:, 2, 1] + m[:, 1, 2]], dim=-1),
            torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 2, 1] + m[:, 1, 2], q_abs[:, 3] ** 2], dim=-1),
        ],
        dim=1,
    )
    quat_candidates = quat_by_rijk / (2.0 * torch.clamp(q_abs[:, :, None], min=1e-8))
    out = quat_candidates[torch.arange(matrix.shape[0], device=matrix.device), torch.argmax(q_abs, dim=-1)]
    return _normalize_tensor(out)


def _load_mesh_np(mesh_path: str):
    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh at {mesh_path}, got {type(mesh)}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid mesh shape: vertices={vertices.shape}, faces={faces.shape}")
    return vertices, faces


def _build_carrier_layout_from_faces(
    mesh_path: str,
    face_ids: np.ndarray,
    carriers_per_face: int,
    thickness_scale: float,
    plane_scale_multiplier: float = 1.0,
    max_plane_scale: float = 0.0,
):
    if carriers_per_face not in BOUND_BARYCENTRIC_TEMPLATES:
        raise ValueError(f"Unsupported carriers_per_face={carriers_per_face}")
    vertices, faces = _load_mesh_np(mesh_path)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    active_faces = faces[face_ids]
    triangles = vertices[active_faces]
    bary = BOUND_BARYCENTRIC_TEMPLATES[carriers_per_face]

    edge_u = triangles[:, 1] - triangles[:, 0]
    edge_u = _normalize_np(edge_u)
    normals = _normalize_np(np.cross(edge_u, triangles[:, 2] - triangles[:, 0]))
    edge_v = _normalize_np(np.cross(normals, edge_u))
    face_centers = triangles.mean(axis=1, keepdims=True)
    offsets = triangles - face_centers
    scale_u = np.max(np.abs((offsets * edge_u[:, None, :]).sum(axis=-1)), axis=1) * float(plane_scale_multiplier)
    scale_v = np.max(np.abs((offsets * edge_v[:, None, :]).sum(axis=-1)), axis=1) * float(plane_scale_multiplier)
    if float(max_plane_scale) > 0.0:
        scale_u = np.minimum(scale_u, float(max_plane_scale))
        scale_v = np.minimum(scale_v, float(max_plane_scale))
    scale_n = np.minimum(scale_u, scale_v) * float(thickness_scale)

    return {
        "vertices": vertices,
        "faces": faces,
        "face_ids": np.repeat(face_ids, carriers_per_face).astype(np.int64, copy=False),
        "bary_coords": np.tile(bary[None], (face_ids.shape[0], 1, 1)).reshape(-1, 3).astype(np.float32, copy=False),
        "scale_u": np.repeat(scale_u.astype(np.float32), carriers_per_face),
        "scale_v": np.repeat(scale_v.astype(np.float32), carriers_per_face),
        "scale_n": np.repeat(scale_n.astype(np.float32), carriers_per_face),
    }


class SugarLikeMeshGaussianModel(GaussianModel):
    """SuGaR-like mesh-bound Gaussian model for SOF experiments.

    The Gaussian centers are not independent xyz parameters. They are evaluated
    from mesh vertices, fixed face ids, and fixed barycentric coordinates. This
    keeps meshGS on the selected surface while still allowing color, opacity,
    in-plane scale, in-plane rotation, and optionally mesh vertices to optimize.
    """

    def __init__(self, sh_degree: int, use_SBs: bool = False):
        super().__init__(sh_degree=sh_degree, use_SBs=use_SBs)
        self.is_sugar_like_meshgs = True
        self._surface_vertices = torch.empty((0, 3), device="cuda")
        self._surface_faces = torch.empty((0, 3), dtype=torch.long, device="cuda")
        self._bound_face_ids = torch.empty((0,), dtype=torch.long, device="cuda")
        self._surface_global_vertex_ids = torch.empty((0,), dtype=torch.long, device="cuda")
        self._surface_global_face_ids = torch.empty((0,), dtype=torch.long, device="cuda")
        self._bound_global_face_ids = torch.empty((0,), dtype=torch.long, device="cuda")
        self._bound_bary_coords = torch.empty((0, 3), device="cuda")
        self._surface_plane_scaling = torch.empty((0, 2), device="cuda")
        self._surface_log_thickness = torch.empty((0, 1), device="cuda")
        self._surface_inplane_rotation = torch.empty((0, 2), device="cuda")
        self._normal_consistency_pairs = torch.empty((0, 2), dtype=torch.long, device="cuda")

    @property
    def get_xyz(self):
        if self._bound_face_ids.numel() == 0:
            return self._xyz
        face_vertices = self._surface_faces[self._bound_face_ids]
        vertices = self._surface_vertices[face_vertices]
        return (vertices * self._bound_bary_coords[..., None]).sum(dim=1)

    def _bound_basis(self):
        face_vertices = self._surface_faces[self._bound_face_ids]
        vertices = self._surface_vertices[face_vertices]
        edge_u = _normalize_tensor(vertices[:, 1] - vertices[:, 0])
        normals = _normalize_tensor(torch.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], dim=-1))
        edge_v = _normalize_tensor(torch.cross(normals, edge_u, dim=-1))
        rot = _normalize_tensor(self._surface_inplane_rotation)
        tangent_u = rot[:, 0:1] * edge_u + rot[:, 1:2] * edge_v
        tangent_v = -rot[:, 1:2] * edge_u + rot[:, 0:1] * edge_v
        return tangent_u, tangent_v, normals

    @property
    def get_rotation(self):
        if self._bound_face_ids.numel() == 0:
            return self.rotation_activation(self._rotation)
        tangent_u, tangent_v, normals = self._bound_basis()
        matrix = torch.stack([tangent_u, tangent_v, normals], dim=2)
        return _quaternion_from_rotation_matrix(matrix)

    @property
    def get_scaling(self):
        if self._bound_face_ids.numel() == 0:
            return self.scaling_activation(self._scaling)
        plane_scales = self.scaling_activation(self._surface_plane_scaling)
        normal_scale = torch.exp(self._surface_log_thickness)
        return torch.cat([plane_scales, normal_scale], dim=-1)

    def get_view2gaussian(self, viewmatrix):
        # The inherited method reads _rotation directly. Materialize the current
        # mesh-derived rotation for compatibility when this optional path is used.
        old_rotation = self._rotation
        self._rotation = self.get_rotation
        try:
            return super().get_view2gaussian(viewmatrix)
        finally:
            self._rotation = old_rotation

    def materialize_gaussian_model(self) -> GaussianModel:
        gs = GaussianModel(self.max_sh_degree, use_SBs=self.use_SBs)
        xyz = self.get_xyz.detach()
        gs.active_sh_degree = self.active_sh_degree
        gs.spatial_lr_scale = self.spatial_lr_scale
        gs._xyz = nn.Parameter(xyz.clone(), requires_grad=False)
        gs._features_dc = nn.Parameter(self._features_dc.detach().clone(), requires_grad=False)
        gs._features_rest = nn.Parameter(self._features_rest.detach().clone(), requires_grad=False)
        gs._opacity = nn.Parameter(self._opacity.detach().clone(), requires_grad=False)
        gs._scaling = nn.Parameter(torch.log(torch.clamp(self.get_scaling.detach(), min=1e-8)), requires_grad=False)
        gs._rotation = nn.Parameter(self.get_rotation.detach().clone(), requires_grad=False)
        gs.filter_3D = self.filter_3D.detach().clone()
        gs.max_radii2D = torch.zeros((xyz.shape[0],), dtype=torch.float32, device="cuda")
        gs.init_tracking_state(xyz.shape[0], source_tag=int(GaussianSourceTag.PRIOR_INJECTED))
        return gs

    def save_classic_ply(self, path: str):
        self.materialize_gaussian_model().save_ply(path)

    def save_bound_metadata(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "surface_vertices": self._surface_vertices.detach().cpu(),
                "surface_faces": self._surface_faces.detach().cpu(),
                "bound_face_ids": self._bound_face_ids.detach().cpu(),
                "surface_global_vertex_ids": self._surface_global_vertex_ids.detach().cpu(),
                "surface_global_face_ids": self._surface_global_face_ids.detach().cpu(),
                "bound_global_face_ids": self._bound_global_face_ids.detach().cpu(),
                "bound_bary_coords": self._bound_bary_coords.detach().cpu(),
                "surface_plane_scaling": self._surface_plane_scaling.detach().cpu(),
                "surface_log_thickness": self._surface_log_thickness.detach().cpu(),
                "surface_inplane_rotation": self._surface_inplane_rotation.detach().cpu(),
                "normal_consistency_pairs": self._normal_consistency_pairs.detach().cpu(),
            },
            path,
        )

    def build_optimizer(
        self,
        feature_lr: float,
        opacity_lr: float,
        surface_vertex_lr: float = 0.0,
        plane_scale_lr: float = 0.0,
        inplane_rotation_lr: float = 0.0,
    ):
        groups = [
            {"params": [self._features_dc], "lr": float(feature_lr), "name": "f_dc"},
            {"params": [self._opacity], "lr": float(opacity_lr), "name": "opacity"},
        ]
        if self._surface_vertices.requires_grad and surface_vertex_lr > 0:
            groups.append({"params": [self._surface_vertices], "lr": float(surface_vertex_lr), "name": "surface_vertices"})
        if self._surface_plane_scaling.requires_grad and plane_scale_lr > 0:
            groups.append({"params": [self._surface_plane_scaling], "lr": float(plane_scale_lr), "name": "surface_plane_scaling"})
        if self._surface_inplane_rotation.requires_grad and inplane_rotation_lr > 0:
            groups.append({"params": [self._surface_inplane_rotation], "lr": float(inplane_rotation_lr), "name": "surface_inplane_rotation"})
        return torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def normal_consistency_loss(self) -> torch.Tensor:
        if self._normal_consistency_pairs.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device="cuda")
        pair_face_ids, inverse = torch.unique(self._normal_consistency_pairs.reshape(-1), return_inverse=True)
        faces = self._surface_faces[pair_face_ids]
        vertices = self._surface_vertices[faces]
        normals = _normalize_tensor(torch.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], dim=-1))
        pair_normals = normals[inverse].reshape(-1, 2, 3)
        return (1.0 - (pair_normals[:, 0] * pair_normals[:, 1]).sum(dim=-1).clamp(-1.0, 1.0)).mean()

    def _build_normal_consistency_pairs_np(self, face_ids: np.ndarray, faces: np.ndarray, max_pairs: int = 500_000) -> np.ndarray:
        face_id_set = set(np.asarray(face_ids, dtype=np.int64).tolist())
        edge_to_face = {}
        pairs = []
        for face_id in face_id_set:
            tri = faces[int(face_id)]
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for a, b in edges:
                key = (int(min(a, b)), int(max(a, b)))
                other = edge_to_face.get(key)
                if other is None:
                    edge_to_face[key] = int(face_id)
                elif other in face_id_set:
                    pairs.append((other, int(face_id)))
                    if len(pairs) >= int(max_pairs):
                        return np.asarray(pairs, dtype=np.int64)
        return np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)

    def initialize_from_arrays(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_ids: np.ndarray,
        bary_coords: np.ndarray,
        colors: np.ndarray,
        scale_u: np.ndarray,
        scale_v: np.ndarray,
        scale_n: np.ndarray,
        opacity: np.ndarray,
        learn_surface_vertices: bool = False,
        learn_plane_scales: bool = False,
        learn_inplane_rotation: bool = False,
        build_normal_pairs: bool = True,
        max_normal_pairs: int = 500_000,
    ):
        num_gaussians = int(face_ids.shape[0])
        global_bound_face_ids = face_ids.astype(np.int64, copy=False)
        unique_global_face_ids = np.unique(global_bound_face_ids)
        active_global_faces = faces[unique_global_face_ids]
        unique_global_vertex_ids, compact_vertex_inverse = np.unique(
            active_global_faces.reshape(-1),
            return_inverse=True,
        )
        compact_vertices = vertices[unique_global_vertex_ids]
        compact_faces = compact_vertex_inverse.reshape(-1, 3).astype(np.int64, copy=False)
        local_bound_face_ids = np.searchsorted(unique_global_face_ids, global_bound_face_ids).astype(np.int64, copy=False)

        colors = np.clip(colors.astype(np.float32, copy=False), 0.0, 1.0)
        scale_u = np.clip(scale_u.astype(np.float32, copy=False), 1e-8, None)
        scale_v = np.clip(scale_v.astype(np.float32, copy=False), 1e-8, None)
        scale_n = np.clip(scale_n.astype(np.float32, copy=False), 1e-8, None)
        opacity = np.clip(opacity.astype(np.float32, copy=False), 1e-5, 0.995)

        self._surface_vertices = nn.Parameter(
            torch.from_numpy(compact_vertices.astype(np.float32, copy=False)).cuda(),
            requires_grad=bool(learn_surface_vertices),
        )
        self._surface_faces = torch.from_numpy(compact_faces).long().cuda()
        self._bound_face_ids = torch.from_numpy(local_bound_face_ids).long().cuda()
        self._surface_global_vertex_ids = torch.from_numpy(unique_global_vertex_ids.astype(np.int64, copy=False)).long().cuda()
        self._surface_global_face_ids = torch.from_numpy(unique_global_face_ids.astype(np.int64, copy=False)).long().cuda()
        self._bound_global_face_ids = torch.from_numpy(global_bound_face_ids).long().cuda()
        self._bound_bary_coords = torch.from_numpy(bary_coords.astype(np.float32, copy=False)).cuda()
        self._surface_plane_scaling = nn.Parameter(
            torch.log(torch.from_numpy(np.stack([scale_u, scale_v], axis=1)).float().cuda()),
            requires_grad=bool(learn_plane_scales),
        )
        self._surface_log_thickness = torch.log(torch.from_numpy(scale_n[:, None]).float().cuda())
        self._surface_inplane_rotation = nn.Parameter(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32, device="cuda").repeat(num_gaussians, 1),
            requires_grad=bool(learn_inplane_rotation),
        )

        fused_color = RGB2SH(torch.from_numpy(colors).float().cuda())
        features = torch.zeros((num_gaussians, 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device="cuda")
        features[:, :3, 0] = fused_color
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous(), requires_grad=False)
        self._opacity = nn.Parameter(inverse_sigmoid(torch.from_numpy(opacity[:, None]).float().cuda()), requires_grad=True)
        self._xyz = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        self._scaling = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        self._rotation = torch.empty((0, 4), dtype=torch.float32, device="cuda")
        self.filter_3D = torch.zeros((num_gaussians, 1), dtype=torch.float32, device="cuda")
        self.max_radii2D = torch.zeros((num_gaussians,), dtype=torch.float32, device="cuda")
        self.active_sh_degree = 0
        self.init_tracking_state(num_gaussians, source_tag=int(GaussianSourceTag.PRIOR_INJECTED))

        if build_normal_pairs:
            local_face_ids = np.arange(unique_global_face_ids.shape[0], dtype=np.int64)
            pairs = self._build_normal_consistency_pairs_np(local_face_ids, compact_faces, max_pairs=max_normal_pairs)
            self._normal_consistency_pairs = torch.from_numpy(pairs).long().cuda()
        else:
            self._normal_consistency_pairs = torch.empty((0, 2), dtype=torch.long, device="cuda")

    def initialize_from_fused_carrier_payload(
        self,
        mesh_path: str,
        payload_path: str,
        min_confidence: float = 0.05,
        max_disagreement: float = 0.08,
        min_views: int = 2,
        max_count: int = 0,
        seed: int = 0,
        scale_multiplier: float = 1.0,
        thickness_multiplier: float = 0.5,
        init_opacity: float = 0.35,
        learn_surface_vertices: bool = False,
        learn_plane_scales: bool = False,
        learn_inplane_rotation: bool = False,
        build_normal_pairs: bool = True,
        max_normal_pairs: int = 500_000,
    ) -> Dict[str, float]:
        payload = np.load(payload_path)
        valid = payload["valid_mask"].astype(bool)
        valid &= payload["confidence"] >= float(min_confidence)
        valid &= payload["disagreement"] <= float(max_disagreement)
        valid &= payload["view_count"] >= int(min_views)
        indices = np.flatnonzero(valid)
        if max_count > 0 and indices.size > int(max_count):
            rng = np.random.default_rng(int(seed))
            indices = np.sort(rng.choice(indices, size=int(max_count), replace=False))
        if indices.size == 0:
            raise RuntimeError("No valid mesh-bound carriers after filtering.")

        vertices, faces = _load_mesh_np(mesh_path)
        colors = np.clip(payload["fused_rgb"][indices].astype(np.float32), 0.0, 1.0)
        confidence = np.clip(payload["confidence"][indices].astype(np.float32), 0.0, 1.0)
        disagreement = payload["disagreement"][indices].astype(np.float32)
        disagreement_gate = 1.0 - np.clip(disagreement / max(float(max_disagreement), 1e-8), 0.0, 1.0)
        opacity = np.clip(float(init_opacity) * (0.25 + 0.75 * confidence * disagreement_gate), 1e-4, 0.95)

        self.initialize_from_arrays(
            vertices=vertices,
            faces=faces,
            face_ids=payload["face_ids"][indices].astype(np.int64),
            bary_coords=payload["bary_coords"][indices].astype(np.float32),
            colors=colors,
            scale_u=payload["scale_u"][indices].astype(np.float32) * float(scale_multiplier),
            scale_v=payload["scale_v"][indices].astype(np.float32) * float(scale_multiplier),
            scale_n=payload["scale_n"][indices].astype(np.float32) * float(scale_multiplier) * float(thickness_multiplier),
            opacity=opacity,
            learn_surface_vertices=learn_surface_vertices,
            learn_plane_scales=learn_plane_scales,
            learn_inplane_rotation=learn_inplane_rotation,
            build_normal_pairs=build_normal_pairs,
            max_normal_pairs=max_normal_pairs,
        )
        return {
            "payload_count": int(payload["valid_mask"].shape[0]),
            "selected_count": int(indices.size),
            "selected_ratio": float(indices.size / max(payload["valid_mask"].shape[0], 1)),
            "normal_consistency_pairs": int(self._normal_consistency_pairs.shape[0]),
        }

    def initialize_from_mesh_faces(
        self,
        mesh_path: str,
        face_ids: np.ndarray,
        carriers_per_face: int = 1,
        init_rgb=(0.5, 0.5, 0.5),
        thickness_scale: float = 0.1,
        plane_scale_multiplier: float = 1.0,
        max_plane_scale: float = 0.0,
        init_opacity: float = 0.1,
        **kwargs,
    ) -> Dict[str, float]:
        layout = _build_carrier_layout_from_faces(
            mesh_path,
            face_ids,
            carriers_per_face,
            thickness_scale,
            plane_scale_multiplier=plane_scale_multiplier,
            max_plane_scale=max_plane_scale,
        )
        colors = np.tile(np.asarray(init_rgb, dtype=np.float32)[None], (layout["face_ids"].shape[0], 1))
        opacity = np.full((layout["face_ids"].shape[0],), float(init_opacity), dtype=np.float32)
        self.initialize_from_arrays(
            vertices=layout["vertices"],
            faces=layout["faces"],
            face_ids=layout["face_ids"],
            bary_coords=layout["bary_coords"],
            colors=colors,
            scale_u=layout["scale_u"],
            scale_v=layout["scale_v"],
            scale_n=layout["scale_n"],
            opacity=opacity,
            **kwargs,
        )
        return {
            "selected_face_count": int(np.unique(layout["face_ids"]).shape[0]),
            "selected_gaussian_count": int(layout["face_ids"].shape[0]),
            "normal_consistency_pairs": int(self._normal_consistency_pairs.shape[0]),
            "plane_scale_multiplier": float(plane_scale_multiplier),
            "max_plane_scale": float(max_plane_scale),
            "scale_u_mean": float(np.mean(layout["scale_u"])) if layout["scale_u"].size else 0.0,
            "scale_u_p90": float(np.percentile(layout["scale_u"], 90)) if layout["scale_u"].size else 0.0,
            "scale_u_max": float(np.max(layout["scale_u"])) if layout["scale_u"].size else 0.0,
            "scale_v_mean": float(np.mean(layout["scale_v"])) if layout["scale_v"].size else 0.0,
            "scale_v_p90": float(np.percentile(layout["scale_v"], 90)) if layout["scale_v"].size else 0.0,
            "scale_v_max": float(np.max(layout["scale_v"])) if layout["scale_v"].size else 0.0,
        }

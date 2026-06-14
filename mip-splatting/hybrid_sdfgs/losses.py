import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation


def linearized_signed_distance(xyz, sdf_values, sdf_grads):
    """Create a first-order surrogate SDF that backpropagates to xyz.

    sdf_values/sdf_grads are queried from a teacher at detached anchor points.
    """
    anchor = xyz.detach()
    return sdf_values + ((xyz - anchor) * sdf_grads).sum(dim=-1, keepdim=True)


def surface_distance_loss(linearized_sdf, epsilon=0.0):
    abs_sdf = linearized_sdf.abs()
    if epsilon > 0:
        abs_sdf = F.relu(abs_sdf - epsilon)
    return abs_sdf.mean()


def gaussian_normals_from_params(rotations_raw, scales, axis="min_scale"):
    rotations = build_rotation(rotations_raw)
    if axis == "x":
        idx = torch.zeros((rotations.shape[0],), dtype=torch.long, device=rotations.device)
    elif axis == "y":
        idx = torch.ones((rotations.shape[0],), dtype=torch.long, device=rotations.device)
    elif axis == "z":
        idx = torch.full((rotations.shape[0],), 2, dtype=torch.long, device=rotations.device)
    elif axis == "max_scale":
        idx = torch.argmax(scales, dim=1)
    else:
        idx = torch.argmin(scales, dim=1)

    batch_idx = torch.arange(rotations.shape[0], device=rotations.device)
    normals = rotations[batch_idx, :, idx]
    normals = F.normalize(normals, dim=-1)
    return normals


def normal_alignment_loss(rotations_raw, scales, sdf_grads, axis="min_scale"):
    if sdf_grads.shape[0] == 0:
        return torch.zeros((), device=sdf_grads.device)
    gauss_normals = gaussian_normals_from_params(rotations_raw, scales, axis=axis)
    sdf_normals = F.normalize(sdf_grads, dim=-1)
    dot_val = (gauss_normals * sdf_normals).sum(dim=-1).abs()
    return (1.0 - dot_val).mean()


def offsurface_opacity_loss(opacity, linearized_sdf, margin=0.05):
    if margin <= 0:
        return torch.zeros((), device=opacity.device)
    mask = linearized_sdf.abs() > margin
    if not mask.any():
        return torch.zeros((), device=opacity.device)
    return opacity[mask].mean()

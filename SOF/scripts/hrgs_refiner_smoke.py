from __future__ import annotations

import torch

from models.hrgs_refiner import HRGSRefiner
from utils.gs_action_aggregator import aggregate_gs_actions
from utils.surface_payload_lifter import lift_surface_payload


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, v, h, w = 1, 3, 64, 64
    num_gaussians = 128
    k = 4

    model = HRGSRefiner().to(device)
    sr_images = torch.rand(b, v, 3, h, w, device=device)
    lr_up_images = torch.rand(b, v, 3, h, w, device=device)
    gs_buffers = {
        "render_rgb": torch.rand(b, v, 3, h, w, device=device),
        "depth": torch.rand(b, v, 1, h, w, device=device) * 3.0 + 0.5,
        "normal": torch.randn(b, v, 3, h, w, device=device),
        "alpha": torch.rand(b, v, 1, h, w, device=device),
        "diagnostics": torch.rand(b, v, 3, h, w, device=device),
    }
    gs_buffers["normal"] = torch.nn.functional.normalize(gs_buffers["normal"], dim=2)

    intrinsics = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, v, 1, 1)
    intrinsics[..., 0, 0] = 128.0
    intrinsics[..., 1, 1] = 128.0
    intrinsics[..., 0, 2] = float(w) / 2.0
    intrinsics[..., 1, 2] = float(h) / 2.0
    cam_to_world = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    cam_to_world[:, 1, 0, 3] = 0.05
    cam_to_world[:, 2, 1, 3] = 0.05

    outputs = model(
        sr_images=sr_images,
        lr_up_images=lr_up_images,
        gs_buffers=gs_buffers,
        cameras={"intrinsics": intrinsics, "cam_to_world": cam_to_world},
    )

    payload = lift_surface_payload(
        depth_surf=outputs["surface_2d"]["depth_surf"],
        normal_surf=outputs["surface_2d"]["normal_surf"],
        conf_geo=outputs["surface_2d"]["conf_geo"],
        mask_surface=outputs["surface_2d"]["mask_surface"],
        sr_images=sr_images,
        cameras={"intrinsics": intrinsics, "cam_to_world": cam_to_world},
    )

    visibility_records = {
        "gaussian_ids": torch.randint(-1, num_gaussians, (b, v, h, w, k), device=device),
        "weights": torch.rand(b, v, h, w, k, device=device),
    }
    action_payload = aggregate_gs_actions(
        masks_2d={
            "mask_update2d": outputs["surface_2d"]["mask_update2d"],
            "mask_surface": outputs["surface_2d"]["mask_surface"],
            "mask_detail": outputs["surface_2d"]["mask_detail"],
            "prior_color_weight2d": outputs["update_2d"]["prior_color_weight2d"],
        },
        action_features_2d=outputs["update_2d"]["action_features2d"],
        visibility_records=visibility_records,
        num_gaussians=num_gaussians,
    )

    print("surface depth:", tuple(outputs["surface_2d"]["depth_surf"].shape))
    print("carrier centers:", tuple(payload["centers"].shape))
    print("gs update strength:", tuple(action_payload["update_strength"].shape))


if __name__ == "__main__":
    main()

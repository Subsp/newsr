import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from scene.dataset_readers import sceneLoadTypeCallbacks
STATUS_COLORS = {
    "success": (40, 205, 90, 180),
    "weak": (255, 196, 64, 190),
    "dead": (235, 60, 60, 210),
}

STATUS_ORDER = {
    "success": 0,
    "weak": 1,
    "dead": 2,
}


def parse_args():
    parser = ArgumentParser(description="Project extension-probe outcome faces into camera images.")
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--view_set", choices=["train", "test"], default="test")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--probe_records", type=str, required=True, help="probe_outcomes.json or manifest-like json with status fields.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_views", type=int, default=12)
    parser.add_argument("--min_visible", type=int, default=8)
    parser.add_argument("--point_radius", type=int, default=6)
    parser.add_argument("--depth_min", type=float, default=0.2)
    parser.add_argument("--status_score_success", type=float, default=1.0)
    parser.add_argument("--status_score_weak", type=float, default=1.5)
    parser.add_argument("--status_score_dead", type=float, default=3.0)
    return parser.parse_args()


def load_camera_infos(scene_path: str, images: str, eval_mode: bool, view_set: str):
    if os.path.exists(os.path.join(scene_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](scene_path, images, eval_mode, init_type="sfm")
    elif os.path.exists(os.path.join(scene_path, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](scene_path, False, eval_mode)
    else:
        raise ValueError(f"Could not recognize scene type at {scene_path}")
    return scene_info.test_cameras if view_set == "test" else scene_info.train_cameras


def aggregate_records(records: List[Dict], mesh: trimesh.Trimesh) -> List[Dict]:
    by_face: Dict[int, Dict] = {}
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)

    for item in records:
        if "status" not in item:
            continue
        face_id = int(item["target_face_id"])
        status = str(item["status"])
        if face_id < 0 or face_id >= faces.shape[0]:
            continue
        tri = vertices[faces[face_id]]
        center = tri.mean(axis=0)

        existing = by_face.get(face_id)
        if existing is None:
            by_face[face_id] = {
                "target_face_id": face_id,
                "status": status,
                "center": center.tolist(),
                "count": 1,
                "max_final_opacity": float(item.get("final_opacity", 0.0)),
            }
            continue

        existing["count"] += 1
        existing["max_final_opacity"] = max(existing["max_final_opacity"], float(item.get("final_opacity", 0.0)))
        if STATUS_ORDER.get(status, -1) > STATUS_ORDER.get(existing["status"], -1):
            existing["status"] = status

    return list(by_face.values())


def project_points(cam_info, points_xyz: np.ndarray, depth_min: float) -> Tuple[np.ndarray, np.ndarray]:
    xyz_cam = points_xyz @ cam_info.R + cam_info.T[None, :]
    z = xyz_cam[:, 2]
    valid = z > depth_min

    tan_fovx = np.tan(cam_info.FovX / 2.0)
    tan_fovy = np.tan(cam_info.FovY / 2.0)
    focal_x = cam_info.width / (2.0 * tan_fovx)
    focal_y = cam_info.height / (2.0 * tan_fovy)

    x = xyz_cam[:, 0] / np.clip(z, 1e-6, None) * focal_x + cam_info.width / 2.0
    y = xyz_cam[:, 1] / np.clip(z, 1e-6, None) * focal_y + cam_info.height / 2.0

    valid &= x >= 0
    valid &= x < cam_info.width
    valid &= y >= 0
    valid &= y < cam_info.height

    proj = np.stack([x, y, z], axis=1)
    return proj, valid


def draw_overlay(base_image: Image.Image, projected: np.ndarray, statuses: List[str], radius: int, title: str, counts: Dict[str, int]) -> Image.Image:
    base = base_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()

    order = np.argsort(-projected[:, 2])
    for idx in order:
        x, y, _ = projected[idx]
        status = statuses[idx]
        fill = STATUS_COLORS[status]
        outline = (255, 255, 255, 220)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=1)

    draw.rectangle((10, 10, 290, 94), fill=(0, 0, 0, 140))
    draw.text((18, 16), title, fill=(255, 255, 255, 255), font=font)
    y0 = 38
    for status in ("success", "weak", "dead"):
        color = STATUS_COLORS[status]
        draw.rectangle((18, y0, 30, y0 + 12), fill=color, outline=(255, 255, 255, 220))
        draw.text((38, y0 - 1), f"{status}: {counts.get(status, 0)}", fill=(255, 255, 255, 255), font=font)
        y0 += 18

    return Image.alpha_composite(base, overlay).convert("RGB")


def build_contact_sheet(image_paths: List[Path], output_path: Path, columns: int = 2):
    if not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    thumb_width = min(widths)
    thumb_height = min(heights)
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (18, 18, 18))

    for idx, img in enumerate(images):
        row = idx // columns
        col = idx % columns
        resized = img.resize((thumb_width, thumb_height))
        sheet.paste(resized, (col * thumb_width, row * thumb_height))
    sheet.save(output_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(Path(args.probe_records).read_text())
    mesh = trimesh.load_mesh(args.mesh_path, process=False)
    aggregated = aggregate_records(records, mesh)
    if not aggregated:
        raise RuntimeError("No probe records with status information were found.")

    points_xyz = np.asarray([item["center"] for item in aggregated], dtype=np.float32)
    statuses = [str(item["status"]) for item in aggregated]

    cameras = load_camera_infos(args.scene_path, args.images, args.eval, args.view_set)
    view_entries = []

    score_weights = {
        "success": float(args.status_score_success),
        "weak": float(args.status_score_weak),
        "dead": float(args.status_score_dead),
    }

    for cam in cameras:
        projected, valid = project_points(cam, points_xyz, depth_min=args.depth_min)
        if int(valid.sum()) < args.min_visible:
            continue
        visible_statuses = [statuses[idx] for idx in np.flatnonzero(valid)]
        counts = {
            "success": int(sum(status == "success" for status in visible_statuses)),
            "weak": int(sum(status == "weak" for status in visible_statuses)),
            "dead": int(sum(status == "dead" for status in visible_statuses)),
        }
        score = (
            score_weights["success"] * counts["success"]
            + score_weights["weak"] * counts["weak"]
            + score_weights["dead"] * counts["dead"]
        )
        view_entries.append(
            {
                "camera": cam,
                "projected": projected[valid],
                "statuses": visible_statuses,
                "counts": counts,
                "visible_count": int(valid.sum()),
                "score": float(score),
            }
        )

    view_entries.sort(key=lambda item: (item["score"], item["visible_count"]), reverse=True)
    selected = view_entries[: args.max_views]

    overlay_paths: List[Path] = []
    summary_views = []
    for idx, entry in enumerate(selected):
        cam = entry["camera"]
        image = Image.open(cam.image_path).convert("RGB")
        title = f"{idx:02d} {cam.image_name}  visible={entry['visible_count']}"
        overlay = draw_overlay(
            base_image=image,
            projected=entry["projected"],
            statuses=entry["statuses"],
            radius=args.point_radius,
            title=title,
            counts=entry["counts"],
        )
        out_path = output_dir / f"{idx:02d}_{cam.image_name}_overlay.png"
        overlay.save(out_path)
        overlay_paths.append(out_path)
        summary_views.append(
            {
                "rank": idx,
                "image_name": cam.image_name,
                "image_path": cam.image_path,
                "visible_count": entry["visible_count"],
                "score": entry["score"],
                "counts": entry["counts"],
                "overlay_path": str(out_path),
            }
        )

    build_contact_sheet(overlay_paths, output_dir / "contact_sheet.png")

    summary = {
        "scene_path": str(Path(args.scene_path).resolve()),
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "probe_records": str(Path(args.probe_records).resolve()),
        "view_set": args.view_set,
        "eval": bool(args.eval),
        "n_records": len(records),
        "n_faces_visualized": len(aggregated),
        "n_candidate_views": len(view_entries),
        "n_selected_views": len(selected),
        "views": summary_views,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()

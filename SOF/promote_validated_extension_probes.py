import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch

from arguments import (
    MeshingParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    SplattingSettings,
    get_combined_args,
)
from scene import Scene
from scene.appearance_network import AppearanceEmbedding, PGSREmbedding
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import safe_state


def build_appearance_embedding(mesh_args, num_views: int):
    if mesh_args.use_decoupled_appearance:
        return AppearanceEmbedding(num_views=num_views)
    if mesh_args.use_pgsr_appearance:
        return PGSREmbedding(num_views=num_views)
    return None


def resolve_start_checkpoint(model_path: str, start_checkpoint: Optional[str], iteration: int) -> str:
    if start_checkpoint:
        return start_checkpoint
    if iteration < 0:
        raise ValueError("iteration must be explicit when start_checkpoint is not provided.")
    return os.path.join(model_path, f"chkpnt{iteration}.pth")


def parse_label_list(value: str) -> Set[str]:
    labels = {chunk.strip() for chunk in str(value).split(",") if chunk.strip()}
    if not labels:
        raise ValueError("keep_labels cannot be empty")
    return labels


def load_json_records(path: str) -> List[Dict]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list at {path}")
    return payload


def build_validated_seed_ids(
    probe_records: List[Dict],
    consensus_records: List[Dict],
    keep_labels: Set[str],
) -> Dict[str, object]:
    target_to_label: Dict[int, str] = {}
    label_counts: Dict[str, int] = {}
    for item in consensus_records:
        target_face_id = int(item["target_face_id"])
        label = str(item["validation_status"])
        target_to_label[target_face_id] = label
        label_counts[label] = label_counts.get(label, 0) + 1

    kept_seed_ids: Set[int] = set()
    kept_target_face_ids: Set[int] = set()
    missing_targets = 0

    for item in probe_records:
        if "seed_id" not in item or "target_face_id" not in item:
            raise ValueError("probe_records entries must contain seed_id and target_face_id")
        seed_id = int(item["seed_id"])
        target_face_id = int(item["target_face_id"])
        label = target_to_label.get(target_face_id)
        if label is None:
            missing_targets += 1
            continue
        if label in keep_labels:
            kept_seed_ids.add(seed_id)
            kept_target_face_ids.add(target_face_id)

    return {
        "kept_seed_ids": kept_seed_ids,
        "kept_target_face_ids": kept_target_face_ids,
        "label_counts": label_counts,
        "missing_targets": missing_targets,
    }


def default_output_paths(dataset, checkpoint_iteration: int, output_checkpoint: Optional[str], output_summary: Optional[str], output_preview_dir: Optional[str]):
    root = Path(dataset.model_path)
    checkpoint_path = Path(output_checkpoint) if output_checkpoint else root / f"chkpnt{checkpoint_iteration}_validatedonly.pth"
    summary_path = Path(output_summary) if output_summary else checkpoint_path.with_suffix(".summary.json")
    preview_dir = Path(output_preview_dir) if output_preview_dir else checkpoint_path.with_suffix("")
    return checkpoint_path, summary_path, preview_dir


def main():
    parser = ArgumentParser(description="Keep only validated extension probes in a checkpoint.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    splatting = SplattingSettings(parser, render=True)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--probe_records", type=str, required=True, help="probe_outcomes.json or manifest-like json with seed_id and target_face_id")
    parser.add_argument("--consensus_records", type=str, required=True, help="per_target_consensus_all.json from GT-free multiview consensus")
    parser.add_argument("--keep_labels", type=str, default="validated", help="Comma-separated validation labels to keep, default: validated")
    parser.add_argument("--output_checkpoint", type=str, default=None)
    parser.add_argument("--output_summary", type=str, default=None)
    parser.add_argument("--output_preview_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for validated-only promotion.")
        args.data_device = "cpu"

    safe_state(args.quiet)
    keep_labels = parse_label_list(args.keep_labels)

    dataset = model.extract(args)
    opt_args = opt.extract(args)
    pipe_args = pipe.extract(args)
    mesh_args = mesh.extract(args)
    splatting.get_settings(args)

    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe_args.convert_SBs_python)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras()

    appearance_embedding = build_appearance_embedding(mesh_args, num_views=len(train_cameras))
    gaussians.training_setup(opt_args, mesh_args, appearance_embedding)

    loaded_iteration = scene.loaded_iter if scene.loaded_iter is not None else args.iteration
    checkpoint_path = resolve_start_checkpoint(dataset.model_path, args.start_checkpoint, loaded_iteration)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_params, checkpoint_iteration, appearance_state = torch.load(checkpoint_path)
    if appearance_embedding is not None and appearance_state[0] is not None:
        appearance_embedding.restore(*appearance_state)
    gaussians.restore(model_params, opt_args, mesh_args, appearance_embedding)

    probe_records = load_json_records(args.probe_records)
    consensus_records = load_json_records(args.consensus_records)
    promotion_payload = build_validated_seed_ids(probe_records, consensus_records, keep_labels)
    kept_seed_ids = promotion_payload["kept_seed_ids"]

    if not kept_seed_ids:
        raise RuntimeError(f"No extension probe seeds matched keep_labels={sorted(keep_labels)}")

    source_tag = gaussians._source_tag
    seed_id = gaussians._seed_id
    is_probe = source_tag == int(GaussianSourceTag.EXTENSION_PROBE)

    keep_seed_tensor = torch.tensor(sorted(kept_seed_ids), dtype=torch.int64, device=seed_id.device)
    keep_probe_mask = torch.isin(seed_id, keep_seed_tensor)
    prune_mask = is_probe & (~keep_probe_mask)

    total_before = int(source_tag.shape[0])
    probe_before = int(is_probe.sum().item())
    prune_probe_count = int(prune_mask.sum().item())

    gaussians.prune_points(prune_mask)

    source_tag_after = gaussians._source_tag
    is_probe_after = source_tag_after == int(GaussianSourceTag.EXTENSION_PROBE)

    total_after = int(source_tag_after.shape[0])
    probe_after = int(is_probe_after.sum().item())
    retained_seed_ids = set(int(x) for x in gaussians._seed_id[is_probe_after].detach().cpu().tolist())

    output_checkpoint, output_summary, output_preview_dir = default_output_paths(
        dataset=dataset,
        checkpoint_iteration=checkpoint_iteration,
        output_checkpoint=args.output_checkpoint,
        output_summary=args.output_summary,
        output_preview_dir=args.output_preview_dir,
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_preview_dir.mkdir(parents=True, exist_ok=True)

    torch.save((gaussians.capture(), checkpoint_iteration, appearance_state), str(output_checkpoint))
    gaussians.save_ply(str(output_preview_dir / "point_cloud.ply"))
    gaussians.save_tracking_metadata(str(output_preview_dir / "gaussian_tags.pt"))

    filtered_probe_records = []
    for item in probe_records:
        if int(item["seed_id"]) in retained_seed_ids:
            filtered_probe_records.append(item)
    filtered_probe_records_path = output_preview_dir / "validated_probe_records.json"
    filtered_probe_records_path.write_text(json.dumps(filtered_probe_records, indent=2), encoding="utf-8")

    summary = {
        "mode": "validated_only_promotion",
        "scene_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "input_checkpoint": str(Path(checkpoint_path).resolve()),
        "output_checkpoint": str(output_checkpoint.resolve()),
        "probe_records": str(Path(args.probe_records).resolve()),
        "consensus_records": str(Path(args.consensus_records).resolve()),
        "keep_labels": sorted(keep_labels),
        "checkpoint_iteration": int(checkpoint_iteration),
        "counts": {
            "total_gaussians_before": total_before,
            "total_gaussians_after": total_after,
            "probe_gaussians_before": probe_before,
            "probe_gaussians_after": probe_after,
            "probe_gaussians_pruned": prune_probe_count,
            "probe_records_total": len(probe_records),
            "consensus_records_total": len(consensus_records),
            "kept_seed_count_from_consensus": len(kept_seed_ids),
            "kept_target_face_count_from_consensus": len(promotion_payload["kept_target_face_ids"]),
            "missing_probe_targets_in_consensus": int(promotion_payload["missing_targets"]),
        },
        "consensus_label_counts": promotion_payload["label_counts"],
        "paths": {
            "preview_point_cloud": str((output_preview_dir / "point_cloud.ply").resolve()),
            "preview_tags": str((output_preview_dir / "gaussian_tags.pt").resolve()),
            "validated_probe_records": str(filtered_probe_records_path.resolve()),
        },
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Validated-only checkpoint saved to: {output_checkpoint}")
    print(f"Summary saved to: {output_summary}")


if __name__ == "__main__":
    main()

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.sof_mesh_patch_enhancer_v0 import SOFMeshPatchEnhancerMLP, stats_from_array


def load_dataset(path: str):
    payload = torch.load(path, map_location="cpu")
    required = ["features", "target_delta_local", "target_confidence", "target_train_weight"]
    for key in required:
        if key not in payload:
            raise KeyError(f"Dataset missing '{key}': {path}")
    return payload


def main():
    parser = ArgumentParser(description="Train a geometry-only SOF mesh patch enhancer v0.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--confidence_loss_weight", type=float, default=0.25)
    parser.add_argument("--delta_l2_weight", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[sof-mesh-patch-train] CUDA unavailable; falling back to CPU.")
        args.device = "cpu"
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    dataset = load_dataset(args.dataset_path)
    features = dataset["features"].float()
    target_delta = dataset["target_delta_local"].float()
    target_conf = dataset["target_confidence"].float()
    train_weight = dataset["target_train_weight"].float()
    if float(train_weight.sum().item()) <= 0.0:
        raise RuntimeError("No reliable/weak training samples. Relax dataset thresholds.")

    tensor_dataset = TensorDataset(features, target_delta, target_conf, train_weight)
    loader = DataLoader(
        tensor_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model = SOFMeshPatchEnhancerMLP(
        in_dim=int(features.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
    ).to(device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(range(1, int(args.steps) + 1), desc="train-sof-mesh-patch-v0", dynamic_ncols=True)
    loader_iter = iter(loader)
    ema = {"loss": 0.0, "delta": 0.0, "conf": 0.0}

    for step in progress:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        x, y_delta, y_conf, weight = [item.to(device=args.device) for item in batch]

        out = model(x)
        pred_delta = out["delta_local"]
        pred_conf_logit = out["confidence_logit"]
        delta_loss_per = F.smooth_l1_loss(pred_delta, y_delta, reduction="none").mean(dim=1)
        weight_sum = weight.sum().clamp_min(1e-6)
        delta_loss = (delta_loss_per * weight).sum() / weight_sum
        conf_loss_per = F.binary_cross_entropy_with_logits(pred_conf_logit, y_conf, reduction="none")
        conf_loss = (conf_loss_per * (0.25 + 0.75 * y_conf)).mean()
        loss = delta_loss + float(args.confidence_loss_weight) * conf_loss
        if float(args.delta_l2_weight) > 0.0:
            loss = loss + float(args.delta_l2_weight) * (pred_delta.square().sum(dim=1) * (1.0 - y_conf)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        values = {
            "loss": float(loss.detach().item()),
            "delta": float(delta_loss.detach().item()),
            "conf": float(conf_loss.detach().item()),
        }
        for key, value in values.items():
            ema[key] = 0.05 * value + 0.95 * ema[key] if step > 1 else value
        if step % 20 == 0:
            progress.set_postfix({key: f"{value:.6f}" for key, value in ema.items()})

        if int(args.save_every) > 0 and step % int(args.save_every) == 0:
            ckpt_path = output_dir / f"sof_mesh_patch_enhancer_step_{step:06d}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_dim": int(features.shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "dataset_path": str(Path(args.dataset_path).resolve()),
                    "step": int(step),
                },
                str(ckpt_path),
            )

    checkpoint_path = output_dir / f"sof_mesh_patch_enhancer_step_{int(args.steps):06d}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": int(features.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "step": int(args.steps),
            "feature_schema": dataset.get("feature_schema", []),
            "bbox_center": dataset.get("bbox_center"),
            "bbox_diag": dataset.get("bbox_diag"),
        },
        str(checkpoint_path),
    )

    with torch.no_grad():
        pred = []
        conf = []
        for start in range(0, features.shape[0], int(args.batch_size)):
            out = model(features[start : start + int(args.batch_size)].to(device=args.device))
            pred.append(out["delta_local"].detach().cpu())
            conf.append(torch.sigmoid(out["confidence_logit"]).detach().cpu())
        pred_delta = torch.cat(pred, dim=0).numpy()
        pred_conf = torch.cat(conf, dim=0).numpy()

    summary = {
        "mode": "train_sof_mesh_patch_enhancer_v0",
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "parameters": vars(args),
        "counts": {
            "samples": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "weighted_samples": int((train_weight > 0).sum().item()),
        },
        "stats": {
            "target_delta_norm": stats_from_array(np.linalg.norm(target_delta.numpy(), axis=1)),
            "pred_delta_norm": stats_from_array(np.linalg.norm(pred_delta, axis=1)),
            "target_confidence": stats_from_array(target_conf.numpy()),
            "pred_confidence": stats_from_array(pred_conf),
            "train_weight": stats_from_array(train_weight.numpy()),
        },
    }
    summary_path = output_dir / "train_sof_mesh_patch_enhancer_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[sof-mesh-patch-train] saved checkpoint: {checkpoint_path}")
    print(f"[sof-mesh-patch-train] saved summary: {summary_path}")


if __name__ == "__main__":
    main()

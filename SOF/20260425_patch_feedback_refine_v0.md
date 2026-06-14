# Patch Feedback Refine v0

`refine_gaussians_with_patch_feedback_v0.py` is a standalone experiment that turns one saved dropout snapshot into a small patch-level geometry feedback loop for the original SOF GS field.

It does four things:

1. Load one dropout snapshot and aggregate `bias/std/coverage` from visible GS to mesh patches.
2. Select a small set of candidate patches with enough support GS.
3. For each candidate patch, try `+delta * n_p` and `-delta * n_p` as local geometry targets for the patch support GS.
4. Keep the direction only if local render loss does not get worse too much and the patch dropout metric improves.

The current v0 keeps the implementation deliberately conservative:

- It only updates `xyz`.
- It only optimizes support GS assigned to the candidate patch.
- It evaluates on the single camera stored in the input dropout snapshot.
- It uses a circular local mask around the candidate patch projection.

## Required inputs

- base SOF model directory / checkpoint
- `candidate_payload` with at least:
  - `nearest_face_id`
  - `nearest_surface_normal`
  - preferably `nearest_surface_point`
- `mesh_patch_bank_v0.npz`
- one `snapshot.pt` from `training_diagnostics_v0`

## Example

```bash
cd /path/to/SOF

python refine_gaussians_with_patch_feedback_v0.py \
  -s /path/to/scene \
  -m /path/to/base_model \
  --iteration 30000 \
  --mesh_path /path/to/mesh.obj \
  --candidate_payload /path/to/mesh_outside_candidates_v0.pt \
  --patch_bank_path /path/to/mesh_patch_bank_v0.npz \
  --dropout_snapshot /path/to/training_diagnostics_v0/iter_030600_xxx/snapshot.pt \
  --max_candidate_patches 8 \
  --max_patch_updates 3 \
  --local_steps 20 \
  --dropout_tile_size 16 \
  --dropout_num_masks 8
```

## Outputs

- model/checkpoint under `output_model_path`
- `patch_feedback_v0_summary.json`
- per-patch preview bundles under `patch_feedback_previews_v0/`

Each patch preview bundle contains:

- local mask
- base / plus / minus renders
- one contact sheet
- patch-level summary json

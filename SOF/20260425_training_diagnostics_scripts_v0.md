# Training Diagnostics Scripts v0

Added scripts for quick validation of the two new training-time diagnostics:

- `gradient tracking`
- `2D dropout -> gradient consistency`

## Scripts

Main entry:

- `scripts/run_training_diagnostics_scene.sh`

Convenience wrappers:

- `scripts/run_training_diagnostics_gradient_scene.sh`
- `scripts/run_training_diagnostics_dropout_scene.sh`
- `scripts/run_training_diagnostics_both_scene.sh`

## Default Behavior

The scripts assume:

- a scene alias can be created from `images_8 -> images_2`
- the baseline checkpoint is `early4k_soft` at `30000`
- a short validation finetune runs for `600` more steps

Default outputs:

- model: `output/<scene>_training_diagnostics_v0/<run_name>/model`
- diagnostics: `<model>/training_diagnostics_v0`

## Example

Run both lines together:

```bash
cd /Users/ltl/Desktop/codex_playground/SOF
PYTHON_BIN=/opt/miniconda3/bin/python \
SCENE_NAME=kitchen \
SCENE_ROOT=/path/to/scene \
TARGET_IMAGES_SUBDIR=images_2 \
SOURCE_IMAGES_SUBDIR=images_8 \
scripts/run_training_diagnostics_both_scene.sh
```

Run only gradient tracking:

```bash
cd /Users/ltl/Desktop/codex_playground/SOF
PYTHON_BIN=/opt/miniconda3/bin/python \
SCENE_NAME=kitchen \
SCENE_ROOT=/path/to/scene \
scripts/run_training_diagnostics_gradient_scene.sh
```

## Useful Overrides

- `BASELINE_MODEL_DIR`
- `BASELINE_CKPT`
- `BASE_ITER`
- `DIAG_RUN_ITERS`
- `FINAL_ITER`
- `DIAGNOSTIC_FROM_ITER`
- `RUN_NAME`
- `DIAGNOSTIC_BASIS_MODE`
- `DIAGNOSTIC_SURFACE_PAYLOAD`
- `RUN_RENDER_AFTER`
- `RUN_AGGREGATE_AFTER`
- `TRAIN_BASELINE_IF_MISSING`
- `DRY_RUN`

## Dry Run

Print commands without executing:

```bash
DRY_RUN=1 CHECK_INPUTS=0 scripts/run_training_diagnostics_both_scene.sh
```

# Experiment Scripts

This folder is centered on the current mainline pipeline, with a small number of helper launchers kept for warm-up and ablations:

1. setup
2. FlashVSR prior generation
3. GS-bootstrap + SDF densify training
4. evaluation

Most unrelated legacy scripts were removed, and the mainline path remains continuous from `00`.

## Fixed paths in scripts

- HBSR repo: `/root/autodl-tmp/HBSR`
- dataset: `/root/autodl-tmp/kitchen`
- LR train images: `images_4`
- GT eval images: `images`

## Script list (current)

- `00_install_experiment.sh`: base environment install for hybrid_sdfgs.
- `01_install_uav_env_cu118.sh`: legacy UAV environment script (kept for backward compatibility, not used by FlashVSR route).
- `02_run_baseline_gs_lego.sh`: vanilla GS baseline run.
- `03_generate_video_priors_flashvsr_lego.sh`: generate FlashVSR priors (`--input_downsample 4` by default), cache kept under `${OUTPUT_ROOT}/cache`.
- `04_run_hybrid_gsbootstrap_sdfdensify_4x_lego.sh`: current main experiment.
- `05_eval_model.sh`: evaluate output checkpoint/folder.
- `13_run_vanilla_2dgs_lr_images8.sh`: vanilla 2DGS warm-up on the `kitchen` scene before hybrid/SR changes.
- `14_generate_swinir_priors_and_masks_x8to2.sh`: generate SwinIR priors plus IE-SRGS-style `usable_masks` and `fused_priors`.
- `15_run_vanilla_2dgs_lr_images8_with_swinir_prior.sh`: one-shot vanilla 2DGS run that consumes SwinIR priors and masks.
- `16_install_sugar_system_python.sh`: create a SuGaR `venv` with system `python3` and install training dependencies via `pip`.
- `17_run_sugar_kitchen_images8.sh`: run SuGaR on `kitchen/images_8` by wrapping it into the `images/` layout that upstream SuGaR expects.
- `19_run_ie_srgs_style_x8to2_scene.sh`: IE-SRGS-style scene launcher using top-level `mip-splatting/train.py` as baseline plus `hybrid_sdfgs/train.py` with `usable_masks` and `fused_priors`.

## Recommended run order

```bash
cd /root/autodl-tmp/HBSR/hybrid_sdfgs/exp_scripts
bash 00_install_experiment.sh
bash 03_generate_video_priors_flashvsr_lego.sh
bash 04_run_hybrid_gsbootstrap_sdfdensify_4x_lego.sh
bash 05_eval_model.sh /root/autodl-tmp/HBSR/outputs/hybrid_gsbootstrap_sdfdensify_4x_kitchen
```

## Vanilla 2DGS Warm-Up

Before changing mesh extraction or adding SR priors, first make sure the vanilla 2DGS baseline can run on the existing `kitchen` scene.

Expected scene layout:

```text
/root/autodl-tmp/kitchen
├── images_8
└── sparse/0
```

Default launcher:

```bash
cd /root/autodl-tmp/HBSR/hybrid_sdfgs/exp_scripts
CUDA_DEVICE=0 \
PYTHON_BIN=python3 \
VANILLA_2DGS_ROOT=/root/autodl-tmp/2d-gaussian-splatting \
SCENE_ROOT=/root/autodl-tmp/kitchen \
IMAGES_SUBDIR=images_8 \
OUTPUT_PATH=/root/autodl-tmp/HBSR/outputs/vanilla_2dgs_lr_kitchen_images_8 \
bash 13_run_vanilla_2dgs_lr_images8.sh
```

Notes:

- `PYTHON_BIN` now defaults to the system `python3`, and falls back to `python` if needed. No conda path is required for this launcher.
- The launcher expects a COLMAP-style scene root with `sparse/0` and the selected image subdirectory.
- Keep `RUN_RENDER=1` if you want mesh export right after training; set `RUN_RENDER=0` for training only.

## SuGaR With System Python

SuGaR expects a COLMAP scene with `images/`, while the current warm-up scene uses `images_8`. The helper below creates a temporary scene alias with `images -> images_8`, so you do not need to rename the real dataset on the server.

### 1. Prepare the environment

```bash
cd /root/autodl-tmp/HBSR/hybrid_sdfgs/exp_scripts
BOOTSTRAP_SYSTEM_PYTHON=1 \
SUGAR_ROOT=/root/autodl-tmp/SuGaR \
SUGAR_ENV_DIR=/root/autodl-tmp/HBSR/.venvs/sugar-system-py \
bash 16_install_sugar_system_python.sh
```

Notes:

- The installer creates a `venv` from the system Python, not from conda.
- By default it also installs the small SwinIR runtime gap in the same `venv`: `timm`, `opencv-python-headless`, and `requests`.
- If the server has no distro Python yet, `BOOTSTRAP_SYSTEM_PYTHON=1` will first install `python3`, `python3-venv`, `python3-dev`, and `build-essential` with `apt-get`.
- Ubuntu 20.04's default system Python `3.8` is supported by this installer; Python `3.8`, `3.9`, and `3.10` are all accepted.
- `numpy` is pinned automatically by Python version: `3.8 -> 1.24.4`, `3.9/3.10 -> 1.26.4`.
- The script installs PyTorch CUDA `11.8`, PyTorch3D, Open3D, the Gaussian rasterizer submodules, and optionally `nvdiffrast`.

### 2. Run SuGaR on `kitchen/images_8`

```bash
cd /root/autodl-tmp/HBSR/hybrid_sdfgs/exp_scripts
CUDA_DEVICE=0 \
SUGAR_ROOT=/root/autodl-tmp/SuGaR \
SCENE_ROOT=/root/autodl-tmp/kitchen \
IMAGES_SUBDIR=images_8 \
PYTHON_BIN=/root/autodl-tmp/HBSR/.venvs/sugar-system-py/bin/python \
REGULARIZATION_TYPE=dn_consistency \
POLY_MODE=high \
REFINEMENT_TIME=short \
bash 17_run_sugar_kitchen_images8.sh
```

Optional knobs:

- Set `GS_OUTPUT_DIR=/path/to/vanilla_gs_output` to skip SuGaR's initial vanilla 3DGS warm-up.
- Set `POLY_MODE=low` for a faster low-poly run, or `POLY_MODE=custom` with `N_VERTICES_IN_MESH` and `GAUSSIANS_PER_TRIANGLE`.
- Set `EXPORT_OBJ=0` if you only want the hybrid representation and want to skip UV-textured mesh export.
- The helper creates a temporary alias scene under `${HBSR_ROOT}/outputs/sugar_scene_aliases/`, while upstream SuGaR still writes its checkpoints and meshes under `${SUGAR_ROOT}/output`.

## SwinIR Priors And IE-SRGS Masks

The SwinIR launcher follows the same system-Python policy as the SuGaR helpers:

- it prefers `${HBSR_ROOT}/.venvs/sugar-system-py/bin/python` if available
- otherwise it falls back to distro `python3` / `python`
- it intentionally avoids conda interpreters

### 1. Generate SR priors, usable masks, and fused priors

```bash
cd /root/autodl-tmp/HBSR/hybrid_sdfgs/exp_scripts
HBSR_ROOT=/root/autodl-tmp/HBSR \
SWINIR_ROOT=/root/autodl-tmp/SwinIR \
PYTHON_EXE=/root/autodl-tmp/HBSR/.venvs/sugar-system-py/bin/python \
SCENE_ROOT=/root/autodl-tmp/kitchen \
INPUT_SUBDIR=images_8 \
REFERENCE_DIR=/root/autodl-tmp/kitchen/images_2 \
OUTPUT_ROOT=/root/autodl-tmp/priors/kitchen_swinir_x8to2_classical \
bash 14_generate_swinir_priors_and_masks_x8to2.sh
```

To run HAT-L x4 instead, keep the same script and switch the backend plus checkpoint:

```bash
cd /root/autodl-tmp/HBSR/hybrid_sdfgs/exp_scripts
HBSR_ROOT=/root/autodl-tmp/HBSR \
SR_BACKEND=hat \
HAT_ROOT=/root/autodl-tmp/HAT \
MODEL_PATH=/root/autodl-tmp/HAT/experiments/pretrained_models/HAT-L_SRx4_ImageNet-pretrain.pth \
PYTHON_EXE=/root/autodl-tmp/HBSR/.venvs/sugar-system-py/bin/python \
SCENE_ROOT=/root/autodl-tmp/kitchen \
INPUT_SUBDIR=images_8 \
REFERENCE_DIR=/root/autodl-tmp/kitchen/images_2 \
OUTPUT_ROOT=/root/autodl-tmp/priors/kitchen_hat_x8to2_classical \
bash 14_generate_swinir_priors_and_masks_x8to2.sh
```

Outputs:

- raw priors: `${OUTPUT_ROOT}/priors`
- usable masks: `${OUTPUT_ROOT}/usable_masks`
- discrepancy maps: `${OUTPUT_ROOT}/discrepancy`
- aligned references: `${OUTPUT_ROOT}/aligned_references`
- masked priors: `${OUTPUT_ROOT}/masked_priors`
- masked references: `${OUTPUT_ROOT}/masked_references`
- fused priors: `${OUTPUT_ROOT}/fused_priors`

### 2. Run vanilla 2DGS with IE-SRGS-style defaults

`15_run_vanilla_2dgs_lr_images8_with_swinir_prior.sh` now defaults to:

- `PRIOR_SUBDIR=fused_priors`
- `PRIOR_MASK_SUBDIR=usable_masks`

So when `REFERENCE_DIR` is provided, it uses the fused prior plus the usable-mask gate by default.

```bash
cd /root/autodl-tmp/HBSR/hybrid_sdfgs/exp_scripts
CUDA_DEVICE=0 \
PYTHON_BIN=/root/autodl-tmp/HBSR/.venvs/sugar-system-py/bin/python \
VANILLA_2DGS_ROOT=/root/autodl-tmp/2d-gaussian-splatting \
SWINIR_ROOT=/root/autodl-tmp/SwinIR \
SCENE_ROOT=/root/autodl-tmp/kitchen \
IMAGES_SUBDIR=images_8 \
REFERENCE_DIR=/root/autodl-tmp/kitchen/images_2 \
PRIOR_ROOT=/root/autodl-tmp/priors/kitchen_swinir_x8to2_classical \
bash 15_run_vanilla_2dgs_lr_images8_with_swinir_prior.sh
```

## Main experiment notes

`04_run_hybrid_gsbootstrap_sdfdensify_4x_lego.sh` includes:

- explicit LR initialization via `-i images_4`
- `--sdf_mode gs_bootstrap` (GS->pseudo-SDF online adapter)
- `--sdf_densify_enable` + SR-guided densify scoring
- `--sdf_proposal_enable`:
  - 2D SR proposal extraction
  - ray backprojection + SDF hit
  - GS/SDF consistency gating
  - tangent-plane local sampling for densify anchors

`04_run_hybrid_gsbootstrap_sdfdensify_4x_kitchen_x8to2.sh` now also accepts optional prior-mask env vars:

- `PRIOR_MASK_SUBDIR=usable_masks` or a future mesh-projected structure-mask folder
- `PRIOR_MASK_FLOOR=0.0`

`05_eval_model.sh` evaluates on GT images by explicitly switching back to:

- `-i images`
- `render.py` -> `metrics.py`

## IE-SRGS-style with existing StableSR priors

This is the current runnable path when you already have StableSR priors under `<scene>/priors`.

Server-side environment:

```bash
cd /root/autodl-tmp/mip-splatting
conda create -n sdfgs-hybrid python=3.10 -y
conda activate sdfgs-hybrid
PYTHON_BIN=$(which python) \
bash hybrid_sdfgs/exp_scripts/00_install_experiment.sh
```

Assumed repos and assets:

- this repo at `/root/autodl-tmp/mip-splatting`
- scene root like `/root/autodl-tmp/kitchen`
- precomputed StableSR priors at `/root/autodl-tmp/kitchen/priors`

Example run:

```bash
cd /root/autodl-tmp/mip-splatting
conda activate sdfgs-hybrid

SCENE_ROOT=/root/autodl-tmp/kitchen \
SCENE_NAME=kitchen \
EXISTING_PRIOR_DIR=/root/autodl-tmp/kitchen/priors \
PYTHON_BIN=$(which python) \
bash hybrid_sdfgs/exp_scripts/19_run_ie_srgs_style_x8to2_scene.sh
```

The launcher now does:

1. baseline warm-up with top-level `mip-splatting/train.py`
2. `prepare_existing_sr_priors.py` to build `usable_masks` and `fused_priors`
3. prior-guided training with `hybrid_sdfgs/train.py`
4. GT-side eval on `images_2`
5. summary export via `compare_eval_results.py`

# hybrid_sdfgs

All hybrid SDF + Gaussian Splatting changes are isolated in this directory.

## Dataset convention

Input is always treated as Blender-style data.

Supported markers under `-s/--source_path`:

- `metadata.json` (mip multi-scale Blender format)
- `transforms_train.json` + `transforms_test.json` (NeRF synthetic)
- `transforms.json` (Instant-NGP / Neuralangelo style)

COLMAP auto-detection via `sparse/` is disabled.

## Entry point

Run from `mip-splatting` root:

```bash
python hybrid_sdfgs/train.py \
  -s <dataset_path> \
  -m <output_path> \
  --eval \
  --hybrid_enable \
  --sdf_mode analytic
```

Neuralangelo SDF teacher:

```bash
python hybrid_sdfgs/train.py \
  -s <dataset_path> \
  -m <output_path> \
  --eval \
  --hybrid_enable \
  --sdf_mode neuralangelo \
  --neuralangelo_root /path/to/neuralangelo \
  --sdf_config <path/to/config.yaml> \
  --sdf_checkpoint <path/to/checkpoint.pt>
```

Distilled pseudoSDF teacher from a 2DGS `point_cloud.ply`:

```bash
python hybrid_sdfgs/tools/distill_2dgs_surfel_pseudosdf.py \
  --point_cloud_ply <model_path>/point_cloud/iteration_30000/point_cloud.ply \
  --output_dir <distill_output_dir>

python hybrid_sdfgs/train.py \
  -s <dataset_path> \
  -m <output_path> \
  --eval \
  --hybrid_enable \
  --sdf_mode surfel_mlp \
  --sdf_checkpoint <distill_output_dir>/surfel_pseudosdf_mlp.pt
```

External prior-assisted training (video SR or other pre-generated priors):

```bash
python hybrid_sdfgs/train.py \
  -s <dataset_path> \
  -m <output_path> \
  --eval \
  --hybrid_enable \
  --sdf_mode analytic \
  --external_prior_root <prior_root> \
  --external_prior_subdir priors \
  --prior_l1_weight 0.02 \
  --prior_hf_weight 0.10
```

`<prior_root>/priors` should contain images named with camera image stems
(example: `r_23.png` for camera `r_23`).

## Experimental blocks

Optional blocks (all off by default):

- `--sdf_densify_enable`
- `--fm_sds_enable`
- `--freq_loss_enable` (`--freq_method fft|haar`)
- `--scaffold_enable`

Files:

- `blocks/sdf_densify_block.py`
- `blocks/fm_sds_block.py`
- `blocks/frequency_block.py`
- `geometry/scaffold_provider.py`
- `blocks/scaffold_geometry_block.py`

## Unified environment

`environment.unified.yml` and `requirements.unified.txt` support:

- `mip-splatting` (with mip filtering/post-process path)
- `neuralangelo`
- video-SR prior generation scripts

Recommended install:

```bash
cd /path/to/HBSR
conda create -n sdfgs-hybrid python=3.10 -y
conda activate sdfgs-hybrid
python -m pip install -U pip "setuptools<81" wheel packaging cmake ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
python -m pip install -r hybrid_sdfgs/requirements.unified.txt

python -m pip uninstall -y opencv-python opencv-python-headless || true
python -m pip install --no-cache-dir numpy==1.26.4 opencv-python-headless==4.10.0.84

python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation \
  git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Equivalent helper script after activating the env:

```bash
cd /path/to/mip-splatting
PYTHON_BIN=$(which python) \
bash hybrid_sdfgs/exp_scripts/00_install_experiment.sh
```

## IE-SRGS-style x8->x2 with existing StableSR priors

Current practical recipe for indoor Mip-NeRF 360 style scenes:

- baseline: top-level `mip-splatting/train.py`
- prior training: `hybrid_sdfgs/train.py`
- prior source: precomputed StableSR priors under `<scene>/priors`
- prior gating: `usable_masks + fused_priors`
- no depth constraint required

Why this route:

- the baseline stays inside the `mip-splatting` repo
- the actual prior-capable training entry in this repo is `hybrid_sdfgs/train.py`
- `prepare_existing_sr_priors.py` turns existing StableSR priors plus GT references into IE-SRGS-style `usable_masks` and `fused_priors`

Recommended indoor defaults are intentionally conservative:

- `prior_l1_weight=0.02`
- `prior_hf_weight=0.10`
- `prior_delta_clip=0.08`
- `prior_consistency_threshold=0.08`
- `prior_min_valid_ratio=0.80`

Example:

```bash
cd /path/to/mip-splatting
conda activate sdfgs-hybrid

SCENE_ROOT=/root/autodl-tmp/kitchen \
SCENE_NAME=kitchen \
EXISTING_PRIOR_DIR=/root/autodl-tmp/kitchen/priors \
PYTHON_BIN=$(which python) \
bash hybrid_sdfgs/exp_scripts/19_run_ie_srgs_style_x8to2_scene.sh
```

Outputs:

- baseline model: `outputs/ie_srgs_repro/<scene>/baseline_vanilla_2dgs_lr_images_8`
- prior model: `outputs/ie_srgs_repro/<scene>/stablesr_ie_srgs_style_images_8_to_images_2`
- compare json: `outputs/ie_srgs_repro/<scene>/compare_stablesr_ie_srgs_style.json`

## Files

- `train.py`: standalone hybrid training entry.
- `scene.py`: `HybridScene` with extra `transforms.json` support.
- `camera_bridge.py`: transforms.json camera/point-cloud bridge.
- `sdf_adapter.py`: SDF teacher adapters.
- `losses.py`: hybrid regularization losses.
- `scheduler.py`: stage-wise hybrid loss scaling.
- `blocks/`: independent experimental blocks.
- `tools/generate_video_sr_priors.py`: decoupled video prior generation.
  - supports `realbasic`, `rvrt`, `vrt`, `uav (Upscale-A-Video)`, `flashvsr`.
  - `flashvsr` dispatches to `tools/generate_flashvsr_priors.py` (prior-only + cache-oriented).
- `code_mem.txt`: project memory and parameter anchors.
- `environment.unified.yml`: shared conda env spec.
- `requirements.unified.txt`: shared pip dependency list.

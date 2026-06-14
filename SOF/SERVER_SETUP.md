# Server Setup

This fork is intended to run `SOF` remotely on a Linux + NVIDIA CUDA server.

## Clone

```bash
git clone --recursive git@github.com:Subsp/SOFSR.git
cd SOFSR
```

Recommended server layout:

```text
/root/autodl-tmp/
  SOFSR/
  kitchen/
  experiments/sof/kitchen_pseudogt/
    alias/
    model/
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Fast Server Environment

Do not use `conda env create -f environment.yml` on slow or fragile servers unless you really need the full solver-driven setup.

Use conda only to create an isolated Python, then install the rest with `pip`.

```bash
conda create -n sof python=3.10 -y
conda activate sof
```

Install system packages first:

```bash
apt-get update
apt-get install -y build-essential git cmake libgmp-dev libcgal-dev libgl1 libglib2.0-0
```

Upgrade packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

Install PyTorch first. This example uses CUDA 12.1 wheels:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the Python dependencies:

```bash
pip install -r requirements-server.txt
```

If your server can clone GitHub over `ssh` but times out on `https`, rewrite GitHub URLs once:

```bash
git config --global url."git@github.com:".insteadOf https://github.com/
```

Then install all CUDA extensions with the helper script:

```bash
bash ./scripts/install_server_extensions.sh
```

If your server is on CUDA 11.8 instead of 12.1, replace the PyTorch install line with:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```

Notes:

- `fused-ssim` is optional now. If that install fails, `train.py` will automatically fall back to the repository's slower Python SSIM implementation.
- `scripts/install_server_extensions.sh` auto-detects `CUDA_HOME`, exports it for `simple-knn` and `diff-gaussian-rasterization`, rewrites GitHub `https` URLs to `ssh` when enabled, and builds tetra triangulation with an explicit `CMAKE_CUDA_ARCHITECTURES`.
- If you need to override the tetra build architecture, for example on a machine where `native` is not desirable, use:

```bash
TETRA_CUDA_ARCHITECTURES=89 bash ./scripts/install_server_extensions.sh
```

## Sync

First time on the server:

```bash
git clone --recursive git@github.com:Subsp/SOFSR.git
cd SOFSR
```

Later updates:

```bash
cd SOFSR
git pull origin main
git submodule update --init --recursive
```

If upstream submodules changed and you want them refreshed aggressively:

```bash
cd SOFSR
git submodule sync --recursive
git submodule update --init --recursive --remote
```

## Kitchen Pseudo-GT Experiment

Assume your dataset lives at `/data/scenes/kitchen` and contains:

```text
/data/scenes/kitchen/
  images_8/
  images_2/
  sparse/0/
```

Prepare the bicubic pseudo scene from `images_8 -> images_2`:

```bash
cd /root/autodl-tmp/SOFSR
SCENE_ROOT=/data/scenes/kitchen \
EXPERIMENT_ROOT=/root/autodl-tmp/experiments/sof/kitchen_pseudogt \
PREPARE_ONLY=1 \
PYTHON_BIN=python \
bash ./scripts/run_kitchen_pseudogt_nvs.sh
```

Run training:

```bash
cd /root/autodl-tmp/SOFSR
python train.py \
  --splatting_config configs/hierarchical.json \
  -s /root/autodl-tmp/experiments/sof/kitchen_pseudogt/alias \
  --eval \
  -m /root/autodl-tmp/experiments/sof/kitchen_pseudogt/model \
  --iterations 30000
```

Render against the real `images_2` views:

```bash
cd /root/autodl-tmp/SOFSR
python render.py \
  -m /root/autodl-tmp/experiments/sof/kitchen_pseudogt/model \
  -s /data/scenes/kitchen \
  -i images_2 \
  --eval \
  --skip_train \
  --data_device cpu
```

Evaluate PSNR / SSIM / LPIPS / FLIP:

```bash
cd /root/autodl-tmp/SOFSR
python metrics.py -m /root/autodl-tmp/experiments/sof/kitchen_pseudogt/model
```

Results will be written to:

```text
/root/autodl-tmp/experiments/sof/kitchen_pseudogt/model/results_full.json
/root/autodl-tmp/experiments/sof/kitchen_pseudogt/model/per_view.json
```

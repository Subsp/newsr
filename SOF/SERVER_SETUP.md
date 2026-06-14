# Server Setup

This repo is intended to run the extracted `newsr` mainline on a Linux +
NVIDIA CUDA server.

## Sync

First clone:

```bash
cd /root/autodl-tmp
git clone git@github.com:Subsp/newsr.git
cd newsr
```

If the server only has HTTPS credentials:

```bash
cd /root/autodl-tmp
git clone https://github.com/Subsp/newsr.git
cd newsr
```

Later updates:

```bash
cd /root/autodl-tmp/newsr
git pull --ff-only origin main
```

Recommended server layout:

```text
/root/autodl-tmp/
  newsr/
  kitchen/
  external/
    NAFNet/
    Restormer/
```

The `kitchen` scene is expected to contain at least:

```text
/root/autodl-tmp/kitchen/
  images_8/
  images_2/
  sparse/0/
```

For x1 render-restoration priors, also provide:

```text
/root/autodl-tmp/kitchen/
  renders_lr_same_size/
```

## Base Environment

Use conda only to create an isolated Python, then install the rest with `pip`.

```bash
conda create -n newsr python=3.10 -y
conda activate newsr
```

Install system packages:

```bash
apt-get update
apt-get install -y build-essential git cmake libgmp-dev libcgal-dev libgl1 libglib2.0-0
```

Upgrade packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

Install PyTorch first. For CUDA 12.1:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8 instead:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```

Install Python dependencies:

```bash
cd /root/autodl-tmp/newsr
pip install -r SOF/requirements-server.txt
pip install -r mip-splatting/requirements.txt
pip install numpy==1.26.4 scipy pyyaml requests tensorboard imageio imageio-ffmpeg kornia trimesh pillow
```

Do not blindly install `mip-splatting/hybrid_sdfgs/requirements.unified.txt` on
this env because it pins an older CUDA/PyTorch stack.

## CUDA Extensions

Install the SOF-side CUDA extensions:

```bash
cd /root/autodl-tmp/newsr/SOF
bash scripts/install_server_extensions.sh
```

Install the mip-splatting-side CUDA extensions:

```bash
cd /root/autodl-tmp/newsr/mip-splatting
pip install --no-build-isolation submodules/simple-knn/
pip install --no-build-isolation submodules/diff-gaussian-rasterization/
pip install --no-build-isolation submodules/diff-gaussian-rasterization-sof-vanilla/
```

If CUDA is not auto-detected, set it explicitly before installing extensions:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
```

If GitHub HTTPS installs time out but SSH works:

```bash
git config --global url."git@github.com:".insteadOf https://github.com/
```

## External Restoration Backends

Clone restoration repos outside this repo so their weights and environments do
not get committed accidentally.

```bash
mkdir -p /root/autodl-tmp/external
cd /root/autodl-tmp/external
git clone https://github.com/megvii-research/NAFNet.git
git clone https://github.com/swz30/Restormer.git
```

Install their extra dependencies in the same `newsr` env first. If dependency
conflicts become annoying, create separate envs and pass
`EXTERNAL_RESTORATION_PYTHON=/path/to/python`.

```bash
conda activate newsr
cd /root/autodl-tmp/external/NAFNet
pip install -r requirements.txt || true
python setup.py develop --no_cuda_ext || true

cd /root/autodl-tmp/external/Restormer
pip install -r requirements.txt || true
python setup.py develop --no_cuda_ext || true
```

Download the NAFNet/Restormer pretrained checkpoints according to each upstream
repo and update their option files if needed.

## Smoke Checks

Check the prior generator CLI:

```bash
cd /root/autodl-tmp/newsr
python SOF/scripts/generate_enhancement_sr_priors.py --help
```

Check NAFNet x1 prior generation on a few render frames:

```bash
cd /root/autodl-tmp/newsr
python SOF/scripts/generate_enhancement_sr_priors.py \
  --input_dir /root/autodl-tmp/kitchen/renders_lr_same_size \
  --output_dir /root/autodl-tmp/kitchen/render_x1_priors_nafnet_smoke \
  --backend nafnet \
  --external_repo_root /root/autodl-tmp/external/NAFNet \
  --external_config options/test/REDS/NAFNet-width64.yml \
  --limit 2
```

Check Restormer x1 prior generation on a few render frames:

```bash
cd /root/autodl-tmp/newsr
python SOF/scripts/generate_enhancement_sr_priors.py \
  --input_dir /root/autodl-tmp/kitchen/renders_lr_same_size \
  --output_dir /root/autodl-tmp/kitchen/render_x1_priors_restormer_smoke \
  --backend restormer \
  --external_repo_root /root/autodl-tmp/external/Restormer \
  --restormer_task Single_Image_Defocus_Deblurring \
  --restormer_tile 720 \
  --limit 2
```

## Main Runs

Run the x1 NAFNet render-restoration prior branch:

```bash
cd /root/autodl-tmp/newsr/SOF
SOURCE_IMAGES_SUBDIR=renders_lr_same_size \
ENHANCEMENT_BACKEND=nafnet \
NAFNET_ROOT=/root/autodl-tmp/external/NAFNet \
EXTERNAL_RESTORATION_CONFIG=options/test/REDS/NAFNet-width64.yml \
bash scripts/run_mipsplatting_render_restoration_prior_scratch_v0_kitchen.sh
```

Run the x1 Restormer render-restoration prior branch:

```bash
cd /root/autodl-tmp/newsr/SOF
SOURCE_IMAGES_SUBDIR=renders_lr_same_size \
ENHANCEMENT_BACKEND=restormer \
RESTORMER_ROOT=/root/autodl-tmp/external/Restormer \
RESTORMER_TASK=Single_Image_Defocus_Deblurring \
RESTORMER_TILE=720 \
bash scripts/run_mipsplatting_render_restoration_prior_scratch_v0_kitchen.sh
```

Run the canonical NoSR cleanup mainline:

```bash
cd /root/autodl-tmp/newsr/SOF
bash scripts/run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh
```

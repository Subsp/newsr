# LR Train + HR Eval Scripts

Goal:

- Train with LR supervision (`--resolution 4`)
- Evaluate the same model on HR views (`--resolution 1`)

## Fixed path baseline

- HBSR repo: `/data3/liutl/HBSR`
- Dataset: `/data3/liutl/nerf_synthetic/lego`
- Video prior root (example): `/data3/liutl/HBSR/priors/lego_video_realbasic`

## Run

```bash
cd /data3/liutl/HBSR/hybrid_sdfgs/exp_scripts_lr_train_hr_eval

# LR train + external video prior
bash 10_train_lr4_hybrid_external_video_prior_lego.sh

# Optional LR train baseline (no external prior)
bash 11_train_lr4_hybrid_analytic_lego.sh

# Evaluate HR render quality from LR-trained checkpoint
bash 20_eval_hr_from_lr4_model.sh

# Optional LR-side eval
bash 21_eval_lr4_from_lr4_model.sh
```

## Default output convention

- `/data3/liutl/HBSR/outputs/hybrid_external_video_prior_lego_lr4_train`
- `/data3/liutl/HBSR/outputs/hybrid_analytic_lego_lr4_train`

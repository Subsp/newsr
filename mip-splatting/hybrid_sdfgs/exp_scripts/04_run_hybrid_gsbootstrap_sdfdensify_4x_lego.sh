#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/root/autodl-tmp/HBSR"
DATASET_PATH="/root/autodl-tmp/kitchen"
TRAIN_IMAGES="images_4"
OUTPUT_PATH="/root/autodl-tmp/HBSR/outputs/hybrid_gsbootstrap_sdfdensify_4x_kitchen"
EXTERNAL_PRIOR_ROOT="/root/autodl-tmp/priors/kitchen_video_flashvsr_3tile"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUTPUT_PATH}"
cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python hybrid_sdfgs/train.py \
  -s "${DATASET_PATH}" \
  -i "${TRAIN_IMAGES}" \
  -m "${OUTPUT_PATH}" \
  --eval \
  --white_background \
  --disable_gui \
  --iterations 30000 \
  --hybrid_enable \
  --sdf_mode gs_bootstrap \
  --hybrid_points_per_iter 4096 \
  --gs_bootstrap_update_interval 500 \
  --gs_bootstrap_adaptive_refresh_enable \
  --gs_bootstrap_refresh_interval_a 50 \
  --gs_bootstrap_refresh_interval_b 100 \
  --gs_bootstrap_refresh_interval_c 200 \
  --gs_bootstrap_max_anchors 24000 \
  --gs_bootstrap_k_neighbors 8 \
  --gs_bootstrap_opacity_min 0.01 \
  --gs_bootstrap_normal_axis min_scale \
  --gs_bootstrap_sigma_mode geom_mid_min \
  --gs_bootstrap_distance_floor 0.05 \
  --sdf_densify_enable \
  --sdf_densify_interval 400 \
  --sdf_densify_topk 512 \
  --sdf_densify_min_score 0.04 \
  --sdf_densify_surface_coef 1.0 \
  --sdf_densify_normal_coef 0.5 \
  --sdf_densify_offsurface_coef 0.5 \
  --sdf_densify_loss_weight 0.01 \
  --sdf_densify_sr_weight 0.05 \
  --sdf_densify_sr_levels 3 \
  --sdf_densify_sr_samples_per_point 2 \
  --sdf_densify_sr_jitter_scale 0.5 \
  --sdf_densify_sr_max_points 8192 \
  --sdf_densify_sr_score_coef 0.5 \
  --sdf_proposal_enable \
  --sdf_proposal_topk 256 \
  --sdf_proposal_min_hf 0.02 \
  --sdf_proposal_ray_near 0.05 \
  --sdf_proposal_ray_far 6.0 \
  --sdf_proposal_ray_samples 64 \
  --sdf_proposal_surface_thresh 0.03 \
  --sdf_proposal_gs_support_thresh 0.02 \
  --sdf_proposal_gs_support_floor 0.002 \
  --sdf_proposal_gs_support_quantile 0.85 \
  --sdf_proposal_align_thresh 0.06 \
  --sdf_proposal_align_sigma_thresh 5.0 \
  --sdf_proposal_teacher_p90_soft 0.02 \
  --sdf_proposal_teacher_p90_hard 0.04 \
  --sdf_proposal_teacher_soft_scale 0.5 \
  --sdf_proposal_gs_knn_k 8 \
  --sdf_proposal_plane_samples 4 \
  --sdf_proposal_tangent_scale 0.8 \
  --sdf_proposal_normal_scale 0.25 \
  --external_prior_root "${EXTERNAL_PRIOR_ROOT}" \
  --external_prior_subdir "priors" \
  --prior_l1_weight 0.02 \
  --prior_hf_weight 0.10 \
  --prior_delta_clip 0.08 \
  --prior_consistency_threshold 0.08 \
  --prior_min_valid_ratio 0.80 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000

echo "[hybrid-gsbootstrap-sdfdensify-4x] done: ${OUTPUT_PATH} (train_images=${TRAIN_IMAGES})"

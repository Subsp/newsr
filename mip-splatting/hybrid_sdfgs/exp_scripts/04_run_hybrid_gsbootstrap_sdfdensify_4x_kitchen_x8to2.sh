#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/root/autodl-tmp/HBSR"
DATASET_PATH="/root/autodl-tmp/kitchen"
TRAIN_IMAGES="images_8"
OUTPUT_PATH="/root/autodl-tmp/HBSR/outputs/hybrid_gsbootstrap_sdfdensify_4x_kitchen_x8to2"
EXTERNAL_PRIOR_ROOT="${PRIOR_ROOT:-/root/autodl-tmp/priors/kitchen_video_flashvsr_x8to2_officiallike_seqmat}"
EXTERNAL_PRIOR_SUBDIR="${PRIOR_SUBDIR:-priors}"
EXTERNAL_PRIOR_MASK_SUBDIR="${PRIOR_MASK_SUBDIR:-}"
PRIOR_MASK_FLOOR="${PRIOR_MASK_FLOOR:-0.0}"
USE_GT_AS_PRIOR="${USE_GT_AS_PRIOR:-0}"
FULL_STACK_ENABLE="${FULL_STACK_ENABLE:-0}"
DISABLE_TENSORBOARD="${DISABLE_TENSORBOARD:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${USE_GT_AS_PRIOR}" == "1" ]]; then
  EXTERNAL_PRIOR_ROOT="${DATASET_PATH}"
  EXTERNAL_PRIOR_SUBDIR="images_2"
  PRIOR_ROOT_FLAG="--external_prior_use_dataset_root"
else
  PRIOR_ROOT_FLAG=""
fi

if [[ "${FULL_STACK_ENABLE}" == "1" ]]; then
  FULL_STACK_FLAG="--sdf_proposal_full_stack_enable"
else
  FULL_STACK_FLAG=""
fi

if [[ "${DISABLE_TENSORBOARD}" == "1" ]]; then
  TENSORBOARD_FLAG="--disable_tensorboard"
else
  TENSORBOARD_FLAG=""
fi

if [[ -n "${EXTERNAL_PRIOR_MASK_SUBDIR}" ]]; then
  PRIOR_MASK_ARGS=(--external_prior_mask_subdir "${EXTERNAL_PRIOR_MASK_SUBDIR}")
else
  PRIOR_MASK_ARGS=()
fi

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
  ${TENSORBOARD_FLAG} \
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
  ${FULL_STACK_FLAG} \
  --sdf_proposal_fuse_max_views 4 \
  --sdf_proposal_fuse_min_views 2 \
  --sdf_proposal_view_min_hf 0.01 \
  --sdf_proposal_delta_steps 5 \
  --sdf_proposal_delta_scale 0.75 \
  --sdf_proposal_delta_floor 0.005 \
  --sdf_proposal_tangent_steps 3 \
  --sdf_proposal_tangent_search_scale 0.35 \
  --sdf_proposal_feature_hf_weight 1.0 \
  --sdf_proposal_feature_dir_weight 0.5 \
  --sdf_proposal_feature_desc_weight 0.35 \
  --sdf_proposal_sdf_cost_weight 0.25 \
  --sdf_proposal_delta_reg_weight 0.05 \
  --sdf_proposal_tangent_reg_weight 0.03 \
  --sdf_proposal_newton_steps 1 \
  --sdf_proposal_realign_enable \
  --sdf_proposal_realign_start_iter 1000 \
  --sdf_proposal_realign_interval 10 \
  --sdf_proposal_realign_ema 0.2 \
  --sdf_proposal_realign_rotation_ema 0.15 \
  --sdf_proposal_realign_scale_ema 0.15 \
  --sdf_proposal_realign_max_shift 0.05 \
  --sdf_proposal_realign_max_ratio 1.25 \
  --sdf_proposal_realign_min_depth 0.001 \
  --sdf_proposal_native_birth_start_iter 1500 \
  --sdf_proposal_native_birth_interval 200 \
  --sdf_proposal_native_birth_max_points 64 \
  --sdf_proposal_prior_hf_prior_weight 0.25 \
  --sdf_proposal_neighbor_guidance_weight 1.0 \
  --sdf_proposal_neighbor_visibility_weight 0.5 \
  --sdf_proposal_neighbor_distance_weight 0.1 \
  ${PRIOR_ROOT_FLAG} \
  --external_prior_root "${EXTERNAL_PRIOR_ROOT}" \
  --external_prior_subdir "${EXTERNAL_PRIOR_SUBDIR}" \
  "${PRIOR_MASK_ARGS[@]}" \
  --prior_l1_weight 0.02 \
  --prior_hf_weight 0.10 \
  --prior_delta_clip 0.08 \
  --prior_mask_floor "${PRIOR_MASK_FLOOR}" \
  --prior_consistency_threshold 0.08 \
  --prior_min_valid_ratio 0.80 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000

echo "[hybrid-gsbootstrap-sdfdensify-4x-x8to2] done: ${OUTPUT_PATH} (train_images=${TRAIN_IMAGES})"

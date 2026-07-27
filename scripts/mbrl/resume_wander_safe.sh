#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python3 scripts/mbrl/train.py \
  --headless \
  --task Flat-Unitree-Go2-train-v0 \
  --num_envs 64 \
  --seed 42 \
  --resume_checkpoint logs/mbrl/go2_walk_2026-07-22_20-14-48/checkpoints/model_221000.pt \
  --resume_replay logs/mbrl/go2_walk_2026-07-22_20-14-48/checkpoints/replay_latest.pt \
  --no-resume_load_optim \
  --resume_warmup_steps 1000 \
  --seed_with_model_policy \
  --train_steps 260000 \
  --buffer_capacity 300000 \
  --model_type latent \
  --latent_dim 256 \
  --num_q 5 \
  --horizon 5 \
  --batch_size 4096 \
  --updates_per_step 8 \
  --candidates 512 \
  --elites 64 \
  --planner mppi \
  --planner_iterations 6 \
  --planner_start_steps 5000 \
  --planner_min_length_fraction 0.75 \
  --planner_recovery_steps 1000 \
  --planner_recent_episodes 100 \
  --planner_temperature 0.5 \
  --num_pi_trajs 24 \
  --seed_policy_noise 0.02 \
  --wander \
  --wander_x_min 0.05 \
  --wander_x_max 0.8 \
  --wander_y_min -0.3 \
  --wander_y_max 0.3 \
  --wander_yaw_min -0.3 \
  --wander_yaw_max 0.3 \
  --wander_resample_min 3.0 \
  --wander_resample_max 5.0 \
  --save_interval 5000 \
  --max_checkpoints 5 \
  --save_replay \
  --eval_interval 10 \
  --wandb \
  --wandb_project ldm-quad-mbrl \
  --wandb_name tdmpc_wander_resume_221k_modelpi_warmup_safe_checkpoints

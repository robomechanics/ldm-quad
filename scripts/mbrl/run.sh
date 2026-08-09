#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

SEED="${SEED:-43}"
PROJECT="${WANDB_PROJECT:-ldm-quad-mbrl}"

latest_run_dir() {
  find logs/mbrl -maxdepth 1 -type d -name 'go2_walk_*' -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

common_args=(
  --headless
  --task Flat-Unitree-Go2-train-v0
  --num_envs 64
  --seed "$SEED"
  --buffer_capacity 1000000
  --model_type latent
  --latent_dim 256
  --num_q 5
  --horizon 8
  --batch_size 1024
  --updates_per_step 8
  --utd 0.25
  --candidates 512
  --elites 64
  --planner mppi
  --planner_iterations 6
  --discount 0.995
  --planner_start_steps 2000
  --planner_min_length_fraction 0.0
  --planner_recovery_steps 2000
  --planner_recent_episodes 200
  --planner_temperature 0.5
  --planner_use_continue_model
  --planner_velocity_objective_weight 0.0
  --num_pi_trajs 24
  --seed_steps 2000
  --seed_action_mode smooth_zero
  --seed_action_noise_std 0.08
  --seed_action_smoothing 0.92
  --seed_pretrain_updates 5000
  --seed_policy_noise 0.02
  --save_interval 5000
  --max_checkpoints 5
  --save_replay
  --save_best_metric stable_tracking
  --eval_interval 50
  --wandb
  --wandb_project "$PROJECT"
)

echo "[RUN] Stage 1/3: no-prior TD-MPC stand-and-survive curriculum"
python3 scripts/mbrl/train.py \
  "${common_args[@]}" \
  --train_steps 80000 \
  --command_x 0.0 \
  --command_y 0.0 \
  --command_yaw 0.0 \
  --wandb_name "tdmpc_noprior_full_s${SEED}_stage1_stand"

STAGE1_RUN="$(latest_run_dir)"
STAGE1_CKPT="${STAGE1_RUN}/checkpoints/model_final.pt"
if [[ ! -f "$STAGE1_CKPT" ]]; then
  echo "[ERROR] Stage 1 checkpoint not found: $STAGE1_CKPT" >&2
  exit 1
fi

echo "[RUN] Stage 2/3: resume no-prior TD-MPC, slow forward command x=0.20"
echo "[RUN] Resuming from: $STAGE1_CKPT"
python3 scripts/mbrl/train.py \
  "${common_args[@]}" \
  --resume_checkpoint "$STAGE1_CKPT" \
  --train_steps 160000 \
  --command_x 0.2 \
  --command_y 0.0 \
  --command_yaw 0.0 \
  --wandb_name "tdmpc_noprior_full_s${SEED}_stage2_x0p2"

STAGE2_RUN="$(latest_run_dir)"
STAGE2_CKPT="${STAGE2_RUN}/checkpoints/model_final.pt"
if [[ ! -f "$STAGE2_CKPT" ]]; then
  echo "[ERROR] Stage 2 checkpoint not found: $STAGE2_CKPT" >&2
  exit 1
fi

echo "[RUN] Stage 3/3: resume no-prior TD-MPC, target forward command x=0.40"
echo "[RUN] Resuming from: $STAGE2_CKPT"
python3 scripts/mbrl/train.py \
  "${common_args[@]}" \
  --resume_checkpoint "$STAGE2_CKPT" \
  --train_steps 260000 \
  --command_x 0.4 \
  --command_y 0.0 \
  --command_yaw 0.0 \
  --wandb_name "tdmpc_noprior_full_s${SEED}_stage3_x0p4"

STAGE3_RUN="$(latest_run_dir)"
echo "[RUN] Done. Final run: $STAGE3_RUN"
echo "[RUN] Play final:"
echo "python -u scripts/mbrl/play.py --checkpoint ${STAGE3_RUN}/checkpoints/model_final.pt --num_envs 1 --num_episodes 1 --max_steps 300 --command_x 0.4 --command_y 0.0 --command_yaw 0.0"
echo "[RUN] Play best stable-tracking checkpoint:"
echo "python -u scripts/mbrl/play.py --checkpoint ${STAGE3_RUN}/checkpoints/model_best.pt --num_envs 1 --num_episodes 1 --max_steps 300 --command_x 0.4 --command_y 0.0 --command_yaw 0.0"

#!/usr/bin/env bash
# FINAL consolidation of the tuned x=0.4 Go2 walker.
#
# The velocity-tracking reward is now the tuned best (baked into flat_env_cfg.py:
# track weight=8.0, std=0.11, alive=0.10), which reaches ~0.31 m/s at command_x=0.4
# (reward-tuning ceiling; see scripts/mbrl/TUNING_RESULTS.md). This script resumes the
# tuned checkpoint and runs a full-length x=0.4 stage at planner_iterations=6 (final
# quality, vs the sweep's speed-oriented 3) to consolidate a robust walker.
# It early-stops on a genuine stable plateau (conservative defaults: stable_tracking
# metric, 10k patience, 10k resume-grace, >=90% episode length) so it won't burn
# compute past convergence -- but only ever stops a stable, plateaued policy.
set -uo pipefail

cd "$(dirname "$0")/../.."

PY="${PY:-/home/rml2/anaconda3/envs/isaaclab/bin/python}"
SEED="${SEED:-43}"
PROJECT="${WANDB_PROJECT:-ldm-quad-mbrl}"
# Tuned x=0.4 walker (weight-8.0 sweep winner, vel ~0.309). Override with BASE_CKPT=...
BASE_CKPT="${BASE_CKPT:-logs/mbrl/go2_walk_2026-08-22_06-19-42/checkpoints/model_final.pt}"
TRAIN_STEPS="${TRAIN_STEPS:-260000}"   # +40k on top of the tuned checkpoint (step 220000)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[$(ts)] [ERROR] tuned base checkpoint not found: $BASE_CKPT" >&2
  exit 1
fi

echo "[$(ts)] [RUN] Final x=0.4 consolidation: resume $BASE_CKPT -> train_steps=$TRAIN_STEPS (planner_iterations 6, tuned reward from flat_env_cfg)"
"$PY" -u scripts/mbrl/train.py \
  --headless --task Flat-Unitree-Go2-train-v0 --num_envs 64 --seed "$SEED" \
  --buffer_capacity 1000000 --replay_device auto \
  --model_type latent --latent_dim 256 --num_q 5 --horizon 8 --batch_size 1024 \
  --updates_per_step 8 --utd 0.25 --candidates 512 --elites 64 \
  --planner mppi --planner_iterations 6 --discount 0.99 \
  --planner_start_steps 2000 --planner_min_length_fraction 0.0 --planner_recovery_steps 2000 \
  --planner_recent_episodes 200 --planner_temperature 0.5 \
  --planner_use_continue_model --planner_continue_threshold 0.5 \
  --planner_velocity_objective_weight 0.0 --num_pi_trajs 24 \
  --q_dropout 0.1 --entropy_coef 0.0003 \
  --seed_steps 2000 --seed_action_mode smooth_zero --seed_action_noise_std 0.08 --seed_action_smoothing 0.92 \
  --seed_pretrain_updates 5000 --seed_policy_noise 0.02 \
  --save_interval 5000 --max_checkpoints 5 --save_replay \
  --save_best_metric stable_tracking --eval_interval 50 \
  --wandb --wandb_project "$PROJECT" \
  --resume_checkpoint "$BASE_CKPT" \
  --early_stop --early_stop_metric stable_tracking --early_stop_patience 10000 \
  --early_stop_min_steps 10000 --early_stop_length_fraction 0.9 \
  --train_steps "$TRAIN_STEPS" --command_x 0.4 --command_y 0.0 --command_yaw 0.0 \
  --wandb_name "tdmpc_final_s${SEED}_x0p4_w8"
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "[$(ts)] [ERROR] final run exited non-zero (rc=$RC)." >&2
  exit $RC
fi

FINAL_RUN="$(find logs/mbrl -maxdepth 1 -type d -name 'go2_walk_*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
echo "[$(ts)] [RUN] Done. Final run: $FINAL_RUN"
echo "[RUN] Play:"
echo "$PY -u scripts/mbrl/play.py --checkpoint ${FINAL_RUN}/checkpoints/model_best.pt --num_envs 1 --num_episodes 1 --max_steps 300 --command_x 0.4 --command_y 0.0 --command_yaw 0.0"

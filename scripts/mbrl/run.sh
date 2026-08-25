#!/usr/bin/env bash
# KEPT CURRICULUM RECORD + RUNNER -- Go2 latent-TD-MPC2 walker.
#
# Records the curriculum that is being KEPT (the good, reusable policies) AND carries the
# full-parameter recipe to train/continue it. NOT for throwaway experiments/sweeps -- those
# live in scratchpad drivers. Each stage's best policy is preserved standalone in
# logs/mbrl/best_walker/ (see its README.md).
#
# Reward: the tuned best, baked into flat_env_cfg.py (track w=8.0, std=0.11, alive=0.10).
# Tuning story + hyperparameters: scripts/mbrl/TUNING_RESULTS.md.
#
# Curriculum lineage (each stage resumes the previous, raising command_x):
#   0.0 stand -> x=0.2 -> x=0.3 -> x=0.4
#
#   Stage  cmd_x  Kept checkpoint (logs/mbrl/best_walker/)
#   x=0.2  0.2    model_150000_x0p2_knowngood.pt , best_walker_x0p2_stabletrack.pt
#   x=0.3  0.3    best_walker_x0p3.pt
#   x=0.4  0.4    best_x0p4_reward_w8_v0p309.pt (~0.309) ,
#                 final_x0p4_walker_v0p31.pt (~0.33 m/s, CURRENT BEST)
#
# Usage:
#   ./run.sh                 # print the curriculum manifest
#   ./run.sh play  [stage]   # play a kept walker (x0p2|x0p3|x0p4; default x0p4)
#   ./run.sh train           # run the FULL training recipe (all params below) to
#                            # continue the curriculum (defaults to x=0.4 from current best)
#   Train overrides (env):   CMD_X= RESUME= TRAIN_STEPS= SEED= WANDB_NAME=
#     e.g. continue x=0.3:   CMD_X=0.3 RESUME=logs/mbrl/best_walker/best_walker_x0p2_stabletrack.pt \
#                            TRAIN_STEPS=200000 ./run.sh train
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-/home/rml2/anaconda3/envs/isaaclab/bin/python}"
PROJECT="${WANDB_PROJECT:-ldm-quad-mbrl}"
BW=logs/mbrl/best_walker
ACTION="${1:-manifest}"

case "$ACTION" in
  play)
    STAGE="${2:-x0p4}"
    case "$STAGE" in
      x0p2) CKPT=$BW/best_walker_x0p2_stabletrack.pt; CMD=0.2 ;;
      x0p3) CKPT=$BW/best_walker_x0p3.pt;             CMD=0.3 ;;
      x0p4) CKPT=$BW/final_x0p4_walker_v0p31.pt;      CMD=0.4 ;;
      *) echo "unknown stage: $STAGE (use x0p2|x0p3|x0p4)"; exit 1 ;;
    esac
    [[ -f "$CKPT" ]] || { echo "[run] ERROR: missing $CKPT"; exit 1; }
    echo "[run] play $STAGE: $CKPT (command_x=$CMD)"
    exec "$PY" -u scripts/mbrl/play.py --checkpoint "$CKPT" \
      --num_envs 1 --num_episodes 1 --max_steps 300 \
      --command_x "$CMD" --command_y 0.0 --command_yaw 0.0
    ;;

  train)
    # ---------- FULL curriculum training recipe (all parameters) ----------
    SEED="${SEED:-43}"
    CMD_X="${CMD_X:-0.4}"
    RESUME="${RESUME:-$BW/final_x0p4_walker_v0p31.pt}"   # current best x=0.4 walker
    TRAIN_STEPS="${TRAIN_STEPS:-260000}"                 # absolute target (base is ~231050)
    WANDB_NAME="${WANDB_NAME:-curriculum_x$(echo "$CMD_X" | tr . p)_s${SEED}}"
    [[ -f "$RESUME" ]] || { echo "[run] ERROR: resume checkpoint missing: $RESUME"; exit 1; }
    echo "[run] TRAIN curriculum: cmd_x=$CMD_X resume=$RESUME -> train_steps=$TRAIN_STEPS"
    echo "[run]   (planner_iterations 6 = final quality; tuned reward baked into flat_env_cfg)"
    exec "$PY" -u scripts/mbrl/train.py \
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
      --resume_checkpoint "$RESUME" \
      --train_steps "$TRAIN_STEPS" --command_x "$CMD_X" --command_y 0.0 --command_yaw 0.0 \
      --wandb_name "$WANDB_NAME"
    ;;

  manifest|*)
    cat <<'EOF'
KEPT CURRICULUM -- Go2 latent-TD-MPC2 walker
Reward: tuned best baked into flat_env_cfg.py (track w=8.0, std=0.11, alive=0.10)
Policies preserved standalone in logs/mbrl/best_walker/ (see its README.md).

  Stage  cmd_x  Checkpoint
  x=0.2  0.2    model_150000_x0p2_knowngood.pt / best_walker_x0p2_stabletrack.pt
  x=0.3  0.3    best_walker_x0p3.pt
  x=0.4  0.4    final_x0p4_walker_v0p31.pt   (~0.33 m/s, CURRENT BEST)

  ./run.sh play  [x0p2|x0p3|x0p4]   # play a kept walker
  ./run.sh train                    # full-param training recipe (continue x=0.4 from best)
                                     # overrides: CMD_X= RESUME= TRAIN_STEPS= SEED= WANDB_NAME=
EOF
    ;;
esac

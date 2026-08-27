#!/usr/bin/env bash
# KEPT CURRICULUM RECORD + RUNNER -- Go2 latent-TD-MPC2 walker.
#
# Records the curriculum that is being KEPT (the good, reusable policies) AND carries the
# full-parameter recipe to train/continue it. NOT for throwaway experiments/sweeps.
# Each stage's best policy is preserved standalone in logs/mbrl/best_walker/ (see README.md).
#
# Reward: tuned best, baked into flat_env_cfg.py (track w=8.0, std=0.11, alive=0.10).
# Action scale: baked to 0.40 in flat_env_cfg.py (action-scale sweep winner) -- clears the
# reward-only ~0.33 ceiling to ~0.39 m/s at command 0.4. This is the FROZEN benchmark scale.
# Tuning: scripts/mbrl/TUNING_RESULTS.md.
#
# Curriculum lineage (each stage resumes the previous):
#   0.0 stand -> x=0.2 -> x=0.3 -> x=0.4 (v0p31, scale 0.25) -> x=0.4 fast (v0p39, scale 0.40)
#   -> omni O1 (wander x[-0.2,0.5] y[+-0.15] yaw[+-0.3])  <- CURRENT RUN (Step 2 stage 1)
#
#   Stage   cmd_x  scale  Kept checkpoint (logs/mbrl/best_walker/)          ~vel
#   x0p2    0.2    0.25   best_walker_x0p2_stabletrack.pt / model_150000_*  --
#   x0p3    0.3    0.25   best_walker_x0p3.pt                                --
#   x0p4    0.4    0.25   final_x0p4_walker_v0p31.pt                         ~0.33
#   x0p4f   0.4    0.40   best_x0p4_ascale0p40_v0p39.pt                      ~0.39  (CURRENT BEST)
#   o1      wander 0.40   (training now; resumes x0p4f @ 241450 steps)       --
#
# Omni command curriculum (Step 2 of the benchmark plan): O1 above; planned
#   O2 x[+-0.5] y[+-0.3] yaw[+-0.6]; O3 (final frozen ranges) ~x[+-0.5..0.6] y[+-0.4] yaw[+-0.8].
# O1 also enables the TD-M(PC)^2 BC term (--tdmpc2_bc_coef 0.1; A/B: same speed, longer episodes).
#
# NOTE: x0.2/x0.3/x0.4 were trained at action scale 0.25; the env default is now 0.40, so
# run.sh play passes the right --action_scale per stage to replay each one faithfully.
#
# Usage:
#   ./run.sh                 # print the curriculum manifest
#   ./run.sh play  [stage]   # play a kept walker (x0p2|x0p3|x0p4|x0p4f; default x0p4f)
#   ./run.sh train           # full training recipe (all params) to continue the curriculum
#                            # (defaults to command 0.4, baked action scale 0.40, from current best)
#   Train overrides (env):   CMD_X= RESUME= TRAIN_STEPS= SEED= WANDB_NAME=
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-/home/rml2/anaconda3/envs/isaaclab/bin/python}"
PROJECT="${WANDB_PROJECT:-ldm-quad-mbrl}"
BW=logs/mbrl/best_walker
ACTION="${1:-manifest}"

case "$ACTION" in
  play)
    STAGE="${2:-x0p4f}"
    case "$STAGE" in
      x0p2)  CKPT=$BW/best_walker_x0p2_stabletrack.pt; CMD=0.2; SCALE=0.25 ;;
      x0p3)  CKPT=$BW/best_walker_x0p3.pt;             CMD=0.3; SCALE=0.25 ;;
      x0p4)  CKPT=$BW/final_x0p4_walker_v0p31.pt;      CMD=0.4; SCALE=0.25 ;;
      x0p4f) CKPT=$BW/best_x0p4_ascale0p40_v0p39.pt;   CMD=0.4; SCALE=0.40 ;;
      *) echo "unknown stage: $STAGE (use x0p2|x0p3|x0p4|x0p4f)"; exit 1 ;;
    esac
    [[ -f "$CKPT" ]] || { echo "[run] ERROR: missing $CKPT"; exit 1; }
    echo "[run] play $STAGE: $CKPT (command_x=$CMD, action_scale=$SCALE)"
    exec "$PY" -u scripts/mbrl/play.py --checkpoint "$CKPT" \
      --num_envs 1 --num_episodes 1 --max_steps 300 \
      --command_x "$CMD" --command_y 0.0 --command_yaw 0.0 \
      --action_scale "$SCALE"
    ;;

  train)
    # ---------- FULL curriculum training recipe (all parameters) ----------
    # CURRENT EXPERIMENT: omni O1 (Step 2 stage 1) -- wander commands, BC term on.
    # Set WANDER=0 CMD_X=<v> to fall back to the old fixed-forward-command recipe.
    SEED="${SEED:-43}"
    WANDER="${WANDER:-1}"
    CMD_X="${CMD_X:-0.4}"
    X_MIN="${X_MIN:--0.2}"; X_MAX="${X_MAX:-0.5}"       # O1 ranges
    Y_MIN="${Y_MIN:--0.15}"; Y_MAX="${Y_MAX:-0.15}"
    YAW_MIN="${YAW_MIN:--0.3}"; YAW_MAX="${YAW_MAX:-0.3}"
    BC_COEF="${BC_COEF:-0.1}"                            # TD-M(PC)^2 BC term (0 = vanilla)
    RESUME="${RESUME:-$BW/best_x0p4_ascale0p40_v0p39.pt}"   # curriculum best (241450 env steps, ~0.39 m/s)
    TRAIN_STEPS="${TRAIN_STEPS:-300000}"                    # absolute target (~58k new steps for O1)
    if [[ "$WANDER" == "1" ]]; then
      CMD_ARGS=(--wander --wander_x_min "$X_MIN" --wander_x_max "$X_MAX" \
                --wander_y_min "$Y_MIN" --wander_y_max "$Y_MAX" \
                --wander_yaw_min "$YAW_MIN" --wander_yaw_max "$YAW_MAX")
      WANDB_NAME="${WANDB_NAME:-omni_O1_s${SEED}}"
      echo "[run] TRAIN omni O1: x[$X_MIN,$X_MAX] y[$Y_MIN,$Y_MAX] yaw[$YAW_MIN,$YAW_MAX] bc=$BC_COEF resume=$RESUME -> $TRAIN_STEPS"
    else
      CMD_ARGS=(--command_x "$CMD_X" --command_y 0.0 --command_yaw 0.0)
      WANDB_NAME="${WANDB_NAME:-curriculum_x$(echo "$CMD_X" | tr . p)_s${SEED}}"
      echo "[run] TRAIN fixed: cmd_x=$CMD_X resume=$RESUME -> train_steps=$TRAIN_STEPS"
    fi
    [[ -f "$RESUME" ]] || { echo "[run] ERROR: resume checkpoint missing: $RESUME"; exit 1; }
    echo "[run]   (action scale 0.40 + tuned reward baked into flat_env_cfg; planner_iterations 6)"
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
      --tdmpc2_bc_coef "$BC_COEF" \
      --seed_steps 2000 --seed_action_mode smooth_zero --seed_action_noise_std 0.08 --seed_action_smoothing 0.92 \
      --seed_pretrain_updates 5000 --seed_policy_noise 0.02 \
      --save_interval 5000 --max_checkpoints 5 --save_replay \
      --save_best_metric stable_tracking --eval_interval 50 \
      --wandb --wandb_project "$PROJECT" \
      --resume_checkpoint "$RESUME" \
      --train_steps "$TRAIN_STEPS" "${CMD_ARGS[@]}" \
      --wandb_name "$WANDB_NAME"
    ;;

  manifest|*)
    cat <<'EOF'
KEPT CURRICULUM -- Go2 latent-TD-MPC2 walker
Reward: tuned best baked into flat_env_cfg.py (track w=8.0, std=0.11, alive=0.10)
Action scale: baked to 0.40 (sweep winner) -- the FROZEN benchmark scale.
Policies preserved standalone in logs/mbrl/best_walker/ (see its README.md).

  Stage   commands              scale  Checkpoint                        ~vel
  x0p2    x=0.2                 0.25   best_walker_x0p2_stabletrack.pt   --
  x0p3    x=0.3                 0.25   best_walker_x0p3.pt               --
  x0p4    x=0.4                 0.25   final_x0p4_walker_v0p31.pt        ~0.33
  x0p4f   x=0.4                 0.40   best_x0p4_ascale0p40_v0p39.pt     ~0.39  (BEST forward)
  o1      wander x/y/yaw (O1)   0.40   (TRAINING NOW, resumes x0p4f)     --

  ./run.sh play  [x0p2|x0p3|x0p4|x0p4f]   # play a kept walker (default x0p4f)
  ./run.sh train                          # omni O1 recipe (WANDER=1 default; WANDER=0 CMD_X= for fixed)
                                          # overrides: RESUME= TRAIN_STEPS= SEED= BC_COEF= X_MIN/MAX= Y_MIN/MAX= YAW_MIN/MAX=
EOF
    ;;
esac

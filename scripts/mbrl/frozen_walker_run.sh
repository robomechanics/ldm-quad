#!/usr/bin/env bash
# FROZEN WALKER -- the fixed Go2 locomotion baseline and its SIT context adapter.
#
# The frozen walker is the Stage L omnidirectional latent-TD-MPC2 policy + world model:
#   logs/mbrl/best_walker/stageL_omni_326k.pt   (125MB, not in git; train_state.env_steps=320300)
# Its training lineage lives in omni_run.sh. This file is how the walker is USED.
#
# Fixed evaluation conditions (every adaptation result is reported under these):
#   command      (x, y, yaw) = (0.4, 0.0, 0.0)
#   action scale 0.40, baked into flat_env_cfg.py (never pass --action_scale)
#   reward       baked into flat_env_cfg.py plus Stage L's yaw 4.0/0.5 and linear std 0.20
# The friction sweep, the adapt A/B and the PPO comparison (scripts/paper/) use the same
# checkpoint and command, so their numbers stay comparable with everything here.
#
# RULES
#   1. The walker's weights are never retrained. Locomotion, reward, curriculum and action
#      scale are frozen.
#   2. Adaptation runs through SIT (history-to-context system identification): a set
#      transformer encodes the last 48 (action, next_obs - obs) transitions into a 16-d
#      context on the unit ball. The context enters the encoder, dynamics, reward, Q and
#      policy through ADDITIVE, zero-initialised projections (*.context_weight). Continue
#      and physical heads read the latent only. An empty history maps to a fixed zero
#      context (null_context is a buffer, not a parameter).
#   3. Adapter training (`adapt`) updates ONLY history_encoder.* and the online
#      *.context_weight tensors (128,784 params). All 7,784,058 base params, their
#      target/detach copies and the actor-loss Q scale (q_scale, 9.98) stay bit-identical
#      to the checkpoint; train.py verifies this after the first update and logs
#      base_weight_drift (always 0.0) to metrics.csv.
#
# WHAT THE ADAPTED MODEL GUARANTEES
#   - Empty history (null context) -> exactly the frozen walker (bit-identical actions).
#     This is the null-context arm of every comparison.
#   - Projections zeroed -> exactly the frozen walker.
#   - With a history the policy and model are the frozen network plus a trained additive
#     term W*c; the base network itself never changes.
#
# DYNAMICS THE ADAPTER LEARNS TO IDENTIFY (mbrl/dynamics_rand.py)
#   Each env draws its own parameters at every episode reset; the true values are stored per
#   transition in the replay (dyn_params = [motor_gain, foot_friction], probe targets only,
#   never a model input) and their per-env mean/min/max go to metrics.csv.
#   motor gain     --dyn_motor_gain_range LO HI. Multiplies the joint-position action scale
#                  (0.40) INSIDE the action term, so the observation's last-action term, the
#                  replay and the history all hold the policy's raw command: the gain is a
#                  genuine actuator mismatch, not readable from the observation.
#   foot friction  --dyn_friction_range LO HI. Static = dynamic friction on all 27 robot
#                  collision shapes. The Go2 asset's shapes default to 1.0 (scene default
#                  material) and the ground is 1.0 with combine mode "multiply", so the
#                  effective foot-ground coefficient is exactly the drawn value.
#   Measured on the frozen walker (16 envs x 300 steps, command 0.4, value pinned):
#                   falls   vx      foot slip (m/s, feet on ground)
#     friction 0.25   26    0.24    0.78
#     friction 0.40    1    0.39    0.45
#     friction 0.80    0    0.38    0.34
#     gain 0.60        0    0.10    0.18   (contact fraction 0.62 vs 0.93)
#     nominal (1.0)    0    0.37    0.32
#   The friction cliff lies between 0.25 and 0.40 effective. The recipe draws friction over
#   [0.25, 0.80] so it straddles the cliff, and gain over [0.60, 1.00].
#   Note: --mismatch low_friction in play.py sets the ground AND the scene default material
#   (which the robot inherits) to mu, so its effective coefficient is mu^2: the friction
#   sweep's mu 0.4-0.5 cliff is this same 0.16-0.25 effective region.
#
# EVALUATION ARMS (play.py, history checkpoints; a context-free checkpoint ignores these)
#   --context_mode null       zero context every step == the frozen walker exactly
#   --context_mode rolling    the last --context_len transitions per env (default; K <= 48)
#   --context_mode frozen     rolling until --context_freeze_step, then each env holds that
#                             step's context; envs that reset afterwards get the null context
#   --context_mode truncated  rolling, but every window is cleared at --context_truncate_step:
#                             the oracle arm when it equals --dyn_switch_step
#   Within-episode dynamics switch (no reset; robot state and history carry over):
#   --dyn_switch_step S --dyn_switch_motor_gain G --dyn_switch_friction F
#   Before S the pinned --dyn_*_range values apply (nominal 1.0 when not given); envs that
#   reset after S keep the switched value. Diagnostics rows carry dyn_switch_active,
#   dyn_* parameters, context_norm_mean and context_drift_mean.
#
# CHECKPOINTS
#   A context-free checkpoint (the frozen walker) loads into a history model through a
#   key-validated graft (mbrl/checkpoint.py): the only tensors allowed to be missing are
#   *.context_weight and history_encoder.*; anything else raises. SIT checkpoints record
#   history_context_dim / history_len / sit_train_mode in their args, and play.py rebuilds
#   the history model from those automatically.
#
# COST
#   `adapt` runs at ~0.37 env-steps/s (64 envs, 16 updates/step, batch 1024, MPPI 512x6) --
#   the same rate as full training, because gradients still flow through the frozen network.
#   20k steps ~= 15.5 h. Launch it in its own systemd scope (see `adapt` below).
#
# Usage:
#   ./frozen_walker_run.sh                      # print this summary
#   ./frozen_walker_run.sh play                 # watch the frozen walker (1 env, 300 steps)
#   ./frozen_walker_run.sh eval [mismatch] [play.py args...]
#                                               # headless diagnostics eval, 16 envs x 500 steps
#   ./frozen_walker_run.sh adapt                # SIT adapter v1 (context in every head; the sit-adapt-v1 record)
#   ./frozen_walker_run.sh adapt_v2             # SIT adapter v2 (context in the dynamics only)
#   Overrides (env): CKPT= (eval/play another checkpoint, e.g. an adapted one)
#                    OUT= ENVS= STEPS= SEED= TRAIN_STEPS= CTX_DIM= HIST_LEN= WARMUP= WANDB_NAME=
#                    GAIN_LO= GAIN_HI= FRIC_LO= FRIC_HI= ADAPTER_LR= ADAPTER_WD=
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-/home/rml2/anaconda3/envs/isaaclab/bin/python}"
PROJECT="${WANDB_PROJECT:-ldm-quad-mbrl}"
FROZEN=logs/mbrl/best_walker/stageL_omni_326k.pt
CMD=(--command_x 0.4 --command_y 0.0 --command_yaw 0.0)
ACTION="${1:-manifest}"

case "$ACTION" in
  play)
    CKPT="${CKPT:-$FROZEN}"
    [[ -f "$CKPT" ]] || { echo "[frozen] ERROR: missing $CKPT"; exit 1; }
    echo "[frozen] play $CKPT at (0.4, 0, 0)"
    exec "$PY" -u scripts/mbrl/play.py --checkpoint "$CKPT" \
      --num_envs 1 --num_episodes 1 --max_steps 300 "${CMD[@]}"
    ;;

  eval)
    # One mismatch per call; any further arguments go straight to play.py. Examples:
    #   ./frozen_walker_run.sh eval                                   # nominal reference
    #   ./frozen_walker_run.sh eval low_friction --mismatch_friction 0.3
    #   ./frozen_walker_run.sh eval motor_weakness --mismatch_mode delayed \
    #       --mismatch_start_step 250 --mismatch_motor_scale 0.6        # mid-episode switch
    #   CKPT=<adapted.pt> ./frozen_walker_run.sh eval motor_weakness ... # same, adapted model
    #   ./frozen_walker_run.sh eval nominal --dyn_friction_range 0.3 0.3   # held-out pinned friction
    #   ./frozen_walker_run.sh eval nominal --dyn_motor_gain_range 0.7 0.7 # held-out pinned gain
    #   CKPT=<adapted.pt> ./frozen_walker_run.sh eval nominal \
    #       --dyn_switch_step 250 --dyn_switch_friction 0.3 --context_mode rolling    # switch, adapt
    #   (repeat with --context_mode null / frozen --context_freeze_step 250 /
    #    truncated --context_truncate_step 250 for the other arms; ENVS=16 STEPS=600)
    # Mismatches: nominal low_friction compliant rough slope mass motor_weakness push.
    # For SIT evaluation prefer the --dyn_* flags: they are the axes the adapter trained on,
    # and they keep the raw command in the observation (motor_weakness scales the action
    # before env.step, so the observation and the world model see the scaled action).
    # A history checkpoint keeps a rolling 48-step context per env, reset at episode end.
    CKPT="${CKPT:-$FROZEN}"
    MISMATCH="${2:-nominal}"; shift $(( $# >= 2 ? 2 : 1 ))
    ENVS="${ENVS:-16}"; STEPS="${STEPS:-500}"; SEED="${SEED:-0}"
    EXTRA="$(printf '%s' "$*" | tr -s ' ' '_' | tr -cd 'A-Za-z0-9._-' | sed 's/--//g')"
    TAG="${TAG:-$(basename "$CKPT" .pt)_${MISMATCH}${EXTRA:+_$EXTRA}_s${SEED}}"
    OUT="${OUT:-logs/mbrl/frozen_walker_eval}"
    [[ -f "$CKPT" ]] || { echo "[frozen] ERROR: missing $CKPT"; exit 1; }
    mkdir -p "$OUT"
    echo "[frozen] eval $CKPT mismatch=$MISMATCH envs=$ENVS steps=$STEPS seed=$SEED -> $OUT/$TAG"
    exec "$PY" -u scripts/mbrl/play.py --headless \
      --checkpoint "$CKPT" --seed "$SEED" \
      --num_envs "$ENVS" --num_episodes 1000 --max_steps "$STEPS" "${CMD[@]}" \
      --mismatch "$MISMATCH" \
      --diagnostics --diagnostics_interval 5 --diagnostics_dir "$OUT/$TAG" "$@"
    ;;

  adapt)
    # SIT adapter training on the frozen walker. Launch in its own systemd scope so the
    # VS Code cgroup's OOM sweep cannot take it down:
    #   systemd-run --user --unit=sit-adapt --collect --working-directory=$PWD \
    #     bash -c 'exec bash scripts/mbrl/frozen_walker_run.sh adapt >> LOG 2>&1'
    # Stop it by unit (systemctl --user stop sit-adapt), never with pkill on a flag substring.
    #
    # Fresh replay: the stray best_walker/replay_latest.pt belongs to another run, so
    # --no-auto_resume_replay is required. The buffer fills with the frozen walker's own
    # planner rollouts; WARMUP env steps are collected before the first update.
    # --train_steps is ABSOLUTE and the frozen walker sits at env_steps=320300.
    SEED="${SEED:-43}"
    CTX_DIM="${CTX_DIM:-16}"; HIST_LEN="${HIST_LEN:-48}"
    WARMUP="${WARMUP:-1000}"
    TRAIN_STEPS="${TRAIN_STEPS:-340300}"      # 20k adapter steps; checkpoint every 2500 (8 kept)
    GAIN_LO="${GAIN_LO:-0.6}"; GAIN_HI="${GAIN_HI:-1.0}"
    FRIC_LO="${FRIC_LO:-0.25}"; FRIC_HI="${FRIC_HI:-0.8}"
    ADAPTER_WD="${ADAPTER_WD:-0.0}"
    ADAPTER_LR_FLAG=(); [[ -n "${ADAPTER_LR:-}" ]] && ADAPTER_LR_FLAG=(--sit_adapter_lr "$ADAPTER_LR")
    WANDB_NAME="${WANDB_NAME:-sit_adapter_c${CTX_DIM}_k${HIST_LEN}_s${SEED}}"
    [[ -f "$FROZEN" ]] || { echo "[frozen] ERROR: missing $FROZEN"; exit 1; }
    echo "[frozen] SIT adapter: $FROZEN -> train_steps=$TRAIN_STEPS ctx=$CTX_DIM K=$HIST_LEN warmup=$WARMUP gain=[$GAIN_LO,$GAIN_HI] friction=[$FRIC_LO,$FRIC_HI]"
    exec "$PY" -u scripts/mbrl/train.py \
      --headless --task Flat-Unitree-Go2-train-v0 --num_envs 64 --seed "$SEED" \
      --buffer_capacity 1000000 --replay_device auto \
      --model_type latent --latent_dim 256 --num_q 5 --horizon 8 --batch_size 1024 \
      --utd 0.25 --candidates 512 --elites 64 \
      --planner mppi --planner_iterations 6 --discount 0.99 \
      --planner_start_steps 2000 --planner_min_length_fraction 0.0 --planner_recovery_steps 2000 \
      --planner_recent_episodes 200 --planner_temperature 0.5 \
      --planner_use_continue_model --planner_continue_threshold 0.5 \
      --planner_velocity_objective_weight 0.0 --num_pi_trajs 24 \
      --q_dropout 0.1 --entropy_coef 0.0003 --tdmpc2_bc_coef 0.1 \
      --reward_yaw_weight 4.0 --reward_yaw_std 0.5 --reward_track_std 0.2 \
      --eval_tracking_yaw_weight 0.5 \
      --history_context_dim "$CTX_DIM" --history_len "$HIST_LEN" --sit_train_mode adapter \
      --sit_adapter_weight_decay "$ADAPTER_WD" "${ADAPTER_LR_FLAG[@]}" \
      --dyn_motor_gain_range "$GAIN_LO" "$GAIN_HI" --dyn_friction_range "$FRIC_LO" "$FRIC_HI" \
      --resume_checkpoint "$FROZEN" --no-auto_resume_replay --resume_warmup_steps "$WARMUP" \
      --save_interval 2500 --max_checkpoints 50 --save_replay --eval_interval 50 \
      --wandb --wandb_project "$PROJECT" --wandb_name "$WANDB_NAME" \
      --train_steps "$TRAIN_STEPS" "${CMD[@]}"
    ;;

  adapt_v2)
    # SIT adapter v2 = payAttentionDrift's structure: the context conditions ONLY the dynamics MLP
    # d(z, a, c) (--context_components dynamics_only). The encoder, reward, Q and policy are the
    # frozen walker's own and have no context input at all. Trainable: history_encoder +
    # dynamics.0.context_weight (512 x 16); the policy has nothing to train. The history encoder
    # learns through the consistency loss and whatever reads the predicted latent (reward/Q/continue/
    # physical at t+1).
    #
    # Why v2 (sit-adapt-v1 Level 2): with the context in every head, prediction improved but the
    # planner's reward/Q estimates shifted (plan_best 15.4 vs 19.3) and the walker fell at friction
    # 1.0 where the frozen walker never does. Masking heads at eval time cannot undo that, because
    # they were co-adapted through z. v2 keeps the planner's objective exactly the frozen walker's.
    #
    # Differences from `adapt` (v1): context_components dynamics_only; friction U(0.25, 1.0), so the
    # nominal ground (1.0) is in-distribution; decoupled weight decay 0.013 on the context projection.
    # WEIGHT DECAY 0.013, derived from sit-adapt-v1 model_332500 (the checkpoint whose context/base
    # ratio was acceptable): dynamics.0.context_weight per-element RMS 0.588. Its measured drift is
    # |E[m/sqrt(v)]| = 0.0078 per update (322500 -> 332500, 160k updates at lr 3e-4): a weak, noisy
    # gradient signal, far below Adam's consistent-sign 1.0. With decoupled decay torch applies
    # w <- w * (1 - lr*wd) per step, so the equilibrium is |w*| = drift / wd (lr cancels):
    # wd = 0.0078 / 0.588 = 0.0133. Time constant 1/(lr*wd) = 250k updates = 15.7k env steps, so
    # after 20k steps the RMS is expected near 0.42 (v1 undecayed: about 0.75). The history encoder
    # is not decayed. The drift estimate comes from v1 (context everywhere); v2's gradient into the
    # projection differs, so treat 0.013 as a calibrated starting point and watch the ratio.
    SEED="${SEED:-43}"
    CTX_DIM="${CTX_DIM:-16}"; HIST_LEN="${HIST_LEN:-48}"
    WARMUP="${WARMUP:-1000}"
    TRAIN_STEPS="${TRAIN_STEPS:-340300}"      # 20k adapter steps; checkpoint every 2500 (8 kept)
    GAIN_LO="${GAIN_LO:-0.6}"; GAIN_HI="${GAIN_HI:-1.0}"
    FRIC_LO="${FRIC_LO:-0.25}"; FRIC_HI="${FRIC_HI:-1.0}"
    ADAPTER_WD="${ADAPTER_WD:-0.013}"
    ADAPTER_LR_FLAG=(); [[ -n "${ADAPTER_LR:-}" ]] && ADAPTER_LR_FLAG=(--sit_adapter_lr "$ADAPTER_LR")
    WANDB_NAME="${WANDB_NAME:-sit_adapter_v2_dynonly_c${CTX_DIM}_k${HIST_LEN}_s${SEED}}"
    [[ -f "$FROZEN" ]] || { echo "[frozen] ERROR: missing $FROZEN"; exit 1; }
    echo "[frozen] SIT adapter v2 (dynamics_only): $FROZEN -> train_steps=$TRAIN_STEPS ctx=$CTX_DIM K=$HIST_LEN warmup=$WARMUP gain=[$GAIN_LO,$GAIN_HI] friction=[$FRIC_LO,$FRIC_HI] wd=$ADAPTER_WD"
    exec "$PY" -u scripts/mbrl/train.py \
      --headless --task Flat-Unitree-Go2-train-v0 --num_envs 64 --seed "$SEED" \
      --buffer_capacity 1000000 --replay_device auto \
      --model_type latent --latent_dim 256 --num_q 5 --horizon 8 --batch_size 1024 \
      --utd 0.25 --candidates 512 --elites 64 \
      --planner mppi --planner_iterations 6 --discount 0.99 \
      --planner_start_steps 2000 --planner_min_length_fraction 0.0 --planner_recovery_steps 2000 \
      --planner_recent_episodes 200 --planner_temperature 0.5 \
      --planner_use_continue_model --planner_continue_threshold 0.5 \
      --planner_velocity_objective_weight 0.0 --num_pi_trajs 24 \
      --q_dropout 0.1 --entropy_coef 0.0003 --tdmpc2_bc_coef 0.1 \
      --reward_yaw_weight 4.0 --reward_yaw_std 0.5 --reward_track_std 0.2 \
      --eval_tracking_yaw_weight 0.5 \
      --history_context_dim "$CTX_DIM" --history_len "$HIST_LEN" --sit_train_mode adapter \
      --context_components dynamics_only \
      --sit_adapter_weight_decay "$ADAPTER_WD" "${ADAPTER_LR_FLAG[@]}" \
      --dyn_motor_gain_range "$GAIN_LO" "$GAIN_HI" --dyn_friction_range "$FRIC_LO" "$FRIC_HI" \
      --resume_checkpoint "$FROZEN" --no-auto_resume_replay --resume_warmup_steps "$WARMUP" \
      --save_interval 2500 --max_checkpoints 50 --save_replay --eval_interval 50 \
      --wandb --wandb_project "$PROJECT" --wandb_name "$WANDB_NAME" \
      --train_steps "$TRAIN_STEPS" "${CMD[@]}"
    ;;

  manifest|*)
    cat <<'EOF'
FROZEN WALKER -- Go2 latent-TD-MPC2, Stage L omnidirectional (never retrained)
  checkpoint  logs/mbrl/best_walker/stageL_omni_326k.pt   (env_steps 320300)
  command     (0.4, 0.0, 0.0)      action scale 0.40 (baked)
  adaptation  SIT context adapter: history_encoder + zero-init *.context_weight only;
              every base weight stays bit-identical (base_weight_drift = 0.0);
              empty history == the frozen walker exactly
  dynamics    per-env motor gain U(0.6,1.0) and foot friction U(0.25,0.8), redrawn per episode

  ./frozen_walker_run.sh play                         # watch it walk
  ./frozen_walker_run.sh eval [mismatch] [args...]    # headless diagnostics eval
  ./frozen_walker_run.sh adapt                        # SIT adapter v1: context in every head (record of sit-adapt-v1)
  ./frozen_walker_run.sh adapt_v2                     # SIT adapter v2: context in the dynamics only (~15 h)
  CKPT=<ckpt> ./frozen_walker_run.sh eval ...         # evaluate an adapted checkpoint

Curriculum and how the walker was trained: scripts/mbrl/omni_run.sh
EOF
    ;;
esac

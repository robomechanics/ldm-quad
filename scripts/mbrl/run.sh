#!/usr/bin/env bash
# Improved no-prior latent TD-MPC curriculum (post x=0.4-collapse fix + speed rebalance).
#
# Mirrors the live run: resume the known-good x=0.2 walker (model_150000) and ramp
# gently x=0.3 -> x=0.4, using the fixed hyperparameters (discount 0.99, q_dropout 0.1,
# entropy 3e-4, soft continue mask) together with the committed env/planner/model fixes
# (pessimistic min terminal value, class-balanced continue loss, loosened terminations,
# sharpened velocity-tracking reward). Stage A uses a FRESH replay because the env
# reward/termination changed (old transitions carry stale labels); Stage B warm-resumes.
set -uo pipefail

cd "$(dirname "$0")/../.."

PY="${PY:-/home/rml2/anaconda3/envs/isaaclab/bin/python}"
SEED="${SEED:-43}"
PROJECT="${WANDB_PROJECT:-ldm-quad-mbrl}"
WANDB_UPLOAD_VIDEOS="${WANDB_UPLOAD_VIDEOS:-1}"
WANDB_VIDEO_LENGTH="${WANDB_VIDEO_LENGTH:-300}"
WANDB_VIDEO_MAX_STEPS="${WANDB_VIDEO_MAX_STEPS:-300}"
# Known-good x=0.2 walker to resume from (override with BASE_CKPT=...).
BASE_CKPT="${BASE_CKPT:-logs/mbrl/go2_walk_2026-08-12_06-16-29/checkpoints/model_150000.pt}"

# ---- Speed knobs (all overridable via env; see scripts/mbrl/OPTIMIZATION.md) ----
# REPLAY_DEVICE=auto keeps the replay buffer in VRAM: ~60x faster sampling and no
#   CPU/swap pressure. Zero effect on learning (identical math). Safe to always keep.
REPLAY_DEVICE="${REPLAY_DEVICE:-auto}"
# PLANNER_ITERATIONS: 6 was the reference; 3 makes planning ~1.9x cheaper. MPPI is
#   warm-started + policy-guided, so the last iterations add little. Low risk, but it
#   only affects TRAINING-time data collection (deploy/eval can still plan at 6+).
PLANNER_ITERATIONS="${PLANNER_ITERATIONS:-3}"
# CANDIDATES: 512 = exploration breadth. Dropping to 256 is ~2x cheaper planning but
#   is the higher-risk knob (search breadth over a 96-dim space). Left at 512 by default.
CANDIDATES="${CANDIDATES:-512}"
# UTD: gradient updates per transition. 0.25 -> 0.125 halves update cost but does fewer
#   grad steps/transition (undertraining risk). Left at 0.25 by default.
UTD="${UTD:-0.25}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

latest_run_dir() {
  find logs/mbrl -maxdepth 1 -type d -name 'go2_walk_*' -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

upload_stage_video() {
  local run_dir="$1"
  local command_x="$2"
  local stage_name="$3"
  if [[ "$WANDB_UPLOAD_VIDEOS" != "1" ]]; then
    return
  fi

  local checkpoint="${run_dir}/checkpoints/model_best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    checkpoint="${run_dir}/checkpoints/model_final.pt"
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "[WARN] Skipping W&B video; checkpoint not found for ${stage_name}" >&2
    return
  fi

  echo "[RUN] Recording/uploading W&B video for ${stage_name}: ${checkpoint}"
  "$PY" scripts/mbrl/wandb_record_video.py \
    --checkpoint "$checkpoint" \
    --project "$PROJECT" \
    --command_x "$command_x" \
    --command_y 0.0 \
    --command_yaw 0.0 \
    --video_length "$WANDB_VIDEO_LENGTH" \
    --max_steps "$WANDB_VIDEO_MAX_STEPS" \
    --media_name "Videos / ${stage_name}" \
    || echo "[WARN] W&B video upload failed for ${stage_name}; continuing." >&2
}

common_args=(
  --headless
  --task Flat-Unitree-Go2-train-v0
  --num_envs 64
  --seed "$SEED"
  --buffer_capacity 1000000
  --replay_device "$REPLAY_DEVICE"
  --model_type latent
  --latent_dim 256
  --num_q 5
  --horizon 8
  --batch_size 1024
  --updates_per_step 8
  --utd "$UTD"
  --candidates "$CANDIDATES"
  --elites 64
  --planner mppi
  --planner_iterations "$PLANNER_ITERATIONS"
  --discount 0.99
  --planner_start_steps 2000
  --planner_min_length_fraction 0.0
  --planner_recovery_steps 2000
  --planner_recent_episodes 200
  --planner_temperature 0.5
  --planner_use_continue_model
  --planner_continue_threshold 0.5
  --planner_velocity_objective_weight 0.0
  --num_pi_trajs 24
  --q_dropout 0.1
  --entropy_coef 0.0003
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

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[$(ts)] [ERROR] base x=0.2 walker checkpoint not found: $BASE_CKPT" >&2
  exit 1
fi

echo "[$(ts)] [RUN] Stage A/2: resume good x=0.2 walker, slow forward command x=0.30 (FRESH replay)"
echo "[$(ts)] [RUN] Resuming from: $BASE_CKPT"
"$PY" -u scripts/mbrl/train.py \
  "${common_args[@]}" \
  --resume_checkpoint "$BASE_CKPT" \
  --no-auto_resume_replay \
  --resume_warmup_steps 5000 \
  --train_steps 190000 \
  --command_x 0.3 \
  --command_y 0.0 \
  --command_yaw 0.0 \
  --wandb_name "tdmpc_fixed_s${SEED}_stageA_x0p3"
STAGEA_RC=$?
if [[ $STAGEA_RC -ne 0 ]]; then
  echo "[$(ts)] [ERROR] Stage A exited non-zero (rc=$STAGEA_RC)." >&2
  exit $STAGEA_RC
fi

STAGEA_RUN="$(latest_run_dir)"
STAGEA_CKPT="${STAGEA_RUN}/checkpoints/model_final.pt"
if [[ ! -f "$STAGEA_CKPT" ]]; then
  echo "[$(ts)] [ERROR] Stage A checkpoint not found: $STAGEA_CKPT" >&2
  exit 1
fi
upload_stage_video "$STAGEA_RUN" 0.3 "stageA_x0p3"

echo "[$(ts)] [RUN] Stage B/2: resume Stage A, target forward command x=0.40 (warm replay)"
echo "[$(ts)] [RUN] Resuming from: $STAGEA_CKPT"
"$PY" -u scripts/mbrl/train.py \
  "${common_args[@]}" \
  --resume_checkpoint "$STAGEA_CKPT" \
  --train_steps 240000 \
  --command_x 0.4 \
  --command_y 0.0 \
  --command_yaw 0.0 \
  --wandb_name "tdmpc_fixed_s${SEED}_stageB_x0p4"
STAGEB_RC=$?
if [[ $STAGEB_RC -ne 0 ]]; then
  echo "[$(ts)] [ERROR] Stage B exited non-zero (rc=$STAGEB_RC)." >&2
  exit $STAGEB_RC
fi

STAGEB_RUN="$(latest_run_dir)"
upload_stage_video "$STAGEB_RUN" 0.4 "stageB_x0p4"
echo "[$(ts)] [RUN] Done. Final run: $STAGEB_RUN"
echo "[RUN] Play final:"
echo "$PY -u scripts/mbrl/play.py --checkpoint ${STAGEB_RUN}/checkpoints/model_final.pt --num_envs 1 --num_episodes 1 --max_steps 300 --command_x 0.4 --command_y 0.0 --command_yaw 0.0"
echo "[RUN] Play best stable-tracking checkpoint:"
echo "$PY -u scripts/mbrl/play.py --checkpoint ${STAGEB_RUN}/checkpoints/model_best.pt --num_envs 1 --num_episodes 1 --max_steps 300 --command_x 0.4 --command_y 0.0 --command_yaw 0.0"

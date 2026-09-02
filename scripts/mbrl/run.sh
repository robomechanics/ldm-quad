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
#   -> omni O1 (wander x[-0.2,0.5] y[+-0.15] yaw[+-0.3])  -- xy learned, YAW NOT TRACKED
#   -> omni O1a "Arm A" (same ranges + yaw reward 4.0/0.2)  -- inconclusive, yaw still dead
#   -> Stage T (turning): arcs x[0.2,0.4], yaw +-0.8, yaw rew 4.0/0.5, lin std 0.20 -- TURNING SOLVED
#      (first launch used yaw std 0.2 -> gradient ~9x too weak at err 0.44; relaunched at 0.5)
#   -> Stage L (lateral): + y +-0.35 -- combo 94/97/98% on all three axes at once
#   -> Stage M (mixing): x widened to [-0.3,0.5] -- omni WORKS; backward+fast-fwd broken
#   -> Stage B (backward): x [-0.35,0.2]  <- CURRENT RUN, fixes the backward gap
#
#   Stage   cmd_x  scale  Kept checkpoint (logs/mbrl/best_walker/)          ~vel
#   x0p2    0.2    0.25   best_walker_x0p2_stabletrack.pt / model_150000_*  --
#   x0p3    0.3    0.25   best_walker_x0p3.pt                                --
#   x0p4    0.4    0.25   final_x0p4_walker_v0p31.pt                         ~0.33
#   x0p4f   0.4    0.40   best_x0p4_ascale0p40_v0p39.pt                      ~0.39  (best FORWARD-only)
#   o1      wander 0.40   omni_O1_265k_xyok_yawfail.pt                      tx=0.062, yaw FAIL
#   o1a     wander 0.40   armA_yaw4p0_270k.pt                                 yaw still dead
#   T       arcs   0.40   stageT_turning_301k.pt              TURNING WORKS (tyaw 0.17 vs 0.42 do-nothing)
#   L       arcs+lat 0.40  stageL_omni_326k.pt        combo 94/97/98% (x,y,yaw simultaneously)
#   M       full   0.40   (queued; resumes L @326k, x now [-0.3,0.5])        <- FROZEN TASK
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
    # CURRENT EXPERIMENT: Stage M -- the MIXING stage that DEFINES THE FROZEN BENCHMARK TASK.
    # Lineage: Stage T solved turning (98% of command); Stage L added lateral (combo 94/97/98%
    # on all three axes simultaneously). Stage M widens x so standing / backward / pure-lateral
    # commands are in-distribution -- they were NOT in Stage L (x in [0.2,0.4] only), where
    # pure strafing at x=0 scored 81% one way, 49% the other, and fell over.
    # Fresh replay is INTENTIONAL: the buffer stores rewards at collection time and nothing
    # relabels them, so inheriting an old buffer trains the reward head on stale labels.
    # OPEN DECISION at freeze time: yaw commanded at x~0 demands IN-PLACE turning, the hard
    # skill (Stage T 56%, PPO 37%, both fall). Either accept it, or gate yaw on |x|.
    # NOTE: LAUNCH VIA systemd-run --user (NOT a VS Code terminal, NOT nohup/setsid):
    #   systemd-run --user --unit=stageM-mixing --collect \
    #     --working-directory=$PWD bash -c 'exec bash scripts/mbrl/run.sh train >> LOG 2>&1'
    # Root cause of the 2026-08-30/31 kills was NOT grid5 (measured: 1.2GB total, 7x175MB).
    # It was cpptools: ros2_ws/src/quad-sdk/.vscode/c_cpp_properties.json had
    # "limitSymbolsToIncludedHeaders": false with recursive globs over /opt/ros/jazzy/include/**
    # and /usr/include/**, so the tag parser indexed the whole tree and grew to 23-25GB
    # (23 OOM kills in 3 days). Global OOM made systemd-oomd sweep the entire snap.code.code-*
    # scope ("killed 35 process(es) in this unit"), taking any run launched from a VS Code
    # terminal with it. Fixed 2026-08-31 (flag -> true, /usr/include glob dropped).
    # A run in its own systemd scope survives that sweep regardless.
    # wait_and_run_stageM.sh gated on grid5 and is therefore OBSOLETE -- do not use it.
    # Set WANDER=0 CMD_X=<v> to fall back to the old fixed-forward-command recipe.
    # ============ CURRENT RUN: Stage B (BACKWARD) -- launched 2026-09-01 ============
    # Stage M gave working omnidirectional locomotion (see measured table below) but left
    # TWO gaps, both at the x EXTREMES: backward -0.3 (38% of command, FALLS) and
    # fast-forward 0.5 (60%, FALLS). Everything in the middle -- forward 0.4, standing,
    # both lateral directions, in-place turning, and all three axes combined -- works.
    #
    # WHY A SEPARATE STAGE, NOT MORE OF STAGE M: training Stage M on from 334k to 348k did
    # not converge, it TRADED AXES. model_347500 vs model_best @334k: backward 38%->49%
    # (falls->none) and in-place yaw 86%->92%, but lateral +y COLLAPSED 98%->63% and
    # lateral -y started FALLING. The full-range task is more than one stage holds at once.
    # This mirrors the curriculum's own history: Stage T (turning alone) and Stage L
    # (lateral alone) each succeeded; Stage M attacked three regions and fixed one.
    #
    # DESIGN: x narrowed to [-0.35, 0.2] so BACKWARD gets ~64% of samples (vs ~37% under
    # Stage M's [-0.3,0.5]) while moderate forward is retained to limit forgetting.
    # y +-0.3 and yaw +-0.8 are UNCHANGED so lateral and turning keep being exercised --
    # they already work and must not regress. Reward is unchanged too (yaw 4.0/0.5,
    # track_std 0.20): change ONE variable at a time, which is what worked for T and L.
    # Fast-forward 0.5 is a SEPARATE, later stage -- its cause is different (uniform x
    # sampling gave the top 0.05 band only ~6% of data, i.e. starvation, not skill).
    # ==============================================================================
    # ---------------- REPRODUCING THE STAGE M RESULT (2026-09-01) ----------------
    # run.sh alone does NOT reproduce the kept artifact. To reproduce exactly:
    #  1) STOP POINT: this recipe says TRAIN_STEPS=371000, but the run was STOPPED at
    #     env_steps=348450. The KEPT artifact is model_best.pt @ step ~333750
    #     (stable_tracking 0.457, saved 2026-09-01 08:50). Training past ~334k made it
    #     WORSE: over 334k->348k, len100 fell 682->580 and stable_tracking 0.372->0.284
    #     while tracking errors stalled. Set TRAIN_STEPS=334000 to stop at the keeper.
    #  2) REWARD IS NOT IN THIS FILE. alive=0.10, track w=8.0/std=0.11 and action
    #     scale=0.40 are baked into flat_env_cfg.py (pinned at commit 895797e).
    #     config.txt records reward_alive_weight/reward_track_weight as None for this
    #     reason -- they never pass through run.sh. Check out that revision to reproduce.
    #  3) --updates_per_step 8 below is DEAD: --utd 0.25 overrides it via
    #     train.py:1119 round(utd*num_envs) = round(0.25*64) = 16. config.txt records 16.
    #  4) Needs logs/mbrl/best_walker/stageL_omni_326k.pt (125MB, not in git).
    #
    # MEASURED RESULT of model_best @333750 (fixed-command sweep, logs/mbrl/stageM_sweep/,
    # 3 episodes x 2000 steps each, action_scale 0.40; len=1000 means NO falls):
    #   forward 0.4   -> 0.371 (93%)  no falls      in-place yaw 0.8 -> 0.688 (86%) no falls
    #   lateral +0.3  -> 0.293 (98%)  no falls      combo (.3,.2,.5) -> 86/86/106%  no falls
    #   lateral -0.3  -> -0.235 (78%) no falls      standing x=0     -> 0.006       no falls
    #   fast    0.5   -> 0.301 (60%)  FALLS (650)   backward -0.3    -> -0.115 (38%) FALLS (569)
    # => omnidirectional locomotion WORKS incl. in-place turning (Stage T was 56%, PPO 37%,
    #    both fell). Remaining gaps are the x EXTREMES only: fast-forward 0.5 and backward.
    # ---------------------------------------------------------------------------
    SEED="${SEED:-43}"
    WANDER="${WANDER:-1}"
    CMD_X="${CMD_X:-0.4}"
    # Stage M (MIXING -> defines the frozen benchmark task). x now spans BACKWARD..STOP..FORWARD.
    # Stage L trained x in [0.2,0.4] only, so x~0 was out-of-distribution: pure strafing from
    # standstill scored 81% one way but 49% the other AND fell. Widening x is what makes
    # standing / pure-lateral / near-stationary commands in-distribution at all.
    # X_MAX 0.5. An earlier cap at 0.4 was WRONG: it assumed ~0.39 m/s was a physical ceiling.
    # It is not -- PPO hits 0.4995 at command 0.5 with ZERO falls, so the robot can do 0.5.
    # MBRL fails there only because 0.5 is OUT-OF-DISTRIBUTION for it (Stage L trained
    # x in [0.2,0.4]; the command is part of the observation, obs[9:12], so 0.5 is an unseen
    # encoder input). The tell is REGRESSION not saturation: cmd 0.4 -> 0.374 m/s, 0 falls;
    # cmd 0.5 -> 0.333 m/s, 36 falls. A physical ceiling saturates; it does not go slower.
    # Widening x to 0.5 here is precisely what makes that command in-distribution.
    X_MIN="${X_MIN:--0.35}"; X_MAX="${X_MAX:-0.2}"
    # Stage L: lateral is the ONE new variable. y=0.4 fell over in the diagnostic, so 0.35
    # is the honest ceiling. Yaw stays at the PROVEN +-0.8: dropping it to +-0.2 would put the
    # command back under the +-0.2 sensor noise (the O1 failure) and let turning decay.
    Y_MIN="${Y_MIN:--0.3}"; Y_MAX="${Y_MAX:-0.3}"       # frozen-task lateral range
    YAW_MIN="${YAW_MIN:--0.8}"; YAW_MAX="${YAW_MAX:-0.8}"  # E|yaw|=0.4 > the 0.18-0.58 involuntary drift
    # Yaw tracking reward. std MUST be comparable to the CURRENT error, not the target error:
    # the kernel is exp(-err^2/std^2), and yaw error starts at ~0.44 (the robot does not turn).
    #   std 0.2 @ err 0.44 -> reward 0.008, gradient 0.17   (essentially flat / dead)
    #   std 0.5 @ err 0.44 -> reward 0.461, gradient 1.62   (~9x more learning signal)
    # Tighten std only AFTER turning exists, the same way the linear term was sharpened
    # to 0.11 only after forward walking worked.
    YAW_W="${YAW_W:-4.0}"; YAW_STD="${YAW_STD:-0.5}"
    # Linear std MUST be loosened for turning stages. With std 0.11 (tuned when x was the
    # ONLY error) the reward math is: ignore-yaw = 7.30 vs turn-costing-0.1m/s = 7.26, i.e.
    # turning earns nothing and any bigger speed cost is a net LOSS. At std 0.20 it becomes
    # 7.83 vs 9.99 -> turning clearly pays. Revisit when freezing the final task.
    TRACK_STD="${TRACK_STD:-0.20}"
    # best-checkpoint metric: default yaw weight 0.25 would let a great-x/no-yaw policy win
    # again; 0.5 matches the new reward ratio (yaw 4.0 / linear 8.0).
    EVAL_YAW_W="${EVAL_YAW_W:-0.5}"
    BC_COEF="${BC_COEF:-0.1}"                            # TD-M(PC)^2 BC term (0 = vanilla)
    # 2500 (not 5000): at ~0.37 steps/s a 5000-step interval risks losing ~3.7h to a silent
    # kill (Stage M was OOM/killed once at 16:30 on 2026-08-30). Disk cost only.
    SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
    RESUME="${RESUME:-$BW/stageM_omni_334k.pt}"             # Stage M keeper @~334k (omni works; backward+fast broken)
    # 50k new steps (not 30k): Stage M fixes THREE out-of-distribution regions at once
    # (fast-forward 0.5, standing x~0, backward x<0), where Stage T/L each needed ~25-30k for
    # ONE. Uniform sampling over the 0.8-wide x range also gives the top 0.05 band only ~6%
    # of the data, so the 0.5 end is the thinnest-covered part. Checkpoints every 2500 steps
    # mean we can stop early the moment it plateaus (as Stage T was stopped at 301k).
    TRAIN_STEPS="${TRAIN_STEPS:-354000}"    # resume @~334k + ~20k backward-focused steps
    if [[ "$WANDER" == "1" ]]; then
      CMD_ARGS=(--wander --wander_x_min "$X_MIN" --wander_x_max "$X_MAX" \
                --wander_y_min "$Y_MIN" --wander_y_max "$Y_MAX" \
                --wander_yaw_min "$YAW_MIN" --wander_yaw_max "$YAW_MAX")
      WANDB_NAME="${WANDB_NAME:-stageB_backward_s${SEED}}"
      echo "[run] TRAIN Stage B (backward focus): x[$X_MIN,$X_MAX] y[$Y_MIN,$Y_MAX] yaw[$YAW_MIN,$YAW_MAX] yaw_rew=w$YAW_W/std$YAW_STD lin_std=$TRACK_STD bc=$BC_COEF resume=$RESUME -> $TRAIN_STEPS"
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
      --reward_yaw_weight "$YAW_W" --reward_yaw_std "$YAW_STD" \
      --reward_track_std "$TRACK_STD" \
      --eval_tracking_yaw_weight "$EVAL_YAW_W" \
      --seed_steps 2000 --seed_action_mode smooth_zero --seed_action_noise_std 0.08 --seed_action_smoothing 0.92 \
      --seed_pretrain_updates 5000 --seed_policy_noise 0.02 \
      --save_interval "$SAVE_INTERVAL" --max_checkpoints 5 --save_replay \
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
  o1      wander x/y/yaw (O1)   0.40   omni_O1_265k_xyok_yawfail.pt      tx=.062 yaw FAIL
  o1a     O1 + yaw rew 4.0/.2  0.40   armA_yaw4p0_270k.pt               yaw dead
  T       arcs + yaw 4.0/0.5  0.40   stageT_turning_301k.pt            TURNING OK
  L       T + lateral +-0.35   0.40   stageL_omni_326k.pt               combo 94/97/98%
  M       full x[-0.3,0.5]      0.40   (TRAINING NOW, resumes L @326k)   <- frozen task

  ./run.sh play  [x0p2|x0p3|x0p4|x0p4f]   # play a kept walker (default x0p4f)
  ./run.sh train                          # omni O1 recipe (WANDER=1 default; WANDER=0 CMD_X= for fixed)
                                          # overrides: RESUME= TRAIN_STEPS= SEED= BC_COEF= YAW_W= YAW_STD= X_MIN/MAX= Y_MIN/MAX= YAW_MIN/MAX=
EOF
    ;;
esac

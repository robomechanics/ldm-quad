# Axis-trading fix: path to omnidirectional locomotion (2026-09-06)

Numbered items match the running table kept by both Claude sessions. Diagnosis: the world model holds
both x~0 skills (strafe, in-place turn); the yaw reward `track_ang_vel_z_exp` (w 8.0, std 0.28 since
Stage R) taxes the strafing gait's natural yaw wobble (|wz| ~0.12, peaks ~0.3 -> forfeits 5.5 of 8/step)
while turning induces no lateral wobble. One-directional reward coupling => lateral<->yaw anti-correlation
(T2, r=-0.64). Verified in traces (velobj15a/15d). Everything below follows from that.

## Stage 1 — eval only, ~25 min (launch now)
- **(B)** fall diagnostic: `--diagnostics_interval 1`, 3 eps each:
  388k lat_yp0p3 @ W=0 and @ full bounded objective (exp, W 0.5, lin 8 / yaw 8);
  388k inplace_yaw @ yaw-only (lin 0, yaw 8, W 0.5) and @ yaw-only + deadband 0.25.
  Read: termination term, proj_grav_y vs vy/wz in the last 20 steps, base_height.
- **15g** command-gated yaw objective, G=0.1: 4 ckpts (382/388/390/394k) x lat+yaw, plus combo
  (0.3, 0.2, 0.5) on 388k and 390k.
  PASS: lateral within 5 pts of W=0 with mean_length >= 999 on 4/4, yaw mean >= 90.

## Build while Stage 1 runs (code only)
- **#9** clean base velocity per transition in the replay (root_lin_vel_b xy, root_ang_vel_b z; NaN sentinel
  for legacy); sample-time tracking-reward recompute (env returns the command-independent remainder when on).
  Storage ON for Phase A, recompute OFF until Phase B.
- **#16** `track_ang_vel_z_exp_deadband` in mdp/rewards.py behind `--reward_yaw_deadband` (d 0.20; std 0.28
  and w 8.0 unchanged — the deadband is the only variable).
- **#3** stratified command sampling flag (fraction of envs per resample on canonical commands; per-region
  weight so Phase C can bias backward). OFF for Phase A.
- **#10** worst-axis `save_best_metric` option (max per-axis error).
- Wire `--planner_velocity_objective_{form,lin_weight,lin_std,yaw_weight,yaw_std,yaw_deadband,yaw_gate}`
  into train.py argparse and both build_planner calls.
- **#6a** command skip-connection (concat obs[9:12] to dynamics/reward/Q/pi inputs; zero-init new columns;
  partial checkpoint load) behind a flag; smoke-tested; NOT enabled.
- Sweeper: conditions ref/back/lat/yaw/combo; fail loudly on a missing checkpoint (play.py exits 0 today).

## Review points — do NOT auto-chain phases
Each arrow below is a stop: post the mean_length-first tables, then decide.
- Stage 1 -> Phase A: 15g PASS is mechanical (criterion above). (B) is a judgement read: which term
  terminates and whether the gated objective could trigger it. Review both before launching A.
- Phase A -> Phase B: check (i) lateral/yaw/forward hold or improve vs T2 396k with no new falls,
  (ii) #9 clean velocities are present in the new buffer, (iii) the #9 recompute reproduces the env
  reward on fresh transitions (max abs error small) BEFORE relying on it for relabelling.
- Phase B -> fork / Phase C: Phase B is THE test of the wobble-tax hypothesis. Read at least 5 sweep
  points. Both axes >= 75 with no falls and no anti-correlation -> Phase C. Still anti-correlated -> #6a.
  Neither Phase C nor #6a launches without this read.

## Phase A = #15c (~10k steps, warm)
Resume `logs/mbrl/history/stageT2_warmbuffer_TRADING_CONFIRMED/checkpoints/model_396000.pt` WITH its
`replay_latest.pt`. Current reward. Gated yaw objective ON in collection (exp, W 0.5, lin 0, yaw 8, G 0.1).
#9 storage on, #10 on, save_interval 2000, sweeper attached. Update run.sh as the Phase A recipe.
Launch via `systemd-run --user`.

## Phase B = #16 (~20k steps, warm)
From Phase A's final checkpoint + buffer: `--reward_yaw_deadband 0.20` and the #9 recompute so the Phase A
buffer is relabelled exactly (legacy transitions without clean velocity drop out). Sweeper attached.
Success: lateral and in-place yaw both >= 75% on the same checkpoint, no falls, holding across sweeps.

## Phase C = #14 via #3 (~20k steps, warm)
From Phase B: stratified sampling biased to backward. Backward is a floor, not a swing.

## The one fork
If after Phase B lateral and yaw are still anti-correlated across sweeps: enable **#6a** and continue warm.

## Then
Freeze ranges + reward as the benchmark task -> retrain PPO on the frozen task -> 4 perturbations x 3-way
ablation (the thesis). Report tables with mean_length FIRST; pct_of_cmd alone hid every fall so far.

Dead/dropped: 0, 2, 8 (dead); 5, 13 (dropped: #15 removed their justification); 6 (dropped for 6a);
4, 7 (parked). Done: 1, 11, 12, 15, 15a, 15b, 15d.

# Budget experiment (2026-09-08): which lever raises the ceiling?

Finding: across 8 artifacts every reward/objective change redistributed capability (mean of the 4 axes 74.6 +- 13.5,
best four within 2 pts). Test the levers that could raise the total, one variable per arm, same base, same reward,
same measurement.

## Common protocol (every arm)
- Base: Phase A 402k (`best_walker/phaseA_gatedyaw_402k_backward.pt`) + Phase A's OWN replay at 402k (labels already
  match: stock yaw kernel, orientation -1). No relabel. Phase B's deadband is NOT carried (it traded backward for lateral).
- Reward/objective FROZEN at Phase A's: joint linear 8.0/0.20, yaw 8.0/0.28, orientation -1, gated yaw objective in
  collection (exp, W 0.5, lin 0, yaw 8, gate 0.1). save_best_metric worst_axis. clean_vel on.
- 20k steps per arm (402k -> 422k), save_interval 2000, sweeper attached, objective=off (plain planner) primary.
- Metrics per checkpoint: the 5 fixed conditions, mean_length first. Primary score = WORST-AXIS % among
  fwd/back/lat/yaw with zero falls on all 5; secondary = mean of the 4 axes; tertiary = swing amplitude
  (std across the last 5 checkpoints per axis: the trading metric). Compare at matched env steps; for the
  throughput arm also at matched wall-clock.
- Decision: an arm "raises the budget" if its mean of 4 axes over the last 5 checkpoints exceeds 80 (control ~75)
  AND worst-axis >= 60 with no falls on >= 3 consecutive checkpoints. Winner = highest worst-axis, tie-break mean.

## Arms (numbered as table items)
- **#18 control**: base continued unchanged, 20k. The null: does time alone move the total? Also the reference for
  swing amplitude. Cheapest, run FIRST.
- **#6a command skip-connection**: concat raw cmd (obs[9:12]) to the inputs of dynamics, reward, Q ensemble, and pi;
  zero-init the new input columns; partial checkpoint load (all existing weights copied); Adam state reset for new
  params only. Function-preserving at step 0 (verify: identical planner actions on a fixed batch before/after load).
- **#5w width expansion, WARM** (net2net): latent 256 -> 512 (SimNorm groups 32 -> 64), hidden 512 -> 1024.
  Copy old weights into the top-left blocks, zero-init new encoder output rows (new SimNorm groups start uniform),
  zero-init all new INPUT columns of dynamics/reward/Q/pi/continue/physical heads so the old function is preserved
  exactly; verify identical outputs on a fixed batch before training. If verification fails, this arm is invalid.
- **#19 throughput**: num_envs 64 -> 256. PROBE FIRST (10 min): measure steps/s at 256 envs with the planner; keep
  updates-per-env-step constant by scaling --utd 0.25 -> 0.0625 (utd*num_envs = 16 updates/step). HAZARD: the replay's
  sequence sampler assumes stride = num_envs at write time; a 64-env buffer loaded into a 256-env run makes every old
  row an invalid sequence start (cold start on sequences). Fix before running: store an explicit per-row `next_index`
  (or per-row stride) in ReplayBuffer and use it in _valid_sequence_starts/sample_sequences instead of the current
  global stride. Compare at matched env steps AND matched wall-clock (report both).
- **#20 duration** (after the winner is known): winner's config, 100k+ steps, reward/objective frozen, to see
  whether the model settles when left alone. Not an arm of the comparison.

## Order and cost (one GPU, sequential, ~1 day each)
#18 control -> #6a -> #5w -> #19 (probe first, then the arm) -> #20 with the winner. ~4 days to the winner.
In PARALLEL, no GPU: freeze the task on Phase A 402k; prepare the PPO retrain on the frozen ranges and the
perturbation-study configs, so the thesis starts the moment the GPU frees.

## Review points
Each arm -> read the 5-checkpoint table before launching the next; abort an arm early only on falls >= 2 conditions
across 3 consecutive checkpoints (it is then already worse than the base).

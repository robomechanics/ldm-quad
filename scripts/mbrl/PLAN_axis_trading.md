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

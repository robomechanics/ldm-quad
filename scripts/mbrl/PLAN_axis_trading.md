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

---

# RESULT: #18 control, and why the measurement had to be rebuilt first (2026-09-08)

## #18 control, first 2 sweep points (N=3 harness, superseded)
Five consecutive checkpoints of ONE unchanged config (398k-402k are Phase A, 404k/406k are #18
resumed from 402k with identical settings; all `objective=off`, same sweeper):

| steps | fwd | back | lat | yaw | mean4 | worst | falls |
|---|---|---|---|---|---|---|---|
| 398k | 69.9 | 75.4 | 50.4 | 91.7 | 71.9 | 50.4 | 1 |
| 400k | 84.4 | 76.2 | 72.3 | 69.9 | 75.7 | 69.9 | 1 |
| **402k** (the kept artifact) | 90.6 | 87.0 | 56.2 | 75.5 | 77.3 | 56.2 | **0** |
| 404k | 75.7 | 96.5 | 31.1 | 52.8 | 64.0 | 31.1 | 2 |
| 406k | 109.1 | 78.7 | 22.9 | 69.6 | 70.1 | 22.9 | 1 |

**Doing nothing moves the score as much as every intervention did.** Within this single
unchanged config `mean4` spans 13.3 points (64.0-77.3, sd 5.2) and **worst-axis spans 47 points**
(22.9-69.9). Across all EIGHT different interventions the spreads were 13.5 and 30.5. The noise
band of the null is as wide as -- for worst-axis, wider than -- the entire effect range the arms
are ranked by. No arm verdict resting on worst-axis at N=3 was resolvable, this one included.

**The "conserved budget" finding is not yet evidence of conservation.** Eight artifacts landing
within +-13.5 of each other is exactly what one config produces by itself. Conservation may still
be real; it has not been measured.

**#18 did not "degrade from its parent."** 402k is the only zero-fall point of the five, and it
was selected as the keeper by max-worst-axis-with-zero-falls. Comparing a max-selected point to
unselected continuations is regression to the mean.

## Root cause (sweep_checkpoints.py)
1. `--num_envs 1 --episodes 3`: every cell was 3 episodes of ONE robot.
2. `achieved` time-averaged the WHOLE rollout, so post-reset acceleration from standstill and
   post-fall near-zero velocity counted as achieved velocity. One early fall depressed the number
   twice -- as the fall, and by contaminating the mean.
3. `falls` was inferred as `mean_length < 999`, a binary off one averaged number, not a rate.

Lateral's apparent collapse (72.3->56.2->31.1->22.9) tracks its `mean_length`
(562->1000->766->265) almost exactly: it was largely a fall-rate reading at N=3.

## Harness now (use this for every future arm)
- `play.py` writes `episodes.csv`: one row per completed episode per env
  (`env, ep_index, length, ret, vx, vy, vyaw, terminated`), velocity averaged WITHIN the episode
  from the PRE-step obs (`next_obs` is already the reset observation on a done step).
  `terminated` is split from `truncated`, so a fall is a real termination.
- The statistic is the **first episode of each env**: one independent sample per env. Pooling all
  completed episodes would over-weight fast-failing envs, which finish episode 1 early and start a
  second before slow healthy envs finish their first.
- `--max_steps 1000` (the episode timeout) with the episode cap effectively unbounded, so every
  env contributes exactly one episode. Bounding by episode count instead would end the rollout
  early and silently drop the slowest -- i.e. the best -- envs.
- CSV gains `achieved_sem`, `pct_sem`, `n_eps`, `fall_rate`. A missing `episodes.csv` skips the row
  loudly rather than falling back to the old statistic.
- Cost measured on this box, sharing the GPU with training: 0.355 steps/s at 64 envs vs 0.58 at 1
  env -- 64 envs costs only ~1.4x per step. **~47 min/condition**, ~3.9 h for 5 conditions, for
  ~4.6x less standard error. New flags for keeping pace: `--conditions`, `--min_step_gap`,
  `--checkpoint_list` (re-measure `best_walker/*.pt` artifacts).

**Consequence for the decision rule above:** the thresholds (mean4 > 80, worst-axis >= 60) are
only meaningful against the N=64 harness. Re-measure before ranking any arm, and re-measure the
8-artifact table before repeating the conservation claim.

## #6a: built and verified, NOT launched
`--command_skip_indices 9,10,11` routes the raw command around the encoder into
dynamics/reward/Q/pi. New input columns are zero-init and a no-skip checkpoint is grafted
column-by-column (`expand_state_dict_for_command_skip`) -- `load_state_dict(strict=False)` would
NOT do: widening the first Linear shifts the action columns, the shapes mismatch, torch skips the
tensor, and the head silently keeps its random init.

`scripts/mbrl/verify_command_skip.py` is the gate. On `model_406000.pt`:
float64 worst max|diff| **1.99e-13** (algebraically exact), float32 1.19e-4 on logits
(matmul reduction-order round-off; 1.07e-6 on the reward value). +30,720 params, all zero.
Note `Q(return_type="min")` draws a RANDOM pair of heads via `torch.randperm`, so the check
compares `return_type="all"`; comparing the min measures the draw, not the graft.
Adam state for the 20 widened tensors is dropped on graft: the moments are not reshaped by
`load_state_dict` (it only casts), and carrying the old `step` with `exp_avg_sq==0` in the new
columns makes the first update there ~1/sqrt(1-beta2) ~ 30x oversized, knocking them off zero.
`play.py` now inherits `command_skip_indices` from the checkpoint -- without that a skip-trained
model is untrainable-then-unevaluable, the same asymmetry that contaminated Phase A/A2.

## #19 hazard: reduced in scope, deliberately
The per-row `next_index` rewrite was NOT done. The stride assumption is **not** a correctness bug:
`_valid_sequence_starts` (replay.py:252-256) checks env_ids, episode_ids AND step_id
consecutiveness, so mis-strided rows are rejected, never silently mixed. It is a silent CAPACITY
collapse -- and worse, `train_ready` gates on `can_sample_sequences` (train.py:1937), so a stride
mismatch would make the run step forever, collecting data with ZERO gradient updates and no error.
Guards added instead: a hard failure at resume if a non-empty buffer yields zero valid sequences
(naming the stride vs num_envs mismatch), and a watchdog that raises after 20k env steps past the
warmup with no updates. The per-row chain is only needed if we actually change num_envs for
training -- do it then, as part of #19.

---

# Three distinct lateral defects, separated (2026-09-08, late)

The old harness reported one number for lateral ("22.9%, falls YES"). Under the 64-env harness plus
`--command_start_step` it resolves into three different failures with different fixes.

## 1. LANDING: command-induced, not a spawn artifact
Same 406k checkpoint, same harness, objective off, only the command differs:

| command | dip at step 10 | margin over the 0.20 cut | landing deaths | fall_rate | mean_length |
|---|---|---|---|---|---|
| (0, 0, 0) | 0.2498 | +4.98 cm | **0/64 = 0%** | 0.094 | 943 |
| (0, 0.3, 0) | 0.2162 | +1.62 cm | **30/64 = 47%** | 0.891 | 302 |

Fisher one-sided p = 1.05e-11. The lateral command deepens the settling dip by 3.4 cm, pushing ~47%
of envs under `terminations.base_height` (`minimum_height = 0.20`). So the spawn height (~0.398 vs a
~0.32 nominal stance) is NOT the confound -- with no command the robot clears the cut by 5 cm and
nothing dies. **Do not lower the spawn**: it would hide a real controller defect.

Verified NOT a 64-env artifact: the old N=1 sweep of the same checkpoint shows the identical
collapse (0.3976 -> 0.2162 by step 10, `term_base_height` at step 25) with episodes {13, 14, 765},
mean 264 = the "mean_length 265" that was recorded. The harness made it legible, it did not cause it.

## 2. SUSTAIN: reachable in 5 steps, unsustainable after ~15
From settled (`--command_start_step 50`), the same 406k checkpoint reaches **90% of command** within
5 steps and never collapses (base_height steady 0.27-0.29, no deaths in 130 steps) -- then stalls:
0.2696 -> 0.2740 -> 0.1100 -> 0.0145 -> 0.2203 -> -0.0419. Lateral is not unreachable; it is
unsustainable, and the post-settle mean was hiding that waveform.

Mechanisms tested on the trace:
- **Yaw-wobble tax: REJECTED as the stall driver.** No lag is significant (r = -0.273 at lag 0,
  sign flipping by lag 15, n=16), and the highest wobble of the rollout (0.581) coincides with 70%
  strafing. Decisively, the tax is SATURATED -- median 5.59 of 8.0 forfeited, 69% of steps above
  5.0, and exp(-w^2/0.28^2) ~ 0.03 at w=0.5 -- so it has no gradient left to modulate behaviour.
  It remains a good explanation for why lateral is GLOBALLY discouraged (a near-permanent 5.6/8
  drag whenever the robot strafes). Those are different claims.
- **Lean: SUPPORTED.** corr(lean at t, vy at t+lag) is significant at lag 5-10 and decays by 15:
  tilt -0.716/-0.696, |proj_grav_x| -0.758/-0.704, |proj_grav_y| -0.575/-0.595 (crit ~0.50 at n=16).
  Pitch outpredicts roll for a lateral stall. gx is single-signed here, so read magnitude only.
- **Planner over-caution: REJECTED.** `planner_candidate_return_best` is equal or HIGHER when
  stalled (20.43, 21.48, 22.62) than when strafing at 90% (19.85, 20.88), with its single highest
  prediction on the worst behaviour (vy negative). The planner is not backing off; it cannot tell
  the two apart.

## 3. The reward head cannot see the difference (localised)
Split by regime (strafing |vy|>0.19 n=5, stalled |vy|<0.06 n=6):

| | must resolve | own error | SNR |
|---|---|---|---|
| reward head | 0.1356 /step | 0.115-0.138 /step | **0.99-1.18** |
| physical head | 0.2551 m/s | RMSE 0.088-0.113 | **2.3-2.9 sigma** |

`model_latent_consistency_mse` barely moves between regimes (ratio 1.09), so the information IS in
the latent and the physical head reads it -- the blind spot is the REWARD HEAD's command handling,
not the latent and not the Q bootstrap. (`planner_candidate_return_best` includes the Q bootstrap,
so the raw 2-4 unit misranking is confounded; the SNR split is the sound version.) Reward-head error
~0.13 is ~57% of the entire per-step reward (~0.227). Physical RMSE ~0.088 matches the 0.084 on
record, so it is corroborated outside this trace.

**This is the strongest case for #6a, and specifically for the command entering the REWARD head.**

CAVEAT ON ALL OF SECTION 2-3: n=16 samples (n=5/n=6 for the split) from ONE 8-env 130-step smoke.
Directions are consistent and the lean correlations clear the threshold, but these are hypotheses
with numbers attached. The 64-env 1000-step runs are the measurement. Re-report the same splits
per checkpoint so the SNR claim gets its n.

## Instrumentation traps found
- `planner_predicted_*_return_*` are constant 0.0 and `planner_use_prior_fallback` /
  `planner_prior_accept_rate` are EMPTY whenever no action prior is loaded (plain mppi). A flat
  margin there is NOT evidence about the planner -- the comparison never runs.
- `model_velocity_mse` / `model_obs_rmse` were written as 0.0 for every latent run: the latent model
  is decoder-free so there is no predicted observation, and only the ensemble path can fill them.
  Now ABSENT (blank cell) instead of 0.0, which previously read as "perfect prediction".
- `model_continue_bce` constant 4e-08, `model_continue_accuracy` constant 1.0: the continue head is
  fully saturated, exactly as `LatentWorldModel.loss`'s own comment warns.

---

# SUPERSEDES the n=3 noise section above (2026-09-08, n=64 measurement)

Earlier tonight this doc recorded that the 402k->406k control-arm swing was NOT established
(Fisher p = 0.162 on 0/3 vs 30/64). That was the correct read of the evidence then. The n=64
measurement has now been taken and it reverses the conclusion.

## The control arm genuinely drifts (from-spawn lateral, objective off, n=64 each)

| | landing_fail | n_landed | fall_rate_post | mean_len_post | lateral |
|---|---|---|---|---|---|
| **402k** (keeper) | 2/64 = **3.1%** | 62 | **0.0000** | 1000 | **55.6% +- 0.6%** |
| **406k** (+4k steps, nothing changed) | 30/64 = **46.9%** | 34 | **0.7941** | 558 | 30.1% +- 1.9% (whole-episode) |

Fisher one-sided **p = 2.29e-09**. Four thousand steps of the unchanged Phase A config took lateral
landing failure from 3% to 47%. 402k stays the keeper, and it both lands AND sustains: zero falls
after landing, every survivor ran the full 1000 steps.
(406k's 30.1% is a whole-episode mean from a pre-split run, so it is not strictly comparable to
402k's post-settle 55.6%; E1 supplies the matched post-settle figure.)

## Why the conservation table really failed -- correcting this doc's earlier claim

| ckpt | old N=3 | new post-settle (n=64) | landing fail | |
|---|---|---|---|---|
| 402k | 56.2% | 55.6% | 3.1% | old number essentially EXACT |
| 406k | 22.9% | 29.9% | 46.9% | landing deaths folded into the mean |

The old harness was accurate wherever there were no landing failures and wrong precisely where there
were. So the 8-artifact table failed because the metric **summed two different failures** (landing
survival + velocity tracking), NOT merely because n=3 was underpowered. A large part of the 47-point
worst-axis spread attributed to noise above was real signal, mismeasured. Both defects were present;
the conflation was the bigger one.

## How arms are judged from here (replaces "mean4 > 80, worst-axis >= 60")

- **Measurement noise at n=64: +-0.6%** (clean checkpoint) / **+-1.9%** (landing-failing). That is
  the threshold an intervention must clear.
- **Do NOT use the 402k->406k swing as a noise band.** It is a real deterministic drift at p=2e-9;
  it does not shrink with more episodes, and treating it as a band would declare every arm
  indistinguishable from doing nothing -- a category error that would abort the experiment.
- **Compare each arm to the CONTROL AT MATCHED ENV STEPS**: landing_fail_rate, fall_rate_post, and
  post-settle tracking per axis. There is no single scalar bar because the control's own value moves
  (3% / 56% at 402k; 47% / 30% at 406k).

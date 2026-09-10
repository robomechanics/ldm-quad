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

---

# RULE: never use training-distribution error for model selection or drift detection

Measured 2026-09-08 with `scripts/mbrl/kstep_openloop_error.py` (open-loop k-step rollouts on real
buffer sequences, 986,355 valid H=8 sequences, n=512, CPU-only):

| | phys RMSE k=1 -> k=8 | reward \|err\| k=1 -> k=8 | behaviour (lateral, from-spawn) |
|---|---|---|---|
| 402k (keeper) | 0.0985 -> 0.1247 | 0.0273 -> 0.0339 | 3% landing failure, fall_rate_post 0.00 |
| 406k (+4k steps) | 0.0836 -> 0.1137 | 0.0216 -> 0.0285 | **47% landing failure, fall_rate_post 0.79** |

**406k is BETTER than the keeper on its own training distribution on both heads, while being
catastrophically worse in behaviour.** So "watch the model losses" would have shown nothing, and
on-distribution error cannot rank checkpoints or detect the drift. Model selection and drift
detection must use OFF-distribution fixed-command evaluation.

## Where the model actually fails: off-distribution, and only in the reward head

Same latent, same erratic rollout, 406k:

| head | on-distribution | under sustained lateral cmd | degradation |
|---|---|---|---|
| reward | 0.0216 | 0.1215 | **5.6x** |
| physical | 0.0836 | 0.1079 | **1.3x** |
| | | differential | **4.4x** |

Because both heads read the same latent on the same rollout, "erratic behaviour is harder to
predict" applies equally to both and cannot produce a 4.4x differential. This is the reward-head
command-handling blind spot, and it is the strongest evidence for #6a.

Do NOT quote a relative-error ratio here (it came out 35x): the env reward under a lateral command
is 0.0357 against the buffer's 0.2213, so dividing by a 6x-smaller denominator inflates the figure
and invites the objection that any head looks bad when the true reward is near zero. Quote the
absolute per-head degradation and the differential.

**The SIGN of the under-command error is NOT measured.** `model_reward_abs_error` is unsigned, so
|err| 0.1215 against a true 0.0357 is consistent with predicting either +0.157 or -0.084, and the
buffer's reward range (-0.2818 .. 0.3214) admits both. On-distribution the signed bias is only
-0.8% to -3.0% of target (slightly PESSIMISTIC), so there is no general optimism to extrapolate.
The planner ranking stalled states higher favours over-prediction but is confounded by the Q
bootstrap. Log signed reward error in `play.py` diagnostics before any writeup asserts a direction.

## Horizon integration
Consecutive-step error correlation ALONG the rollout is ~0.21-0.30 for both heads in both
checkpoints (gains 2.17-2.35x over H=8 vs 2.83x if independent). There is NO reward-vs-physical
asymmetry in horizon correlation -- an earlier claim of one came from 5-step-spaced wall-clock
one-step error, which got the ordering backwards. Open-loop error grows only 24-36% from k=1 to
k=8, so the model is not diverging over the planning horizon.

---

# E1-E4 RESULTS: from-settled lateral, 64 envs, max_steps 1000, command_start_step 50 (2026-09-09)

All four `objective=off` unless stated, all eight objective flags passed EXPLICITLY (every checkpoint
bakes in W 0.5 / yaw 8.0 / gate 0.1, so an unspecified flag is inherited).

| run | ckpt | objective | landing_fail | n_landed | fall_rate_post | mean_len_post | post-settle lateral | fails by |
|---|---|---|---|---|---|---|---|---|
| E1 | 406k | off | 0.000 | 64 | **0.625** | 673 | 24.2% +- 1.6% | SINKING (base_h term 0.203) |
| E2 | 406k | lin W0.25 | 0.000 | 64 | **0.750** | 557 | **106.2% +- 0.2%** | ROLL-OVER (orient term 0.312) |
| E3 | **402k keeper** | off | 0.000 | 64 | **0.000** | 1000 | 56.2% +- 0.5% | nothing fires |
| E4 | 408k Phase B | off | 0.094 | 58 | 0.034 | 978 | **84.3% +- 0.5%** | spawn settle only |
| (ref) | 402k keeper | off, FROM-SPAWN | 0.031 | 62 | 0.000 | 1000 | 55.6% +- 0.6% | 2/64 landing |

## What each answers

**E1/E2 -- the explicit velocity objective solves SUSTAIN at zero training.** 24.2% -> 106.2%,
sustained 94-102% at every 50-step sample across the full 1000 steps, no decay. (The earlier "90%"
from an 8-env smoke was an ONSET transient and did not survive; this one is a real regime.) It works
by reading velocity from the physical head, bypassing the reward head.
BUT it CHANGES THE FAILURE MODE: base_height termination 0.203 -> 0.000, bad_orientation 0.062 ->
0.312. E1 falls by sinking; E2 falls by rolling over. fall_rate_post 0.625 -> 0.750 is NOT
significant (p=0.182) but that non-difference hides a complete mechanism change.

**E3 -- the keeper is the only artifact stable in BOTH initiation modes.** 0/64 falls from settled
with nothing firing at all, 2/64 landing deaths from spawn, ~56% either way. This is why 402k stays
the keeper.

**E4 -- the deadband model sustains markedly better: 84.3% vs the keeper's 56.2%, +28 points.**
Its 6 failures are all inside the zero-command hold (length < 50), i.e. it fails to survive the spawn
settle 9.4% of the time even with NO command; once settled it is near-flawless (2/58 falls, 978 mean
length, no aggregate terminations). vs keeper: landing 6/64 vs 0/64 is significant (p=0.028);
fall_rate_post 2/58 vs 0/64 is not (p=0.224).

## LATERAL ONLY -- this cannot pick between 402k and 408k

Taking old-harness numbers as valid where no landing failures occurred (the old metric was exact
there -- 402k lateral 56.2 old vs 55.6 new, 408k 85.2 old vs 84.3 new, both confirmed):

| | fwd | back | lat | yaw | mean4 | worst |
|---|---|---|---|---|---|---|
| 402k keeper | 90.6 | **87.0** | 56.2 | 75.5 | 77.3 | **56.2** |
| 408k Phase B | 89.6 | **45.9** | 85.2 | 78.2 | 74.7 | 45.9 |

408k wins lateral by 28 points and loses backward by 41. **402k keeps the keeper slot on worst-axis.**
The decisive missing measurement is BACKWARD on the new harness for both -- that is the axis that
decides it, not lateral. This is the axis-trading pattern again, now measurable.

## Reward-head blind spot: confirmed on the keeper, cleanest form

| | physical on-dist -> under cmd | reward on-dist -> under cmd |
|---|---|---|
| 402k (E3) | 0.0985 -> 0.0943 = **0.96x** | 0.0273 -> 0.1651 = **6.05x** |
| 406k (E1) | 0.0836 -> 0.1079 = 1.29x | 0.0216 -> 0.1215 = 5.63x |

On the keeper the physical head does not degrade AT ALL under a sustained lateral command while the
reward head degrades 6x. Same latent, same rollout, so erraticness cannot explain it.
Reward-head accuracy is DECOUPLED FROM BEHAVIOUR IN BOTH DIRECTIONS: the keeper has the worst reward
error of the three runs (0.1651) with the best behaviour, and 406k beats the keeper on-distribution
while behaving far worse. Sign of the under-command error is still unmeasured.

## Lean-magnitude hypothesis: DEAD

| | \|pg_y\| mean | \|pg_y\| sd | orient term | fall_post |
|---|---|---|---|---|
| E1 | 0.0824 | 0.0181 | 0.062 | 0.625 |
| E2 | 0.1091 | 0.0154 | 0.312 | 0.750 |
| E3 | 0.1316 | 0.0131 | 0.000 | 0.000 |
| E4 | 0.1362 | 0.0158 | 0.000 | 0.034 |

Lean MAGNITUDE is anti-correlated with falling -- E3/E4 lean the most and fall least. If lean matters
it is STEADINESS (the sd column runs the other way), which is a hypothesis for its own test, not a
claim on four points.

## Landing failure has TWO sub-cases. Keep them apart; they have different fixes.

Landing failure is governed by one continuous quantity — how deep the spawn drop goes against the
hard `minimum_height = 0.20` cut — and it is monotone in the margin:

| | dip @ step10 | margin | landing fail |
|---|---|---|---|
| 406k, ZERO command | 0.2498 | 5.0 cm | 0/64 = 0.0% |
| 402k keeper, lateral | 0.2510 | 5.1 cm | 0/64 = 0.0% |
| 408k Phase B, lateral | 0.2280 | 2.8 cm | 6/64 = 9.4% |
| 406k, lateral FROM SPAWN | 0.2162 | 1.6 cm | 30/64 = 46.9% |

But the margin is eaten by two independent causes, and only one of them is fixable at deployment:

**(i) Command-induced deepening.** A lateral command applied during the drop takes ~3.4 cm off the
margin (406k: 0.2498 with no command -> 0.2162 with lateral 0.3), which is what turns 0% into 47%.
FIX: the settle gate — withhold the command/objective until the robot has settled (~20 steps). This
is a real deployment fix and `--command_start_step` already demonstrates it: from-settled,
landing_fail went to 0.000 for both 406k and 402k.

**(ii) The checkpoint's OWN dip depth at zero command.** 408k dips to 0.2280 with NO command at all
(6/64 = 9.4%), where 402k and 406k both reach ~0.250 and never fail. There is no command to
withhold, so **the settle gate does nothing for this**. FIX: artifact selection — landing-at-
zero-command is its own column per artifact — or a spawn-height change, which would also mask the
defect and is therefore not preferred.

Do not quote the settle gate as fixing landing generally. It fixes (i) only. An earlier note in this
doc said a few cm of spawn margin "would erase it for every artifact" — that conflated the two and
is wrong for (ii).

## Queue (reordered 2026-09-09, all from-settled, 64 envs, ~46 min each)
1. **BACKWARD axis on 402k and 408k** — promoted above everything else: 408k leads lateral 84.3 vs
   56.2 but its backward was 45.9 vs the keeper's 87.0 (old harness, valid there — no landing
   failures in either). If backward really is ~46, 408k cannot be the artifact whatever lateral does,
   so this one run can moot a whole branch rather than polish it.
2. 402k + lin W 0.25 — does the objective lift the keeper's 56% without breaking its 0/64 falls?
   Report E3's termination columns and lean SD (not just the mean) beside it; stability is what is
   at risk.
3. 408k + lin W 0.25 — push 84% toward 100% with its stability intact (only if 1 clears it).
4. A2-406k + lin W 0.25 — orientation penalty -10 as the roll-over lever. NOTE: `model_reward_abs_error`
   is confounded for A2 (trained at -10, env applies -1.0), so use behavioural columns only.
5. W 0.10 on 406k — lighter term: sustain with fewer falls?
6. lin 8 + yaw 8, W 0.25, gate 0.1 — combo commands need both.
7. Settle-gated objective as a planner/evaluator option — deployment fix for sub-case (i).

Prerequisite code, none GPU-bound: log SIGNED reward error in `play.py` diagnostics (the
under-command direction is still unmeasured, so no writeup may assert over- or under-prediction);
one eval at `--diagnostics_interval 1` on the keeper for true lag-1 error correlation.

## Cleanup item (do NOT apply mid-sweep): sweeper's `off` banner is misleading

`OBJECTIVE_FLAGS["off"]` passes only `--planner_velocity_objective_weight 0.0`, so the remaining seven
objective args INHERIT from the checkpoint and the startup banner prints e.g.

    [INFO] VelObjective weight=0.0 form=exp lin_w=0.0 yaw_w=8.0 yaw_gate=0.1 yaw_deadband=0.0

This is FUNCTIONALLY CORRECT -- `_latent_velocity_objective_reward` returns zeros immediately when
`planner_velocity_objective_weight <= 0.0`, before lin/yaw weights are read -- so the inherited
yaw_w=8.0 is inert and the objective really is off. But the banner reads as though the yaw term were
live, which is exactly how the Phase A contamination looked, and anyone auditing a log later would
reasonably flag it. Hand-built runs (JOB 1, E1-E4) pass all eight explicitly and print all zeros.

FIX: make `OBJECTIVE_FLAGS["off"]` pass all eight flags zeroed, so off-runs are unambiguous in the
log as well as in behaviour. Do not apply while a sweep is in flight -- the sweeper spawns play.py
fresh per eval, so an edit mid-sweep would split the run across two versions and a bug would kill the
remaining evals.

# Go2 x=0.4 reward tuning — results (2026-08-20 → 2026-08-23)

Autonomous grid sweep over the velocity-tracking reward at `command_x=0.4`, each trial a
short optimized resume run (`--replay_device auto`, `--planner_iterations 3`) from the
Stage-B x=0.4 walker (`model_205000`). Scored on mean `vel_x` + episode `len` over the
final ~1000 steps. Knobs co-varied per trial (via new CLI overrides in `train.py`:
`--reward_track_weight`, `--reward_track_std`, `--reward_alive_weight`):

| iter | track weight | track std | alive weight | vel_x (cmd 0.4) | len |
|-----:|-----:|-----:|-----:|-----:|----:|
| 1 | 2.5 | 0.25 | 0.35 | 0.198 | 942 |
| 2 | 3.0 | 0.22 | 0.30 | 0.163 | 999 |
| 3 | 3.5 | 0.20 | 0.25 | 0.244 | 976 |
| 4 | 4.0 | 0.18 | 0.20 | 0.266 | 924 |
| 5 | 5.0 | 0.15 | 0.15 | 0.264 | 973 |
| 6 | 6.5 | 0.13 | 0.12 | 0.262 | 972 |
| **7** | **8.0** | **0.11** | **0.10** | **0.309** | **951** |  ← BEST |
| 8 | 10.0 | 0.09 | 0.08 | 0.272 | 533 |
| 9 | 13.0 | 0.07 | 0.06 | 0.146 | 897 |

## Best setting
`track_lin_vel_xy_exp.weight = 8.0`, `track_lin_vel_xy_exp.std = 0.11`, `alive.weight = 0.10`
→ **vel_x ≈ 0.309 m/s, stable (len ≈ 951)** at command_x=0.4.

Checkpoint: `logs/mbrl/go2_walk_2026-08-22_06-19-42/checkpoints/model_best.pt`
(preserved copy: `logs/mbrl/best_walker/best_x0p4_reward_w8_v0p309.pt`).

## Verdict
`PLATEAU` — reward-weight tuning ceiling **~0.31 m/s** (clear peak at weight 8.0; velocity
*declines* past it — 0.272 at weight 10, 0.146 at weight 13). Untuned baseline was ~0.16 m/s,
so tuning gained +0.15, but tops out at **~77% of the 0.4 command / 86% of the 0.36 target**.

Reaching a true 0.4 needs **structural** levers, not more reward weight:
- raise the **action scale** (0.25 → ~0.5) — likely physical speed cap;
- add an **explicit forward-velocity reward** (non-saturating), since `track_lin_vel_xy_exp`
  is `exp(-err²/std²)` and flattens near the command.

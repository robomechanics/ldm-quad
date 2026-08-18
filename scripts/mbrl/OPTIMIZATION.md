# MBRL training-speed notes

Benchmarked on the RTX 5070 Ti (16 GB) with the `Flat-Unitree-Go2-train-v0`
latent TD-MPC setup. Key finding: the run is **compute/latency-bound on the
gradient-update and planning steps**, not memory-bound — even at 1024 envs the
job touches < 5 GB of VRAM. "Using more VRAM" does not make it faster; reducing
the per-iteration compute does.

## What each knob costs (measured, same config as run.sh)

Per training iteration the time splits roughly half planning / half updates.

| Lever | Effect on speed | Effect on results | Default |
|-------|-----------------|-------------------|---------|
| `--replay_device auto` (buffer in VRAM) | ~60x faster sampling; removes CPU->GPU copy + swap thrash | **None** — identical math | on |
| `--planner_iterations 6 -> 3` | planning ~1.9x cheaper | Low. Only affects training-time data collection; deploy/eval can still plan at 6+ | 3 |
| `--candidates 512 -> 256` | planning ~2x cheaper | Moderate — halves exploration breadth over the 96-dim action search | 512 |
| `--utd 0.25 -> 0.125` | update block ~2x cheaper | Fewer grad steps/transition — undertraining risk | 0.25 |
| `--num_envs` up | ~1.27x throughput at 16x envs, and does NOT shorten a fixed `--train_steps` run (train_steps counts loop iterations) | Changes data/learning; not a free win | 64 |

Planner scaling (num_envs=64): 6x512 = 2569 ms, 3x512 = 1324 ms,
3x256 = 611 ms, 2x256 = 380 ms.

## Recommended tiers

1. **Free (no quality cost), always on:** `--replay_device auto`. Also make sure
   the box is not swapping (`free -h`); the old CPU buffer + a swapping host was a
   hidden stall.
2. **Low risk, on by default here:** `--planner_iterations 3`.
3. **Higher speed, validate first:** `CANDIDATES=256` and/or `UTD=0.125`. Run the
   A/B below before trusting them.

## A/B a risky knob from a checkpoint

Run two short (~2-3k step) resumes from the same checkpoint, one baseline and one
with the aggressive knob, and compare `eval_tracking` / return in the metrics.csv
or W&B. Example:

```bash
# baseline planner
PLANNER_ITERATIONS=6 CANDIDATES=512 UTD=0.25 BASE_CKPT=<ckpt> bash scripts/mbrl/run.sh
# aggressive planner
PLANNER_ITERATIONS=3 CANDIDATES=256 UTD=0.125 BASE_CKPT=<ckpt> bash scripts/mbrl/run.sh
```

All knobs are env-overridable in `run.sh`, so no file edits are needed to sweep.

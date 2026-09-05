#!/usr/bin/env python
"""Localize WHICH component owns a behaviour: the policy head, or the world model / Q.

WHY
---
When a capability oscillates or regresses, the aggregate metrics cannot say which part of
the agent is responsible, and the answer changes the cost of the fix by an order of
magnitude (a policy-head change resumes warm in ~a day; a world-model change is
architectural and starts from scratch).

Replays saved checkpoints under three action-selection modes -- no training:

    M1  full planner                      planner + world model + pi   (normal operation)
    M2  --num_pi_trajs 0                  planner + world model, pi REMOVED from candidates
    M3  --policy_only                     pi alone, no planning

Reading the spread of each mode across checkpoints that span one oscillation:
  * M2 stable while M1 swings   -> the POLICY HEAD owns it. Cheap: widen/condition pi and
                                   resume warm.
  * M2 swings too               -> the WORLD MODEL / Q owns it (that is all MPPI has left).
                                   Expensive: structural change, from scratch.
  * M3 flat-bad everywhere      -> pi never learned the skill; it cannot be what oscillates,
                                   and the capability is supplied entirely by planning.

First use (2026-09-04, T2 checkpoints 382k/388k/390k/394k, lat_yp0p3 + inplace_yaw):
M2 spread 54.8/79.3 pts vs M1 41.4/62.2 -- removing pi did NOT stabilise anything, and M3
yaw was ~0% at every checkpoint. Verdict: world model / Q. See
logs/mbrl/history/stageT2_warmbuffer_TRADING_CONFIRMED/RESULT.md

  python scripts/mbrl/localize_trading.py \
      --ckpt_dir logs/mbrl/history/<run>/checkpoints \
      --steps 382000 388000 390000 394000 \
      --conditions lat_yp0p3 inplace_yaw

NOTE: point --ckpt_dir at a STABLE path. play.py exits 0 when the checkpoint is missing, so
a moved directory yields silently-empty runs; this script guards against that explicitly.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as st
import subprocess

PY = os.environ.get("MBRL_PY", "/home/rml2/anaconda3/envs/isaaclab/bin/python")

# name -> (x, y, yaw, velocity key, commanded value on the probed axis)
CONDITIONS = {
    "ref_x0p4": (0.4, 0.0, 0.0, "velocity_x_mean", 0.4),
    "fast_x0p5": (0.5, 0.0, 0.0, "velocity_x_mean", 0.5),
    "back_xm0p3": (-0.3, 0.0, 0.0, "velocity_x_mean", -0.3),
    "lat_yp0p3": (0.0, 0.3, 0.0, "velocity_y_mean", 0.3),
    "lat_ym0p3": (0.0, -0.3, 0.0, "velocity_y_mean", -0.3),
    "inplace_yaw": (0.0, 0.0, 0.8, "velocity_yaw_mean", 0.8),
}
MODES = {"M1": [], "M2": ["--num_pi_trajs", "0"], "M3": ["--policy_only"]}
EPISODE_TIMEOUT = 1000


def run_one(ckpt: str, out_dir: str, cond: str, mode: str, episodes: int, max_steps: int,
            action_scale: float) -> tuple[float | None, float]:
    x, y, w, key, cmd = CONDITIONS[cond]
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, "done")):
        args = [
            PY, "-u", "scripts/mbrl/play.py", "--checkpoint", ckpt,
            "--headless", "--num_envs", "1",
            "--num_episodes", str(episodes), "--max_steps", str(max_steps),
            "--command_x", str(x), "--command_y", str(y), "--command_yaw", str(w),
            "--action_scale", str(action_scale),
            "--diagnostics", "--diagnostics_dir", out_dir, "--diagnostics_interval", "5",
        ] + MODES[mode]
        with open(os.path.join(out_dir, "console.log"), "w") as log:
            rc = subprocess.call(args, stdout=log, stderr=subprocess.STDOUT)
        mf = os.path.join(out_dir, "metrics.csv")
        # play.py can exit 0 having produced nothing (e.g. missing checkpoint) -- check output
        if rc != 0 or not os.path.exists(mf) or os.path.getsize(mf) == 0:
            print(f"[localize] FAILED {os.path.basename(ckpt)} {cond} {mode} rc={rc}", flush=True)
            return None, 0.0
        open(os.path.join(out_dir, "done"), "w").close()
    rows = list(csv.DictReader(open(os.path.join(out_dir, "metrics.csv"))))
    if not rows:
        return None, 0.0
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    achieved = st.mean(vals) if vals else 0.0
    m = re.search(r"mean_length=([\d.]+)", open(os.path.join(out_dir, "console.log")).read())
    return achieved / cmd * 100.0, (float(m.group(1)) if m else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Directory holding model_<step>.pt files.")
    ap.add_argument("--steps", nargs="+", type=int, required=True, help="Checkpoint steps spanning one oscillation.")
    ap.add_argument("--conditions", nargs="+", default=["lat_yp0p3", "inplace_yaw"], choices=sorted(CONDITIONS))
    ap.add_argument("--modes", nargs="+", default=["M1", "M2", "M3"], choices=sorted(MODES))
    ap.add_argument("--out", default="logs/mbrl/localize")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--action_scale", type=float, default=0.40)
    a = ap.parse_args()

    missing = [s for s in a.steps if not os.path.exists(os.path.join(a.ckpt_dir, f"model_{s}.pt"))]
    if missing:
        raise SystemExit(f"[localize] ABORT: missing checkpoints for steps {missing} in {a.ckpt_dir}")

    results: dict[tuple[str, str, int], tuple[float | None, float]] = {}
    for s in a.steps:
        ckpt = os.path.join(a.ckpt_dir, f"model_{s}.pt")
        for cond in a.conditions:
            for mode in a.modes:
                d = os.path.join(a.out, f"{s}__{cond}__{mode}")
                print(f"[localize] {s} {cond} {mode}", flush=True)
                results[(cond, mode, s)] = run_one(ckpt, d, cond, mode, a.episodes, a.max_steps, a.action_scale)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "localize.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["condition", "mode", "env_steps", "pct_of_cmd", "mean_length", "falls"])
        for (cond, mode, s), (p, ln) in sorted(results.items()):
            wr.writerow([cond, mode, s, "" if p is None else f"{p:.1f}", f"{ln:.0f}",
                         "YES" if 0 < ln < EPISODE_TIMEOUT - 1 else "no"])

    for cond in a.conditions:
        print(f"\n=== {cond} ===")
        print(f"{'step':>8} " + " ".join(f"{m:>16}" for m in a.modes))
        for s in a.steps:
            cells = []
            for m in a.modes:
                p, ln = results[(cond, m, s)]
                cells.append(f"{'--':>16}" if p is None else
                             f"{p:>9.1f}% {'FALL' if ln < EPISODE_TIMEOUT - 1 else '  ok':>5}")
            print(f"{s:>8} " + " ".join(cells))
        for m in a.modes:
            v = [results[(cond, m, s)][0] for s in a.steps if results[(cond, m, s)][0] is not None]
            if len(v) > 1:
                print(f"   {m} spread (max-min): {max(v) - min(v):.1f} pts")
    print(f"\n[localize] wrote {os.path.join(a.out, 'localize.csv')}")


if __name__ == "__main__":
    main()

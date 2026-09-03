#!/usr/bin/env python
"""Orchestration ADR: expand/contract the velocity-command box by measured performance.

WHAT IT DOES
------------
Repeats: train a chunk -> evaluate each of the 6 command-box boundaries at FIXED commands
-> widen boundaries that track well, narrow ones that fail -> relaunch. The curriculum
becomes a logged OUTPUT (adr_state.json / adr_history.csv) instead of hand-picked ranges.

WHY ORCHESTRATION AND NOT AN IN-LOOP HOOK
-----------------------------------------
Nothing in the training path changes. run.sh already accepts every boundary as an env
override, so this only sets env vars and reads the sweep CSV -- both proven code paths.
train.py's own --online_eval hangs (verified 2026-09-03), so in-loop eval is not an option
anyway.

Widening the command range does NOT invalidate the replay buffer: rewards are unchanged, the
old data is still correctly labelled, it just does not yet cover the new region. So the buffer
resumes across rounds (--auto_resume_replay, run.sh default) and there is no cold-refill cost.
This is why a REWARD change (Stage R) needs a fresh buffer but a RANGE change does not.

  python scripts/mbrl/adr_loop.py --resume_from <ckpt> --start_steps 364000 --rounds 20
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import time

REPO = "/home/rml2/Documents/thomas_practice/ldm-quad"
PY = os.environ.get("MBRL_PY", "/home/rml2/anaconda3/envs/isaaclab/bin/python")

# Each boundary is probed on its own axis with the other two commanded to zero, so the
# measurement is attributable to that boundary alone.
#   key            env var    axis  sign  step   hard limit (physical/benchmark ceiling)
BOUNDARIES = [
    ("x_max", "X_MAX", 0, +1, 0.05, 0.65),
    ("x_min", "X_MIN", 0, -1, 0.05, -0.45),
    ("y_max", "Y_MAX", 1, +1, 0.05, 0.40),
    ("y_min", "Y_MIN", 1, -1, 0.05, -0.40),
    ("yaw_max", "YAW_MAX", 2, +1, 0.10, 0.90),
    ("yaw_min", "YAW_MIN", 2, -1, 0.10, -0.90),
]
VEL_KEYS = ("velocity_x_mean", "velocity_y_mean", "velocity_yaw_mean")

# Calibrated against measured numbers: ~0.08-0.10 abs error where conditions work,
# ~0.19 where they fail. Falls are treated as an unconditional contract signal.
EXPAND_ERR = 0.12
CONTRACT_ERR = 0.20
EPISODE_TIMEOUT = 1000


def newest_run_dir(after: float) -> str | None:
    dirs = [d for d in glob.glob(os.path.join(REPO, "logs/mbrl/go2_walk_*")) if os.path.isdir(d)]
    dirs = [d for d in dirs if os.path.getmtime(d) >= after - 5]
    return max(dirs, key=os.path.getmtime) if dirs else None


def probe(ckpt: str, out_dir: str, axis: int, value: float, episodes: int, max_steps: int) -> tuple[float, float]:
    """Run one fixed command; return (abs error on that axis, mean episode length)."""
    cmd_vec = [0.0, 0.0, 0.0]
    cmd_vec[axis] = value
    os.makedirs(out_dir, exist_ok=True)
    args = [
        PY, "-u", "scripts/mbrl/play.py", "--checkpoint", ckpt,
        "--headless", "--num_envs", "1",
        "--num_episodes", str(episodes), "--max_steps", str(max_steps),
        "--command_x", str(cmd_vec[0]), "--command_y", str(cmd_vec[1]), "--command_yaw", str(cmd_vec[2]),
        "--action_scale", "0.40",
        "--diagnostics", "--diagnostics_dir", out_dir, "--diagnostics_interval", "5",
    ]
    with open(os.path.join(out_dir, "console.log"), "w") as log:
        if subprocess.call(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT) != 0:
            return float("nan"), 0.0
    mf = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(mf):
        return float("nan"), 0.0
    rows = list(csv.DictReader(open(mf)))
    if not rows:
        return float("nan"), 0.0
    vals = [float(r[VEL_KEYS[axis]]) for r in rows if r.get(VEL_KEYS[axis]) not in (None, "")]
    achieved = sum(vals) / len(vals) if vals else 0.0
    m = re.search(r"mean_length=([\d.]+)", open(os.path.join(out_dir, "console.log")).read())
    return abs(achieved - value), (float(m.group(1)) if m else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume_from", required=True, help="Checkpoint to start round 1 from.")
    ap.add_argument("--start_steps", type=int, required=True, help="env_steps of that checkpoint.")
    ap.add_argument("--chunk", type=int, default=2500, help="Training steps per round.")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--state_dir", default=os.path.join(REPO, "logs/mbrl/adr"))
    # Start narrow and let the loop discover the frontier.
    ap.add_argument("--x_min", type=float, default=-0.10)
    ap.add_argument("--x_max", type=float, default=0.40)
    ap.add_argument("--y_min", type=float, default=-0.10)
    ap.add_argument("--y_max", type=float, default=0.10)
    ap.add_argument("--yaw_min", type=float, default=-0.20)
    ap.add_argument("--yaw_max", type=float, default=0.20)
    a = ap.parse_args()

    os.makedirs(a.state_dir, exist_ok=True)
    state_path = os.path.join(a.state_dir, "adr_state.json")
    hist_path = os.path.join(a.state_dir, "adr_history.csv")

    if os.path.exists(state_path):
        state = json.load(open(state_path))
        print(f"[adr] resuming at round {state['round']}, steps {state['steps']}", flush=True)
    else:
        state = {
            "round": 0, "steps": a.start_steps, "resume": a.resume_from,
            "bounds": {"x_min": a.x_min, "x_max": a.x_max, "y_min": a.y_min,
                       "y_max": a.y_max, "yaw_min": a.yaw_min, "yaw_max": a.yaw_max},
        }

    for _ in range(a.rounds):
        state["round"] += 1
        rnd = state["round"]
        b = state["bounds"]
        target = state["steps"] + a.chunk
        print(f"\n[adr] === round {rnd}: train -> {target}  box x[{b['x_min']:+.2f},{b['x_max']:+.2f}] "
              f"y[{b['y_min']:+.2f},{b['y_max']:+.2f}] yaw[{b['yaw_min']:+.2f},{b['yaw_max']:+.2f}]", flush=True)

        env = dict(os.environ)
        env.update({
            "X_MIN": str(b["x_min"]), "X_MAX": str(b["x_max"]),
            "Y_MIN": str(b["y_min"]), "Y_MAX": str(b["y_max"]),
            "YAW_MIN": str(b["yaw_min"]), "YAW_MAX": str(b["yaw_max"]),
            "RESUME": state["resume"], "TRAIN_STEPS": str(target),
            "SEED": str(a.seed), "WANDB_NAME": f"adr_r{rnd}_s{a.seed}",
            # range changes keep the buffer valid (see module docstring) -> reuse it
            "REPLAY_RESUME": "1",
        })
        t0 = time.time()
        rc = subprocess.call(["bash", "scripts/mbrl/run.sh", "train"], cwd=REPO, env=env)
        run_dir = newest_run_dir(t0)
        if rc != 0 or run_dir is None:
            print(f"[adr] training failed rc={rc}; stopping", flush=True)
            return
        ckpt = os.path.join(run_dir, "checkpoints", "model_final.pt")
        if not os.path.exists(ckpt):
            print(f"[adr] no model_final.pt in {run_dir}; stopping", flush=True)
            return
        state["steps"] = target
        state["resume"] = ckpt

        print(f"[adr] round {rnd}: probing 6 boundaries on {os.path.basename(run_dir)}", flush=True)
        rows = []
        for key, envvar, axis, sign, step, limit in BOUNDARIES:
            val = b[key]
            err, ln = probe(ckpt, os.path.join(a.state_dir, f"r{rnd}", key), axis, val, a.episodes, a.max_steps)
            fell = ln < EPISODE_TIMEOUT - 1
            if err != err:  # NaN -> probe failed; leave the boundary alone
                action, new = "probe_failed", val
            elif fell or err >= CONTRACT_ERR:
                action, new = "contract", val - sign * step
            elif err <= EXPAND_ERR:
                new = val + sign * step
                new = min(new, limit) if sign > 0 else max(new, limit)
                action = "hold_at_limit" if new == val else "expand"
            else:
                action, new = "hold", val
            b[key] = round(new, 3)
            rows.append({"round": rnd, "env_steps": target, "boundary": key, "value_before": val,
                         "abs_err": f"{err:.4f}", "mean_length": f"{ln:.0f}",
                         "action": action, "value_after": b[key]})
            print(f"[adr]   {key:>8} @{val:+.2f}  err={err:.3f} len={ln:.0f}  -> {action} -> {b[key]:+.2f}", flush=True)

        new_file = not os.path.exists(hist_path)
        with open(hist_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new_file:
                w.writeheader()
            w.writerows(rows)
        json.dump(state, open(state_path, "w"), indent=2)

    print("[adr] all rounds complete", flush=True)


if __name__ == "__main__":
    main()

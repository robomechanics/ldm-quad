#!/usr/bin/env python
"""Watch a training run's checkpoints and sweep each new one at FIXED commands.

WHY THIS EXISTS
---------------
Aggregate training metrics average over the wander command box and hide per-axis
collapse. On 2026-09-02 Stage W posted its best-ever stable_tracking (0.443) and
len100 (748) in the same window that in-place turning fell 86% -> 54% and started
falling over. Stage M did the same thing on the lateral axis. Both were only visible
under FIXED commands.

This runs the fixed-command sweep against each checkpoint as it is written, in a
separate process, and appends one row per (checkpoint, condition) to
<run_dir>/checkpoint_sweep.csv. It never touches the training process.

  python scripts/mbrl/sweep_checkpoints.py --run_dir logs/mbrl/go2_walk_... [--once]

NOTE: train.py's --online_eval / run_heldout_eval path HANGS (verified 2026-09-03,
with and without --fixed_command_eval; never used in any recorded run). Do not use
in-training eval until that is fixed -- sweep checkpoints from outside instead.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time

PY = os.environ.get("MBRL_PY", "/home/rml2/anaconda3/envs/isaaclab/bin/python")

# The three conditions that caught both regressions, plus a forward control.
# (name, x, y, yaw, axis index into [x, y, yaw])
CONDITIONS = [
    ("ref_x0p4", 0.4, 0.0, 0.0, 0),
    ("back_xm0p3", -0.3, 0.0, 0.0, 0),
    ("lat_yp0p3", 0.0, 0.3, 0.0, 1),
    ("inplace_yaw", 0.0, 0.0, 0.8, 2),
]
VEL_KEYS = ("velocity_x_mean", "velocity_y_mean", "velocity_yaw_mean")
FIELDS = ["checkpoint", "env_steps", "condition", "commanded", "achieved", "pct_of_cmd", "mean_length", "falls"]


def step_of(name: str) -> int:
    m = re.search(r"model_(\d+)\.pt$", name)
    return int(m.group(1)) if m else -1


def sweep_one(ckpt: str, out_dir: str, action_scale: float, episodes: int, max_steps: int) -> list[dict]:
    rows = []
    for name, cx, cy, cw, axis in CONDITIONS:
        d = os.path.join(out_dir, f"{os.path.basename(ckpt)[:-3]}__{name}")
        os.makedirs(d, exist_ok=True)
        cmd = [
            PY, "-u", "scripts/mbrl/play.py", "--checkpoint", ckpt,
            "--headless", "--num_envs", "1",
            "--num_episodes", str(episodes), "--max_steps", str(max_steps),
            "--command_x", str(cx), "--command_y", str(cy), "--command_yaw", str(cw),
            "--action_scale", str(action_scale),
            "--diagnostics", "--diagnostics_dir", d, "--diagnostics_interval", "5",
        ]
        with open(os.path.join(d, "console.log"), "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f"[ckpt-sweep] FAILED {os.path.basename(ckpt)} {name} rc={rc}", flush=True)
            continue
        mfile = os.path.join(d, "metrics.csv")
        if not os.path.exists(mfile):
            continue
        mrows = list(csv.DictReader(open(mfile)))
        if not mrows:
            continue
        vals = [float(r[VEL_KEYS[axis]]) for r in mrows if r.get(VEL_KEYS[axis]) not in (None, "")]
        achieved = sum(vals) / len(vals) if vals else 0.0
        ln = 0.0
        m = re.search(r"mean_length=([\d.]+)", open(os.path.join(d, "console.log")).read())
        if m:
            ln = float(m.group(1))
        commanded = (cx, cy, cw)[axis]
        rows.append({
            "checkpoint": os.path.basename(ckpt),
            "env_steps": step_of(ckpt),
            "condition": name,
            "commanded": f"{commanded:.3f}",
            "achieved": f"{achieved:.4f}",
            "pct_of_cmd": f"{(achieved / commanded * 100):.1f}" if commanded else "",
            "mean_length": f"{ln:.0f}",
            # episode timeout is 1000; anything short means the robot fell
            "falls": "YES" if ln < 999 else "no",
        })
    return rows


def init_wandb(run_dir: str, project: str, name: str | None):
    """Separate run in the same project, x-axis pinned to env_steps.

    Deliberately NOT resuming the training run: two live writers to one wandb run is
    unreliable, and train.py already syncs tensorboard into it (which hijacks the implicit
    step -- hence its "Step cannot be set when using tensorboard syncing" warning). A sibling
    run with an explicit step_metric overlays cleanly on the same x-axis in the UI.
    """
    try:
        import wandb
    except ImportError:
        print("[ckpt-sweep] wandb not installed; continuing without it", flush=True)
        return None
    try:
        run = wandb.init(
            project=project,
            name=name or (os.path.basename(os.path.normpath(run_dir)) + "_fixedeval"),
            job_type="fixed_command_eval",
            reinit=True,
        )
        wandb.define_metric("env_steps")
        wandb.define_metric("FixedEval/*", step_metric="env_steps")
        print(f"[ckpt-sweep] wandb: {run.url}", flush=True)
        return run
    except Exception as exc:
        print(f"[ckpt-sweep] wandb init failed ({exc}); continuing without it", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Training run directory (contains checkpoints/).")
    ap.add_argument("--action_scale", type=float, default=0.40)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--poll_seconds", type=int, default=300)
    ap.add_argument("--once", action="store_true", help="Sweep everything present, then exit.")
    ap.add_argument("--wandb", action="store_true", help="Also log per-condition results to Weights & Biases.")
    ap.add_argument("--wandb_project", default="ldm-quad-mbrl")
    ap.add_argument("--wandb_name", default=None, help="Defaults to <run_dir basename>_fixedeval.")
    a = ap.parse_args()

    ck_dir = os.path.join(a.run_dir, "checkpoints")
    out_dir = os.path.join(a.run_dir, "checkpoint_sweep")
    csv_path = os.path.join(a.run_dir, "checkpoint_sweep.csv")
    os.makedirs(out_dir, exist_ok=True)

    wb = init_wandb(a.run_dir, a.wandb_project, a.wandb_name) if a.wandb else None

    seen: set[str] = set()
    if os.path.exists(csv_path):
        seen = {r["checkpoint"] for r in csv.DictReader(open(csv_path))}
        print(f"[ckpt-sweep] resuming, {len(seen)} checkpoint(s) already swept", flush=True)

    while True:
        pending = sorted(
            (f for f in os.listdir(ck_dir) if re.match(r"model_\d+\.pt$", f) and f not in seen),
            key=step_of,
        ) if os.path.isdir(ck_dir) else []
        for f in pending:
            print(f"[ckpt-sweep] sweeping {f} at {time.strftime('%H:%M:%S')}", flush=True)
            rows = sweep_one(os.path.join(ck_dir, f), out_dir, a.action_scale, a.episodes, a.max_steps)
            if rows:
                new = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerows(rows)
                print("[ckpt-sweep] " + "  ".join(
                    f"{r['condition']}={r['pct_of_cmd']}%/{'FALL' if r['falls'] == 'YES' else 'ok'}" for r in rows
                ), flush=True)
                if wb is not None:
                    payload = {"env_steps": rows[0]["env_steps"]}
                    for r in rows:
                        c = r["condition"]
                        payload[f"FixedEval/{c}_pct"] = float(r["pct_of_cmd"]) if r["pct_of_cmd"] else 0.0
                        payload[f"FixedEval/{c}_len"] = float(r["mean_length"])
                        payload[f"FixedEval/{c}_falls"] = 1.0 if r["falls"] == "YES" else 0.0
                    # worst-axis summary: the metric save_best_metric should have been using
                    pcts = [float(r["pct_of_cmd"]) for r in rows if r["pct_of_cmd"]]
                    payload["FixedEval/worst_axis_pct"] = min(pcts) if pcts else 0.0
                    payload["FixedEval/n_falls"] = sum(1 for r in rows if r["falls"] == "YES")
                    wb.log(payload)
            seen.add(f)
        if a.once:
            break
        time.sleep(a.poll_seconds)


if __name__ == "__main__":
    main()

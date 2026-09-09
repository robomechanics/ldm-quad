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
    ("combo", 0.3, 0.2, 0.5, 2),
]
VEL_KEYS = ("velocity_x_mean", "velocity_y_mean", "velocity_yaw_mean")
EP_VEL_KEYS = ("vx", "vy", "vyaw")
EP_VEL_POST_KEYS = ("vx_post", "vy_post", "vyaw_post")
FIELDS = ["checkpoint", "env_steps", "condition", "objective", "mode", "commanded",
          "achieved", "achieved_sem", "pct_of_cmd", "pct_sem", "n_eps",
          "landing_fail_rate", "n_landed", "fall_rate_post", "mean_length_post",
          "mean_length", "fall_rate", "falls"]

# LANDING IS NOT LOCOMOTION (2026-09-08)
# -------------------------------------
# The robot spawns at base_height ~0.398 (nominal stance ~0.32) and free-falls; the dip at step
# ~10 lands within centimetres of the terminations.base_height threshold of 0.20. Measured on
# lateral: 402k bottoms at 0.2549 and survives, 406k bottoms at 0.2162 and dies. So ~47% of envs
# at 406k terminate at steps 12-15 having never walked. That is a LANDING failure, and averaging
# it into a velocity number makes the number mostly about spawn bounce: 406k lateral at N=1 was
# episodes {13, 14, 765}, reported as the single figure "mean_length 265".
# `achieved` is therefore the post-settle velocity over episodes that SURVIVED landing, and the
# landing result is reported separately instead of being folded in.

# WHY N=64 ENVS AND PER-EPISODE AVERAGING (2026-09-08)
# ---------------------------------------------------
# This swept --num_envs 1 --episodes 3, so every cell was 3 episodes of ONE robot, and
# `achieved` was a time-average over the whole rollout -- including post-reset acceleration
# and post-fall ~zero velocity. One early fall therefore depressed the velocity twice.
# Measured consequence: across 5 consecutive checkpoints of a SINGLE UNCHANGED config
# (398k-406k) mean-of-4-axes spanned 13.3 points and worst-axis spanned 47 points, while the
# spread across all EIGHT different interventions was 13.5 and 30.5. The noise band of doing
# nothing was as wide as the entire effect range being ranked -- every worst-axis verdict was
# unresolvable. Lateral's apparent collapse (72.3->56.2->31.1->22.9) tracked its mean_length
# (562->1000->766->265) almost exactly: it was largely a fall-rate reading at N=3.
# Fix: 64 envs, average velocity WITHIN an episode, then take the FIRST episode of every env
# (one unbiased sample per env -- play.py stops at num_episodes, so counting every completed
# episode would over-weight fast-failing envs that get through two). Report SEM so the noise
# is visible in the CSV, and define a fall as `terminated` rather than `length < 999`.


# play.py inherits objective settings from the CHECKPOINT when not given on the CLI, so a
# checkpoint trained with weight>0 would otherwise be swept with the objective silently ON.
# (That happened to every Phase A/A2 sweep: weight 0.5 + form exp inherited, but lin_weight
# and yaw_gate came from CLI defaults 8.0/0.0 -- the exact 15b config that rolls lateral over.
# Tell: planner_candidate_return_best ~16 with the objective off vs ~70 with it on.)
OBJECTIVE_FLAGS = {
    "off": ["--planner_velocity_objective_weight", "0.0"],
    "gated": [
        "--planner_velocity_objective_weight", "0.5",
        "--planner_velocity_objective_form", "exp",
        "--planner_velocity_objective_lin_weight", "0.0",
        "--planner_velocity_objective_yaw_weight", "8.0",
        "--planner_velocity_objective_yaw_gate", "0.1",
        "--planner_velocity_objective_yaw_deadband", "0.0",
    ],
}


def step_of(name: str) -> int:
    m = re.search(r"model_(\d+)\.pt$", name)
    if m:
        return int(m.group(1))
    # kept artifacts embed their step as e.g. phaseA_gatedyaw_402k_backward.pt
    m = re.search(r"_(\d+)k(?:_|\.)", name)
    return int(m.group(1)) * 1000 if m else -1


def mode_name(command_start_step: int) -> str:
    """Two initiation modes, both of which belong in the table.

    from_spawn   -- the command is live while the robot is still settling out of the ~0.398 m spawn
                    drop toward the 0.20 m termination cut. A lateral command deepens that dip by
                    3.4 cm and killed 47% of envs at 406k, so this mode is dominated by landing.
    from_settled -- the command arrives after the robot has settled, which is what a deployed
                    controller faces for every command after the first.

    Reporting only from_settled is the flattering version and hides a real controller defect;
    reporting only from_spawn conflates landing with locomotion. Hence a mode COLUMN, not a flag
    that silently changes what the numbers mean.
    """
    return "from_spawn" if command_start_step <= 0 else f"from_settled{command_start_step}"


def sweep_one(ckpt: str, out_dir: str, action_scale: float, episodes: int, max_steps: int,
              objective: str = "off", num_envs: int = 64, settle_steps: int = 20,
              command_start_step: int = 0) -> list[dict]:
    """One fixed-command eval per condition. `num_envs` is also the intended sample count."""
    rows = []
    for name, cx, cy, cw, axis in CONDITIONS:
        mode = mode_name(command_start_step)
        d = os.path.join(out_dir, f"{os.path.basename(ckpt)[:-3]}__{name}__{objective}__{mode}")
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(ckpt):
            print(f"[ckpt-sweep] ABORT: missing checkpoint {ckpt}", flush=True)
            raise SystemExit(1)
        cmd = [
            PY, "-u", "scripts/mbrl/play.py", "--checkpoint", ckpt,
            "--headless", "--num_envs", str(num_envs),
            "--num_episodes", str(episodes), "--max_steps", str(max_steps),
            "--command_x", str(cx), "--command_y", str(cy), "--command_yaw", str(cw),
            "--settle_steps", str(settle_steps),
            "--command_start_step", str(command_start_step),
            "--action_scale", str(action_scale),
            "--diagnostics", "--diagnostics_dir", d, "--diagnostics_interval", "5",
        ] + OBJECTIVE_FLAGS[objective]
        with open(os.path.join(d, "console.log"), "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f"[ckpt-sweep] FAILED {os.path.basename(ckpt)} {name} rc={rc}", flush=True)
            continue
        epfile = os.path.join(d, "episodes.csv")
        if not os.path.exists(epfile):
            # Never silently fall back to the old whole-rollout time-average: it is a
            # different (and much noisier) statistic and would corrupt the column.
            print(f"[ckpt-sweep] NO episodes.csv for {os.path.basename(ckpt)} {name} -- "
                  "stale play.py or the eval died before any episode finished; skipping row",
                  flush=True)
            continue
        first: dict[int, dict] = {}
        for r in csv.DictReader(open(epfile)):
            first.setdefault(int(r["env"]), r)
        samples = list(first.values())
        if not samples:
            continue
        if EP_VEL_POST_KEYS[0] not in samples[0]:
            print(f"[ckpt-sweep] episodes.csv for {os.path.basename(ckpt)} {name} predates the "
                  "landing/locomotion split (no vx_post); skipping rather than mixing statistics",
                  flush=True)
            continue
        n = len(samples)
        if n < num_envs:
            # Every env must finish a first episode, or the survivors -- the envs that were still
            # walking when the rollout ended -- are missing from the denominator and every rate is
            # biased toward failure. With max_steps == the 1000-step timeout this cannot happen:
            # survivors are truncated at 1000, which counts as a completed episode.
            print(f"[ckpt-sweep] WARNING {os.path.basename(ckpt)} {name}: only {n}/{num_envs} envs "
                  f"finished a first episode within max_steps={max_steps}; rates are biased toward "
                  "failure because still-walking envs are excluded", flush=True)
        lens = [float(r["length"]) for r in samples]
        # whole-episode view, kept for continuity
        fall_rate = sum(int(r["terminated"]) for r in samples) / n
        ln = sum(lens) / n
        # landing vs locomotion
        landed = [r for r in samples if float(r["length"]) >= settle_steps]
        landing_fail_rate = sum(
            1 for r in samples if int(r["terminated"]) and float(r["length"]) < settle_steps
        ) / n
        post = [r for r in landed if float(r["n_post"]) > 0.0]
        if post:
            vals = [float(r[EP_VEL_POST_KEYS[axis]]) for r in post]
            m = len(vals)
            achieved = sum(vals) / m
            if m > 1:
                var = sum((v - achieved) ** 2 for v in vals) / (m - 1)
                sem = (var / m) ** 0.5
            else:
                sem = 0.0
        else:
            # nothing survived the spawn drop: there is no locomotion to report
            achieved, sem = float("nan"), float("nan")
        fall_rate_post = (sum(int(r["terminated"]) for r in landed) / len(landed)) if landed else float("nan")
        ln_post = (sum(float(r["length"]) for r in landed) / len(landed)) if landed else float("nan")
        commanded = (cx, cy, cw)[axis]
        rows.append({
            "checkpoint": os.path.basename(ckpt),
            "env_steps": step_of(ckpt),
            "condition": name,
            "objective": objective,
            "mode": mode,
            "commanded": f"{commanded:.3f}",
            "achieved": f"{achieved:.4f}",
            "achieved_sem": f"{sem:.4f}",
            "pct_of_cmd": f"{(achieved / commanded * 100):.1f}" if commanded else "",
            "pct_sem": f"{(sem / abs(commanded) * 100):.1f}" if commanded else "",
            "n_eps": str(n),
            "landing_fail_rate": f"{landing_fail_rate:.4f}",
            "n_landed": str(len(landed)),
            "fall_rate_post": f"{fall_rate_post:.4f}",
            "mean_length_post": f"{ln_post:.0f}" if ln_post == ln_post else "",
            "mean_length": f"{ln:.0f}",
            "fall_rate": f"{fall_rate:.4f}",
            # kept for continuity of the console line; fall_rate is the real number now
            "falls": "YES" if fall_rate > 0 else "no",
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
    ap.add_argument("--command_start_step", type=int, default=0,
                    help="0 = from_spawn (command live during the spawn settle). >0 holds the "
                         "command at zero for that many steps of EVERY episode first = from_settled, "
                         "what a deployed controller actually faces. Recorded as the `mode` column.")
    ap.add_argument("--settle_steps", type=int, default=20,
                    help="Episodes terminating inside this many steps are LANDING failures (spawn "
                         "drop from ~0.398 m toward the 0.20 m termination), reported separately; "
                         "tracking is averaged over steps beyond it.")
    ap.add_argument("--num_envs", type=int, default=64,
                    help="Parallel envs per eval. Each contributes ONE episode to the statistic.")
    ap.add_argument("--episodes", type=int, default=0,
                    help="Episode cap handed to play.py. 0 (default) means 'effectively unbounded' so "
                         "the rollout is bounded by --max_steps instead, which lets EVERY env finish "
                         "its first episode; a tight cap stops the loop early and over-weights the "
                         "fast-failing envs.")
    ap.add_argument("--max_steps", type=int, default=1000,
                    help="One episode timeout (1000). Every env completes exactly one episode.")
    ap.add_argument("--poll_seconds", type=int, default=300)
    ap.add_argument("--once", action="store_true", help="Sweep everything present, then exit.")
    ap.add_argument("--conditions", default="",
                    help="Comma-separated subset of condition names to run (default: all). "
                         "At ~50 min/condition, trimming 'combo' is how the sweep keeps pace with "
                         "training; the four single-axis conditions are the diagnostic ones.")
    ap.add_argument("--min_step_gap", type=int, default=0,
                    help="Skip a checkpoint unless its env_steps is at least this far beyond the "
                         "last one swept. 0 = sweep every checkpoint. Use when one full sweep takes "
                         "longer than training takes to emit the next checkpoint.")
    ap.add_argument("--checkpoint_list", default="",
                    help="Comma-separated checkpoint paths to sweep once, instead of watching "
                         "<run_dir>/checkpoints. For re-measuring kept artifacts (best_walker/*.pt) "
                         "under the current harness. Implies --once.")
    ap.add_argument("--objective", default="off", choices=sorted(OBJECTIVE_FLAGS),
                    help="Which planner objective to evaluate under. ALWAYS passed explicitly so nothing is "
                         "inherited from the checkpoint.")
    ap.add_argument("--wandb", action="store_true", help="Also log per-condition results to Weights & Biases.")
    ap.add_argument("--wandb_project", default="ldm-quad-mbrl")
    ap.add_argument("--wandb_name", default=None, help="Defaults to <run_dir basename>_fixedeval.")
    a = ap.parse_args()

    global CONDITIONS
    if a.conditions.strip():
        wanted = [c.strip() for c in a.conditions.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in {x[0] for x in CONDITIONS}]
        if unknown:
            raise SystemExit(f"[ckpt-sweep] unknown condition(s) {unknown}; "
                             f"known: {[x[0] for x in CONDITIONS]}")
        CONDITIONS = [c for c in CONDITIONS if c[0] in wanted]
        print(f"[ckpt-sweep] conditions restricted to {[c[0] for c in CONDITIONS]}", flush=True)

    ck_dir = os.path.join(a.run_dir, "checkpoints")
    out_dir = os.path.join(a.run_dir, "checkpoint_sweep")
    csv_path = os.path.join(a.run_dir, "checkpoint_sweep.csv")
    os.makedirs(out_dir, exist_ok=True)

    wb = init_wandb(a.run_dir, a.wandb_project, a.wandb_name) if a.wandb else None

    seen: set[str] = set()
    if os.path.exists(csv_path):
        # Key on (checkpoint, objective). Rows WITHOUT an objective column predate the
        # explicit-flags fix and were swept with whatever play.py inherited from the
        # checkpoint -- treat them as unusable, never as "already done". (Defaulting a
        # missing column to "off" silently skipped 15 contaminated Phase A rows.)
        _rows = list(csv.DictReader(open(csv_path)))
        _bad = [r for r in _rows if not r.get("objective")]
        if _bad:
            raise SystemExit(
                f"[ckpt-sweep] ABORT: {csv_path} has {len(_bad)} row(s) with no 'objective' column "
                "(pre-fix, objective inherited from the checkpoint). Delete or repair them first."
            )
        # Key on mode as well. Keying on (checkpoint, objective) alone would let a from_spawn row
        # satisfy a from_settled request and silently skip the sweep -- the same shape of bug as the
        # missing objective column, which skipped 15 contaminated Phase A rows by defaulting to "off".
        _nomode = [r for r in _rows if not r.get("mode")]
        if _nomode:
            raise SystemExit(
                f"[ckpt-sweep] ABORT: {csv_path} has {len(_nomode)} row(s) with no 'mode' column "
                "(predates from_spawn/from_settled). Delete or repair them first."
            )
        _want_mode = mode_name(a.command_start_step)
        seen = {
            r["checkpoint"] for r in _rows
            if r["objective"] == a.objective and r["mode"] == _want_mode
        }
        print(f"[ckpt-sweep] resuming objective={a.objective} mode={_want_mode}: "
              f"{len(seen)} checkpoint(s) already swept", flush=True)

    explicit = [p.strip() for p in a.checkpoint_list.split(",") if p.strip()]
    if explicit:
        missing = [p for p in explicit if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"[ckpt-sweep] ABORT: checkpoint(s) not found: {missing}")
        print(f"[ckpt-sweep] explicit list mode: {len(explicit)} checkpoint(s), then exit", flush=True)

    last_swept_step = -1
    while True:
        if explicit:
            pending = explicit
        else:
            pending = sorted(
                (f for f in os.listdir(ck_dir) if re.match(r"model_\d+\.pt$", f) and f not in seen),
                key=step_of,
            ) if os.path.isdir(ck_dir) else []
        for f in pending:
            path = f if explicit else os.path.join(ck_dir, f)
            if not explicit and a.min_step_gap > 0 and last_swept_step >= 0:
                if step_of(f) - last_swept_step < a.min_step_gap:
                    # Announce every skip: a silently thinned sweep reads as full coverage.
                    print(f"[ckpt-sweep] SKIP {f} (only {step_of(f) - last_swept_step} steps past "
                          f"{last_swept_step}, --min_step_gap={a.min_step_gap})", flush=True)
                    seen.add(f)
                    continue
            print(f"[ckpt-sweep] sweeping {os.path.basename(path)} at {time.strftime('%H:%M:%S')}", flush=True)
            rows = sweep_one(path, out_dir, a.action_scale,
                             a.episodes if a.episodes > 0 else 10 ** 9,
                             a.max_steps, a.objective, a.num_envs, a.settle_steps,
                             a.command_start_step)
            if rows:
                last_swept_step = step_of(os.path.basename(path))
            if rows:
                new = not os.path.exists(csv_path)
                if not new:
                    with open(csv_path) as _fh:
                        _hdr = _fh.readline().rstrip("\n").split(",")
                    if _hdr != FIELDS:
                        raise SystemExit(
                            f"[ckpt-sweep] ABORT: {csv_path} header {_hdr} != {FIELDS}. Appending would "
                            "misalign every column. Repair or delete the file first."
                        )
                with open(csv_path, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerows(rows)
                print("[ckpt-sweep] " + "  ".join(
                    f"{r['condition']}={r['pct_of_cmd']}±{r['pct_sem']}%"
                    f"/land_fail{r['landing_fail_rate']}/fall_post{r['fall_rate_post']}"
                    f"(n={r['n_landed']}/{r['n_eps']})"
                    for r in rows
                ), flush=True)
                if wb is not None:
                    _m = rows[0]["mode"]
                    payload = {"env_steps": rows[0]["env_steps"]}
                    for r in rows:
                        c = r["condition"]
                        payload[f"FixedEval/{_m}/{c}_pct"] = float(r["pct_of_cmd"]) if r["pct_of_cmd"] else 0.0
                        payload[f"FixedEval/{_m}/{c}_pct_sem"] = float(r["pct_sem"]) if r["pct_sem"] else 0.0
                        payload[f"FixedEval/{_m}/{c}_len"] = float(r["mean_length"])
                        payload[f"FixedEval/{_m}/{c}_fall_rate"] = float(r["fall_rate"])
                        payload[f"FixedEval/{_m}/{c}_landing_fail_rate"] = float(r["landing_fail_rate"])
                        payload[f"FixedEval/{_m}/{c}_fall_rate_post"] = float(r["fall_rate_post"])
                        payload[f"FixedEval/{_m}/{c}_n_eps"] = float(r["n_eps"])
                    # worst-axis summary: the metric save_best_metric should have been using
                    pcts = [float(r["pct_of_cmd"]) for r in rows if r["pct_of_cmd"]]
                    payload[f"FixedEval/{_m}/worst_axis_pct"] = min(pcts) if pcts else 0.0
                    payload[f"FixedEval/{_m}/mean_fall_rate"] = sum(float(r["fall_rate"]) for r in rows) / len(rows)
                    payload[f"FixedEval/{_m}/mean4_pct"] = (
                        sum(float(r["pct_of_cmd"]) for r in rows
                            if r["condition"] != "combo" and r["pct_of_cmd"]) / 4.0
                    )
                    wb.log(payload)
            seen.add(f)
        if a.once or explicit:
            break
        time.sleep(a.poll_seconds)


if __name__ == "__main__":
    main()

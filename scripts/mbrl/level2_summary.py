#!/usr/bin/env python3
"""Summarise a Level 2 closed-loop run directory (logs/mbrl/level2_v1 layout).

Each run lives in <root>/<arm>__<cond>__s<seed>/ with play.py's diagnostics metrics.csv
(every 5 steps) and episodes.csv (completed episodes). Per condition, one table: rows = arms,
columns = metrics as mean +- std over seeds, then each rolling arm's paired delta vs null with a
95% CI over seeds (t distribution). Prints the PASS rule and writes <root>/summary.md + .json.

Metrics
  vx            mean velocity_x_mean over steps >= 100 (excludes landing)
  track_x       mean tracking_x_abs_error over steps >= 100
  fell_envs     envs with >= 1 terminated episode (of 64), from episodes.csv
  ep_len        mean length of COMPLETED episodes (episodes still running at step 500 are not
                in episodes.csv, so this is conditional on the episode ending; see n_eps)
  n_eps         completed episodes
  phys_mse      mean model_physical_mse (all steps) -- one-step physical-head prediction error
  latent_mse    mean model_latent_consistency_mse (all steps)
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

# nullctxA = checkpoint A with --context_mode null (must equal null); dynonlyA = checkpoint A, rolling,
# --context_components dynamics_only (context in encoder + dynamics only; planner objective = stageL's).
# rollingV2 = sit-adapt-v2 model_final (context in the dynamics only), rolling; nullctxV2 = the same
# checkpoint with --context_mode null (must equal null).
ARMS = ("null", "rollingA", "rollingB", "rollingV2", "nullctxA", "nullctxV2", "dynonlyA", "dynstrictA")
DELTA_ARMS = ("rollingA", "rollingB", "rollingV2", "nullctxA", "nullctxV2", "dynonlyA", "dynstrictA")
# ref_id (gain 1.0, friction 0.8) is the in-distribution no-regression reference. "nominal" is
# gain 1.0 / friction 1.0, which is OUTSIDE the adapter's training friction range (0.25-0.8),
# so it is reported as the extrapolation condition fric_extrap_1.0.
CONDS = ("ref_id", "nominal", "gain0.8", "gain0.7", "gain0.6", "gain0.5", "fric0.6", "fric0.4", "fric0.3", "fric0.25")
LABEL = {"nominal": "fric_extrap_1.0 (gain 1.0, friction 1.0)", "ref_id": "ref_id (gain 1.0, friction 0.8)"}
METRICS = ("vx", "track_x", "fell_envs", "ep_len", "n_eps", "phys_mse", "latent_mse")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else math.nan


def run_metrics(path: str) -> dict | None:
    if not os.path.exists(os.path.join(path, ".done")):
        return None
    rows = list(csv.DictReader(open(os.path.join(path, "metrics.csv"))))
    late = [r for r in rows if int(r["step"]) >= 100]
    eps = []
    ep_path = os.path.join(path, "episodes.csv")
    if os.path.exists(ep_path):
        eps = list(csv.DictReader(open(ep_path)))
    fell = {int(e["env"]) for e in eps if int(e["terminated"])}
    return {
        "vx": _mean([_f(r["velocity_x_mean"]) for r in late]),
        "track_x": _mean([_f(r["tracking_x_abs_error"]) for r in late]),
        "fell_envs": float(len(fell)),
        "ep_len": _mean([_f(e["length"]) for e in eps]) if eps else math.nan,
        "n_eps": float(len(eps)),
        "phys_mse": _mean([_f(r["model_physical_mse"]) for r in rows]),
        "latent_mse": _mean([_f(r["model_latent_consistency_mse"]) for r in rows]),
        "plan_best": _mean([_f(r["planner_candidate_return_best"]) for r in rows]),
        "ctx_norm": _mean([_f(r.get("context_norm_mean")) for r in rows]) if any(r.get("context_norm_mean") for r in rows) else 0.0,
    }


def mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return math.nan, math.nan
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def paired_delta(a: dict, b: dict, metric: str):
    """mean and 95% CI half-width of (b - a) over seeds present in both."""
    d = [b[s][metric] - a[s][metric] for s in sorted(set(a) & set(b))
         if not (math.isnan(a[s][metric]) or math.isnan(b[s][metric]))]
    if not d:
        return math.nan, math.nan, 0
    m = statistics.fmean(d)
    hw = T975.get(len(d) - 1, 1.96) * statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else math.inf
    return m, hw, len(d)


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "logs/mbrl/level2_v1"
    data: dict = defaultdict(lambda: defaultdict(dict))  # cond -> arm -> seed -> metrics
    for name in sorted(os.listdir(root)):
        parts = name.split("__")
        if len(parts) != 3 or not parts[2].startswith("s"):
            continue
        m = run_metrics(os.path.join(root, name))
        if m is not None:
            data[parts[1]][parts[0]][int(parts[2][1:])] = m

    out_lines, out_json = [], {}

    def emit(s=""):
        print(s)
        out_lines.append(s)

    fmt = {"vx": "{:.3f}", "track_x": "{:.3f}", "fell_envs": "{:.1f}", "ep_len": "{:.0f}", "n_eps": "{:.1f}",
           "phys_mse": "{:.5f}", "latent_mse": "{:.2e}"}
    for cond in CONDS:
        if cond not in data:
            continue
        arms = data[cond]
        emit(f"\n=== {LABEL.get(cond, cond)} ===  (seeds per arm: " + ", ".join(f"{a} {len(arms[a])}" for a in ARMS if a in arms) + ")")
        emit("   arm        " + "".join(f"{m:>20s}" for m in METRICS))
        out_json[cond] = {}
        for arm in ARMS:
            if arm not in arms:
                continue
            cells, js = [], {}
            for m in METRICS:
                mu, sd = mean_std([v[m] for v in arms[arm].values()])
                js[m] = [mu, sd]
                cells.append(f"{fmt[m].format(mu)}+-{fmt[m].format(sd)}".rjust(20))
            out_json[cond][arm] = js
            emit(f"   {arm:10s} " + "".join(cells))
        if "null" in arms:
            for arm in DELTA_ARMS:
                if arm not in arms:
                    continue
                cells, js = [], {}
                for m in METRICS:
                    mu, hw, n = paired_delta(arms["null"], arms[arm], m)
                    js[m] = [mu, hw, n]
                    cells.append(f"{fmt[m].format(mu)}+-{fmt[m].format(hw)}".rjust(20))
                out_json[cond][f"delta_{arm}"] = js
                emit(f"   d:{arm:9s}" + "".join(cells) + "   (paired vs null, 95% CI)")
        san = "  ".join(
            f"{a}: plan_best {_mean([v['plan_best'] for v in arms[a].values()]):.1f} ctx {_mean([v['ctx_norm'] for v in arms[a].values()]):.3f}"
            for a in ARMS if a in arms)
        emit(f"   sanity  {san}")

    # overview: null | v1 A | v2 per condition, deltas vs null and v2 vs A (paired over seeds)
    emit("\n=== overview: null | v1 A (rollingA) | v2 (rollingV2); paired deltas, mean +- 95% CI over seeds ===")
    ov_metrics = (("vx", "{:+.3f}+-{:.3f}"), ("fell_envs", "{:+.1f}+-{:.1f}"), ("phys_mse", "{:+.5f}+-{:.5f}"))
    emit("   cond         arm        " + "".join(f"{m:>12s}" for m, _ in ov_metrics)
         + "   | delta vs null: " + "  ".join(f"d{m}" for m, _ in ov_metrics) + "   | v2 - A: " + "  ".join(f"d{m}" for m, _ in ov_metrics))
    for cond in CONDS:
        if cond not in data or "null" not in data[cond]:
            continue
        arms = data[cond]
        label = "fric_ext1.0" if cond == "nominal" else cond
        for arm in ("null", "rollingA", "rollingV2"):
            if arm not in arms:
                continue
            vals = "".join(f"{mean_std([v[m] for v in arms[arm].values()])[0]:>12.4f}" for m, _ in ov_metrics)
            dnull = "" if arm == "null" else "  ".join(
                f.format(*paired_delta(arms["null"], arms[arm], m)[:2]) for m, f in ov_metrics)
            dva = ""
            if arm == "rollingV2" and "rollingA" in arms:
                dva = "  ".join(f.format(*paired_delta(arms["rollingA"], arms["rollingV2"], m)[:2]) for m, f in ov_metrics)
                out_json[cond]["delta_rollingV2_vs_rollingA"] = {
                    m: list(paired_delta(arms["rollingA"], arms["rollingV2"], m)) for m, _ in ov_metrics}
            emit(f"   {label:12s} {arm:10s} {vals}   | {dnull:48s} | {dva}")

    emit("\n=== PASS RULE ===")
    verdict = {}
    for arm in DELTA_ARMS:
        k = f"delta_{arm}"
        if not any(k in out_json.get(c, {}) for c in CONDS):
            continue
        nom = out_json.get("ref_id", {}).get(k)
        r1 = None
        if nom:
            dvx, hvx, _ = nom["vx"]
            dfell, hfell, _ = nom["fell_envs"]
            r1 = (dvx + hvx >= 0) and (dfell - hfell <= 0)
        r2_conds = {c: out_json[c][k]["phys_mse"][0] < 0 for c in CONDS if c != "ref_id" and k in out_json.get(c, {})}
        r2 = all(r2_conds.values()) if r2_conds else None
        verdict[arm] = {"nominal_no_regression": r1, "pred_better_all_nonnominal": r2, "per_cond": r2_conds}
        emit(f"   {arm}: (1) ref_id vx/falls not worse than null (within CI): {r1}   "
             f"(2) phys prediction < null at every other condition done so far "
             f"({len(r2_conds)} conds): {r2}  " + " ".join(f"{c}:{'y' if v else 'N'}" for c, v in r2_conds.items()))
    emit("   (3) closed-loop deltas are reported above as-is; friction deltas are expected to be small")
    out_json["pass"] = verdict
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("```\n" + "\n".join(out_lines) + "\n```\n")
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(out_json, f, indent=1)
    print(f"\nwrote {root}/summary.md and summary.json")


if __name__ == "__main__":
    main()

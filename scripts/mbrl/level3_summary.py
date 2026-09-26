#!/usr/bin/env python3
"""Summarise Level 3 passive-prediction runs (identification speed after a dynamics switch).

Layout: <root>/<cond>__s<seed>/metrics.csv written by
    play.py --predict_with_checkpoint <ckpt> --predict_arms ... --diagnostics --diagnostics_interval 1
with a within-episode switch at --switch_step. Arms are read from the predictor_<arm>_phys1 columns.

Per condition and arm, over seeds (mean +- 95% CI, t distribution):
  windows     physical MSE at k = 1, 4, 8 in W-step windows aligned to the switch
  pre         mean error over the last `pre` steps before the switch
  peak        max window error in the first `peak` steps after the switch
  steady      mean error over the last `steady` steps of the run
  recovery    steps from the switch to the first window whose error is within `tol` (10%) of the
              ORACLE arm's steady level (oracle = truncated<switch>: window cleared at the switch);
              '>run' if never
Overview: arm x condition -> recovery (k=1) and steady error, with the gap to the oracle.
Also the context-drift time course per arm around the switch.

Usage: python scripts/mbrl/level3_summary.py logs/mbrl/level3_v1 [--switch_step 250] [--window 25]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def mean_ci(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return math.nan, math.nan
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, math.inf
    return m, T975.get(len(xs) - 1, 1.96) * statistics.stdev(xs) / math.sqrt(len(xs))


def load_run(path: str) -> tuple[list[int], dict[str, list[float]]]:
    rows = list(csv.DictReader(open(path)))
    steps = [int(r["step"]) for r in rows]
    cols = {k: [_f(r[k]) for r in rows] for k in rows[0] if k.startswith("predictor_")}
    return steps, cols


def window_means(steps, values, lo, hi, width):
    out = []
    for w0 in range(lo, hi, width):
        v = [x for s, x in zip(steps, values) if w0 <= s < w0 + width and not math.isnan(x)]
        out.append(statistics.fmean(v) if v else math.nan)
    return out


def span_mean(steps, values, lo, hi):
    v = [x for s, x in zip(steps, values) if lo <= s < hi and not math.isnan(x)]
    return statistics.fmean(v) if v else math.nan


def analyse(root: str, switch: int, width: int, pre: int, peak: int, steady: int, tol: float) -> dict:
    runs = defaultdict(dict)  # cond -> seed -> (steps, cols)
    for name in sorted(os.listdir(root)):
        m = re.match(r"^(.+)__s(\d+)$", name)
        path = os.path.join(root, name, "metrics.csv")
        if m and os.path.exists(path) and os.path.exists(os.path.join(root, name, ".done")):
            runs[m.group(1)][int(m.group(2))] = load_run(path)
    result = {}
    for cond, seeds in runs.items():
        any_cols = next(iter(seeds.values()))[1]
        arms = sorted({re.match(r"predictor_(.+)_phys1$", c).group(1) for c in any_cols if c.endswith("_phys1")})
        oracle = f"truncated{switch}"
        end = max(max(s[0]) for s in seeds.values()) + 1
        lo = switch - 4 * width
        res = {"arms": {}, "oracle": oracle if oracle in arms else None, "seeds": sorted(seeds), "window_start": lo,
               "window": width, "switch": switch}
        # oracle steady level per seed and k (recovery threshold)
        steady_oracle = {}
        for k in (1, 4, 8):
            if res["oracle"]:
                steady_oracle[k] = {s: span_mean(st, c[f"predictor_{oracle}_phys{k}"], end - steady, end)
                                    for s, (st, c) in seeds.items()}
        for arm in arms:
            entry = {}
            for k in (1, 4, 8):
                col = f"predictor_{arm}_phys{k}"
                per = {s: (st, c[col]) for s, (st, c) in seeds.items() if col in c}
                wins = [window_means(st, v, lo, end, width) for st, v in per.values()]
                entry[f"k{k}"] = {
                    "windows": [list(mean_ci(list(x))) for x in zip(*wins)],
                    "pre": list(mean_ci([span_mean(st, v, switch - pre, switch) for st, v in per.values()])),
                    "peak": list(mean_ci([max((x for x in window_means(st, v, switch, switch + peak, width)
                                               if not math.isnan(x)), default=math.nan) for st, v in per.values()])),
                    "steady": list(mean_ci([span_mean(st, v, end - steady, end) for st, v in per.values()])),
                }
                if res["oracle"]:
                    rec = []
                    for s, (st, v) in per.items():
                        thr = steady_oracle[k][s] * (1 + tol)
                        post = window_means(st, v, switch, end, width)
                        hit = next((i for i, x in enumerate(post) if not math.isnan(x) and x <= thr), None)
                        rec.append((hit + 1) * width if hit is not None else math.inf)  # window END = steps needed
                    finite = [r for r in rec if math.isfinite(r)]
                    entry[f"k{k}"]["recovery_steps_per_seed"] = rec
                    entry[f"k{k}"]["recovery"] = (list(mean_ci(finite)) + [len(finite), len(rec)]) if finite else \
                        [math.inf, math.nan, 0, len(rec)]
            drift = [window_means(st, c.get(f"predictor_{arm}_ctx_drift", [math.nan] * len(st)), lo, end, width)
                     for st, c in seeds.values()]
            entry["ctx_drift_windows"] = [mean_ci(list(x))[0] for x in zip(*drift)]
            res["arms"][arm] = entry
        result[cond] = res
    return result


def fmt(m, ci, spec="{:.4f}"):
    if math.isnan(m):
        return "   nan"
    if math.isinf(m):
        return "  >run"
    return spec.format(m) + ("" if math.isnan(ci) else ("+-inf" if math.isinf(ci) else "+-" + spec.format(ci)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--switch_step", type=int, default=250)
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--pre", type=int, default=100)
    p.add_argument("--peak", type=int, default=100)
    p.add_argument("--steady", type=int, default=150)
    p.add_argument("--tol", type=float, default=0.10)
    a = p.parse_args()
    res = analyse(a.root, a.switch_step, a.window, a.pre, a.peak, a.steady, a.tol)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    for cond, r in res.items():
        emit(f"\n=== {cond}  (seeds {r['seeds']}, switch at step {r['switch']}, oracle {r['oracle']}) ===")
        for k in (1, 4, 8):
            emit(f" physical MSE k={k}: pre / peak / steady / recovery(steps to within {int(a.tol*100)}% of oracle steady)")
            for arm, e in r["arms"].items():
                x = e[f"k{k}"]
                rec = x.get("recovery")
                rec_s = "-" if rec is None else (fmt(rec[0], rec[1], "{:.0f}") + f" ({rec[2]}/{rec[3]} seeds)")
                emit(f"   {arm:14s} {fmt(*x['pre']):>18s} {fmt(*x['peak']):>18s} {fmt(*x['steady']):>18s}   {rec_s}")
        w0, width = r["window_start"], r["window"]
        n = len(next(iter(r["arms"].values()))["k1"]["windows"])
        emit(f" k=1 error in {width}-step windows (window start steps; switch at {r['switch']})")
        emit("   " + " " * 14 + "".join(f"{w0 + i * width:>8d}" for i in range(n)))
        for arm, e in r["arms"].items():
            emit(f"   {arm:14s}" + "".join(f"{m:8.4f}" for m, _ in e["k1"]["windows"]))
        emit(" context drift |c_t - c_(t-1)| in the same windows")
        for arm, e in r["arms"].items():
            emit(f"   {arm:14s}" + "".join(f"{m:8.3f}" for m in e["ctx_drift_windows"]))

    emit("\n=== overview: recovery (k=1, steps after the switch) and post-switch steady error (k=1) ===")
    emit("   cond              arm             recovery            steady              steady - oracle")
    for cond, r in res.items():
        orc = r["arms"].get(r["oracle"]) if r["oracle"] else None
        for arm, e in r["arms"].items():
            x = e["k1"]
            rec = x.get("recovery", [math.nan, math.nan, 0, 0])
            gap = x["steady"][0] - orc["k1"]["steady"][0] if orc else math.nan
            emit(f"   {cond:17s} {arm:14s} {fmt(rec[0], rec[1], '{:.0f}'):>18s} {fmt(*x['steady']):>18s} {gap:+.5f}")
    with open(os.path.join(a.root, "level3_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: None)
    with open(os.path.join(a.root, "level3_summary.md"), "w") as f:
        f.write("```\n" + "\n".join(lines) + "\n```\n")
    print(f"\nwrote {a.root}/level3_summary.md and level3_summary.json")


if __name__ == "__main__":
    main()

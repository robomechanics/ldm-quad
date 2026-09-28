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
  settling    SETTLING TIME (steps after the switch). Threshold thr = (1 + `tol`) x the ORACLE's
              mean error over the final `steady` span (oracle = truncated<switch>: window cleared at
              the switch). The post-switch run is cut into W-step windows from the switch up to the
              steady span, then the steady span as one final block (judged by its MEAN).
              --settle_rule last_exit (default): '>run' (inf) if the steady block is above thr;
                otherwise the END of the last window above thr (0 = never above the band). Every
                window counts on its own, so one noisy window anywhere before the steady span sets
                the settling time (deliberately: a late excursion is a late excursion).
              --settle_rule suffix_mean: the start of the first block i such that for EVERY block
                j >= i the mean error from j to the end of the run is <= thr. Robust to one noisy
                window, but a long in-band tail dilutes any early excess (on level3_v1 it sends 20
                of 23 nonzero k=4/8 settling times to 0, erasing the rolling96 lag), so it is not
                the reported number.
              Both: '>run' if the steady mean or thr is NaN; a window with no finite rows counts
              as OUT of the band. (Not "first window inside the band": after a gain switch the
              error RISES over the first 25-50 steps as the gait changes, so the first post-switch
              window is trivially inside.)
Overview: arm x condition -> settling time (k=1) and steady error, with the gap to the oracle.
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
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summary_stats import fmt, mean_ci  # noqa: E402


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


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


def settling_time(steps, values, switch: int, end: int, steady: int, width: int, thr: float,
                  rule: str = "last_exit") -> float:
    """Settling time (see the module docstring for the two rules)."""
    if math.isnan(thr):
        return math.inf
    starts = list(range(switch, end - steady, width)) + [end - steady]  # block starts; last = steady span
    means, empty = [], []
    for j, lo in enumerate(starts):
        hi = starts[j + 1] if j + 1 < len(starts) else end
        v = [x for s, x in zip(steps, values) if lo <= s < hi and not math.isnan(x)]
        empty.append(not v)
        means.append(v)
    if empty[-1]:
        return math.inf
    ok, suffix = [False] * len(starts), []
    for j in range(len(starts) - 1, -1, -1):  # suffix means, and "every later suffix is within"
        suffix = means[j] + suffix
        inside = (not empty[j]) and statistics.fmean(suffix) <= thr
        ok[j] = inside and (j == len(starts) - 1 or ok[j + 1])
    if not ok[-1]:
        return math.inf
    if rule == "last_exit":
        out = [j for j in range(len(starts) - 1) if empty[j] or statistics.fmean(means[j]) > thr]
        return float(starts[out[-1]] + width - switch) if out else 0.0
    if rule != "suffix_mean":
        raise ValueError(f"unknown settle rule {rule!r}")
    i = min(j for j in range(len(starts)) if ok[j])
    return float(starts[i] - switch)


def analyse(root: str, switch: int, width: int, pre: int, peak: int, steady: int, tol: float,
            rule: str = "last_exit") -> dict:
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
        if end - steady <= switch:
            raise ValueError(f"{cond}: steady span [{end - steady}, {end}) starts at/before the switch {switch}")
        if (end - steady - switch) % width:
            raise ValueError(f"{cond}: (end - steady - switch) = {end - steady - switch} is not a multiple of "
                             f"the window {width}; a window would straddle the steady span")
        lo = switch - 4 * width
        res = {"arms": {}, "oracle": oracle if oracle in arms else None, "seeds": sorted(seeds), "window_start": lo,
               "window": width, "switch": switch}
        # oracle steady level per seed and k (settling threshold)
        steady_oracle = {}
        for k in (1, 4, 8):
            if res["oracle"]:
                col = f"predictor_{oracle}_phys{k}"
                steady_oracle[k] = {s: span_mean(st, c[col], end - steady, end)
                                    for s, (st, c) in seeds.items() if col in c}
        for arm in arms:
            entry = {}
            for k in (1, 4, 8):
                col = f"predictor_{arm}_phys{k}"
                per = {s: (st, c[col]) for s, (st, c) in seeds.items() if col in c}
                wins = [window_means(st, v, lo, end, width) for st, v in per.values()]
                steady_mean = {s: span_mean(st, v, end - steady, end) for s, (st, v) in per.items()}
                entry[f"k{k}"] = {
                    "windows": [list(mean_ci(list(x))) for x in zip(*wins)],
                    "pre": list(mean_ci([span_mean(st, v, switch - pre, switch) for st, v in per.values()])),
                    "peak": list(mean_ci([max((x for x in window_means(st, v, switch, switch + peak, width)
                                               if not math.isnan(x)), default=math.nan) for st, v in per.values()])),
                    "steady": list(mean_ci(list(steady_mean.values()))),
                }
                if res["oracle"]:
                    settle = []
                    for s, (st, v) in per.items():
                        thr = steady_oracle[k].get(s, math.nan) * (1 + tol)
                        settle.append(math.inf if math.isnan(steady_mean[s]) else
                                      settling_time(st, v, switch, end, steady, width, thr, rule))
                    finite = [r for r in settle if math.isfinite(r)]
                    entry[f"k{k}"]["settling_steps_per_seed"] = settle
                    entry[f"k{k}"]["settling"] = (list(mean_ci(finite)) + [len(finite), len(settle)]) if finite else \
                        [math.inf, math.nan, 0, len(settle)]
            drift = [window_means(st, c.get(f"predictor_{arm}_ctx_drift", [math.nan] * len(st)), lo, end, width)
                     for st, c in seeds.values()]
            entry["ctx_drift_windows"] = [mean_ci(list(x))[0] for x in zip(*drift)]
            res["arms"][arm] = entry
        result[cond] = res
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--switch_step", type=int, default=250)
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--pre", type=int, default=100)
    p.add_argument("--peak", type=int, default=100)
    p.add_argument("--steady", type=int, default=150)
    p.add_argument("--tol", type=float, default=0.10)
    p.add_argument("--settle_rule", choices=["last_exit", "suffix_mean"], default="last_exit")
    a = p.parse_args()
    res = analyse(a.root, a.switch_step, a.window, a.pre, a.peak, a.steady, a.tol, a.settle_rule)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    for cond, r in res.items():
        emit(f"\n=== {cond}  (seeds {r['seeds']}, switch at step {r['switch']}, oracle {r['oracle']}) ===")
        for k in (1, 4, 8):
            emit(f" physical MSE k={k}: pre / peak / steady / settling time ({a.settle_rule}: steps after the switch "
                 f"until the error stays within {int(a.tol*100)}% of the oracle's steady error)")
            for arm, e in r["arms"].items():
                x = e[f"k{k}"]
                rec = x.get("settling")
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

    emit("\n=== overview: settling time (k=1, steps after the switch) and post-switch steady error (k=1) ===")
    emit("   cond              arm             settling            steady              steady - oracle")
    for cond, r in res.items():
        orc = r["arms"].get(r["oracle"]) if r["oracle"] else None
        for arm, e in r["arms"].items():
            x = e["k1"]
            rec = x.get("settling", [math.nan, math.nan, 0, 0])
            gap = x["steady"][0] - orc["k1"]["steady"][0] if orc else math.nan
            emit(f"   {cond:17s} {arm:14s} {fmt(rec[0], rec[1], '{:.0f}'):>18s} {fmt(*x['steady']):>18s} {gap:+.5f}")
    with open(os.path.join(a.root, "level3_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: None)
    with open(os.path.join(a.root, "level3_summary.md"), "w") as f:
        f.write("```\n" + "\n".join(lines) + "\n```\n")
    print(f"\nwrote {a.root}/level3_summary.md and level3_summary.json")


if __name__ == "__main__":
    main()

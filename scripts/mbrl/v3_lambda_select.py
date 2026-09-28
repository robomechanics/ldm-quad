#!/usr/bin/env python3
"""Pick the v3 planner risk weight lambda from the seed-0 sweep (Level 2 harness layout).

Runs: <root>/lam<tag>V3__{ref_id,gain0.8}__s0 for tag in LAMBDAS (e.g. lam05V3 = lambda 0.5),
v3 model_final, EMA 0.05 context, two-stage k=128, 64 envs x 500 steps. Reference: null__ref_id__s0.

Selection rule (ldm-quad-1c, 2026-09-28):
  qualifies  ref_id fell_envs <= null's ref_id fell_envs (seed 0), and ref_id vx >= 0.37
  choice     the qualifying lambda with the highest gain-0.8 vx
  fallback   none qualifies -> lambda 0, flagged
Also reported: the risk-term share = lambda x mean(planner_ensemble_return_std) / |mean(planner_
candidate_return_mean)| -- how large the penalty is relative to the returns it ranks.

Writes <root>/v3_lambda.txt (the chosen lambda, one number) and <root>/v3_lambda_sweep.{txt,json}.
Usage: python scripts/mbrl/v3_lambda_select.py [logs/mbrl/level2_v1]
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import statistics
import sys

LAMBDAS = {"0": 0.0, "05": 0.5, "1": 1.0, "2": 2.0}
MIN_REF_VX = 0.37

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_l2s", os.path.join(_here, "level2_summary.py"))
l2s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l2s)


def _col_mean(path: str, col: str) -> float:
    vals = []
    for r in csv.DictReader(open(os.path.join(path, "metrics.csv"))):
        try:
            v = float(r.get(col) or "nan")
        except ValueError:
            continue
        if math.isfinite(v):
            vals.append(v)
    return statistics.fmean(vals) if vals else math.nan


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "logs/mbrl/level2_v1"
    null_ref = l2s.run_metrics(os.path.join(root, "null__ref_id__s0"))
    if null_ref is None:
        raise SystemExit("null__ref_id__s0 missing: no reference for the fall criterion")
    rows, lines = {}, []
    for tag, lam in LAMBDAS.items():
        entry = {"lambda": lam}
        for cond in ("ref_id", "gain0.8"):
            path = os.path.join(root, f"lam{tag}V3__{cond}__s0")
            m = l2s.run_metrics(path)
            if m is None:
                entry[cond] = None
                continue
            std = _col_mean(path, "planner_ensemble_return_std")
            ret = _col_mean(path, "planner_candidate_return_mean")
            entry[cond] = {"vx": m["vx"], "fell_envs": m["fell_envs"], "track_x": m["track_x"],
                           "plan_best": m["plan_best"], "ens_return_std": std, "return_mean": ret,
                           "risk_share": lam * std / abs(ret) if ret and math.isfinite(ret) else math.nan}
        ref = entry["ref_id"]
        entry["qualifies"] = bool(ref and ref["fell_envs"] <= null_ref["fell_envs"] and ref["vx"] >= MIN_REF_VX)
        rows[tag] = entry
    qual = [t for t, e in rows.items() if e["qualifies"] and e["gain0.8"] is not None]
    if qual:
        chosen = max(qual, key=lambda t: rows[t]["gain0.8"]["vx"])
        flag = ""
    else:
        chosen, flag = "0", "FLAG: no lambda met the ref_id criteria; falling back to lambda 0"
    all_pos_worse = all(e["ref_id"] is not None and e["ref_id"]["fell_envs"] > null_ref["fell_envs"]
                        for t, e in rows.items() if LAMBDAS[t] > 0)

    def f(x, spec):
        return "   -" if x is None or (isinstance(x, float) and not math.isfinite(x)) else spec.format(x)

    lines.append(f"v3 lambda sweep (seed 0, 64 envs x 500 steps); null ref_id s0: vx {null_ref['vx']:.3f} "
                 f"fell {null_ref['fell_envs']:.0f}")
    lines.append("lambda | ref_id vx  fell  risk_share | gain0.8 vx  fell  track_x  risk_share | ens_std(g0.8) | qualifies")
    for tag, e in rows.items():
        r, g = e["ref_id"] or {}, e["gain0.8"] or {}
        lines.append(f"{e['lambda']:6.1f} | {f(r.get('vx'), '{:9.3f}')} {f(r.get('fell_envs'), '{:5.0f}')} "
                     f"{f(r.get('risk_share'), '{:10.4f}')} | {f(g.get('vx'), '{:10.3f}')} {f(g.get('fell_envs'), '{:5.0f}')} "
                     f"{f(g.get('track_x'), '{:8.3f}')} {f(g.get('risk_share'), '{:10.4f}')} | "
                     f"{f(g.get('ens_return_std'), '{:13.4f}')} | {'yes' if e['qualifies'] else 'no'}")
    lines.append(f"chosen lambda = {LAMBDAS[chosen]} "
                 f"(rule: ref_id fell <= null ({null_ref['fell_envs']:.0f}) and ref_id vx >= {MIN_REF_VX}; "
                 f"then highest gain-0.8 vx) {flag}")
    if all_pos_worse:
        lines.append("NOTE: every lambda > 0 falls more than null at ref_id; Level 2 runs at lambda 0 "
                     "(v2 structure + ensemble mean: a control, not the risk-aware planner)")
    print("\n".join(lines))
    with open(os.path.join(root, "v3_lambda_sweep.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(root, "v3_lambda_sweep.json"), "w") as fh:
        json.dump({"rows": rows, "null_ref_id_s0": null_ref, "chosen": LAMBDAS[chosen], "flag": flag,
                   "all_positive_lambdas_fall_more_than_null": all_pos_worse}, fh, indent=1,
                  default=lambda o: None)
    with open(os.path.join(root, "v3_lambda.txt"), "w") as fh:
        fh.write(f"{LAMBDAS[chosen]}\n")


if __name__ == "__main__":
    main()

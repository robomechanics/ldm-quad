"""Per-checkpoint Level 1 trend + the adopted weight-norm alert rule.

Usage: python scripts/mbrl/level1_trend.py <run>/level1   (writes trend.csv, appends ALERT.txt)

Alert when ||W c|| / ||base pre-act|| > 0.5 in any conditioned module among encoder / dynamics
(a dynamics_only checkpoint has only the dynamics ratio) WHILE the Level 1 gain has plateaued
across the last two consecutive checkpoints. Gain = relative open-loop improvement of
the true context over null at k=16 on the held-out buffer (train buffer if no held-out); a
checkpoint counts as "no progress" if it improves on the previous one by < 0.005.
"""
import csv, glob, json, os, re, sys

d = sys.argv[1]
rows = []
results = []
for f in glob.glob(os.path.join(d, "model_*.json")):
    r = json.load(open(f))
    if "buffers" in r:  # skip the pre-held-out single-buffer format
        results.append((int(r["env_steps"]), os.path.basename(f)[: -len(".json")], r))
# model_final has no step in its name: order by the env_steps recorded in the result
for _steps, name, r in sorted(results, key=lambda t: (t[0], t[1] == "model_final")):
    ratios = r["context_term_over_base_preact"]
    row = {"checkpoint": name, "env_steps": r["env_steps"],
           "ratio_encoder": ratios.get("encoder", float("nan")), "ratio_dynamics": ratios.get("dynamics", float("nan")),
           "total_ctx_weight_norm": r["context_weight_total_norm"]}
    for tag, b in r["buffers"].items():
        a = b["open_loop"]["all"]
        for k in (1, 16):
            row[f"{tag}_improve_k{k}"] = 1 - a["true"]["phys_per_k"][k - 1] / a["null"]["phys_per_k"][k - 1]
        row[f"{tag}_r2_gain"] = b["probe_r2"]["context"]["overall"]["motor_gain"]
        row[f"{tag}_r2_friction"] = b["probe_r2"]["context"]["overall"]["foot_friction"]
        row[f"{tag}_pass"] = int(b["pass"]["passed"])
    for tag, cb in r.get("cross_buffer_probe", {}).items():
        row[f"xprobe_{tag}_r2_gain"] = cb["context"]["overall_r2"]["motor_gain"]
    rows.append(row)
if not rows:
    sys.exit(0)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(os.path.join(d, "trend.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
metric = "heldout_improve_k16" if "heldout_improve_k16" in rows[-1] else "train_improve_k16"
last = rows[-1]
over = max(v for v in (last["ratio_encoder"], last["ratio_dynamics"]) if v == v) > 0.5  # skip NaN (absent)
plateau = len(rows) >= 3 and all(
    rows[i][metric] < rows[i - 1][metric] + 0.005 for i in (len(rows) - 1, len(rows) - 2)
)
print(f"[trend] {last['env_steps']}: ratio enc {last['ratio_encoder']:.3f} dyn {last['ratio_dynamics']:.3f}  "
      f"{metric} {last[metric]:+.3f}  plateau={plateau}  over0.5={over}")
if over and plateau:
    with open(os.path.join(d, "ALERT.txt"), "a") as f:
        f.write(json.dumps({"rule": "ratio>0.5 in encoder/dynamics while Level 1 gain plateaued over 2 checkpoints",
                            "metric": metric, "last3": rows[-3:]}, indent=1) + "\n")
    print("[trend] ALERT written")

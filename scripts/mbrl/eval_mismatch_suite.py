#!/usr/bin/env python3

"""Run MBRL play/eval across held-out mismatch cases."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime


DEFAULT_MISMATCHES = ["nominal", "low_friction", "mass", "motor_weakness", "rough", "push"]
EVAL_RE = re.compile(r"mean_return=([-+0-9.eE]+)\s+std_return=([-+0-9.eE]+)\s+mean_length=([-+0-9.eE]+)")
COMPLETED_RE = re.compile(r"steps=([0-9]+)\s+completed_episodes=([0-9]+)")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch scripts/mbrl/play.py once per system/terrain mismatch and collect metrics.",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the MBRL checkpoint to evaluate.")
    parser.add_argument(
        "--mismatches",
        nargs="+",
        default=DEFAULT_MISMATCHES,
        choices=DEFAULT_MISMATCHES,
        help="Mismatch cases to run.",
    )
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--play_script", default="scripts/mbrl/play.py", help="Path to the play script.")
    parser.add_argument("--continue_on_error", action="store_true", default=False, help="Keep running remaining cases after a failure.")
    return parser.parse_known_args()


def parse_metrics(stdout: str) -> dict[str, float]:
    metrics = {
        "steps": 0.0,
        "completed_episodes": 0.0,
        "mean_return": float("nan"),
        "std_return": float("nan"),
        "mean_length": float("nan"),
    }
    for line in stdout.splitlines():
        completed = COMPLETED_RE.search(line)
        if completed:
            metrics["steps"] = float(completed.group(1))
            metrics["completed_episodes"] = float(completed.group(2))
        eval_match = EVAL_RE.search(line)
        if eval_match:
            metrics["mean_return"] = float(eval_match.group(1))
            metrics["std_return"] = float(eval_match.group(2))
            metrics["mean_length"] = float(eval_match.group(3))
    return metrics


def main() -> int:
    args, play_args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    if args.output is None:
        run_dir = os.path.dirname(os.path.dirname(checkpoint))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output = os.path.join(run_dir, f"mismatch_eval_{timestamp}.csv")

    rows: list[dict[str, float | str | int]] = []
    for mismatch in args.mismatches:
        cmd = [
            sys.executable,
            args.play_script,
            "--checkpoint",
            checkpoint,
            "--mismatch",
            mismatch,
            *play_args,
        ]
        print(f"[SUITE] Running mismatch={mismatch}", flush=True)
        print("[SUITE] " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        print(result.stdout, end="", flush=True)

        metrics = parse_metrics(result.stdout)
        row: dict[str, float | str | int] = {
            "mismatch": mismatch,
            "returncode": result.returncode,
            **metrics,
        }
        rows.append(row)
        if result.returncode != 0 and not args.continue_on_error:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mismatch", "returncode", "steps", "completed_episodes", "mean_return", "std_return", "mean_length"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SUITE] Wrote {args.output}", flush=True)
    return 0 if all(int(row["returncode"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

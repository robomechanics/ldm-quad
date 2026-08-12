#!/usr/bin/env python3
"""Upload MBRL metrics.csv rows directly to W&B charts."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--entity", default=None, help="Optional W&B entity/team.")
    parser.add_argument("--run_id", required=True, help="Existing W&B run id to resume.")
    parser.add_argument("--name", default=None, help="Optional W&B run name.")
    parser.add_argument("--follow", action="store_true", help="Keep uploading new rows as metrics.csv grows.")
    parser.add_argument("--poll_seconds", type=float, default=30.0, help="Polling interval for --follow.")
    return parser.parse_args()


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def read_rows(path: Path, min_step: int) -> list[tuple[int, dict[str, float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[tuple[int, dict[str, float]]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            step_value = as_float(raw.get("env_steps", ""))
            if step_value is None:
                continue
            step = int(step_value)
            if step <= min_step:
                continue
            metrics = {}
            for key, value in raw.items():
                parsed = as_float(value or "")
                if parsed is not None:
                    metrics[key] = parsed
            if metrics:
                rows.append((step, metrics))
    return rows


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics).expanduser().resolve()

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        name=args.name,
        resume="allow",
        config={"direct_metrics_csv": os.fspath(metrics_path)},
    )

    last_step = -1
    try:
        while True:
            rows = read_rows(metrics_path, last_step)
            for step, metrics in rows:
                run.log(metrics, step=step)
                last_step = max(last_step, step)
            if rows:
                print(f"[W&B CSV] uploaded rows={len(rows)} last_step={last_step}", flush=True)
            if not args.follow:
                break
            time.sleep(max(args.poll_seconds, 1.0))
    finally:
        run.finish()


if __name__ == "__main__":
    main()

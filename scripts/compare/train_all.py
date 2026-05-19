#!/usr/bin/env python3
"""Run PPO, SAC, and MBRL training jobs with a shared comparison setup."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "ldm_quad"
DEFAULT_METHODS = ("ppo", "sac", "mbrl")
PPO_ROLLOUTS = 24
PRESETS = {
    "smoke": {
        "budget_steps": 24,
        "num_envs": 4,
        "mbrl_num_envs": 4,
        "mbrl_seed_steps": 8,
        "mbrl_updates_per_step": 2,
    },
    "debug": {
        "budget_steps": 1000,
        "num_envs": 256,
        "mbrl_num_envs": 64,
        "mbrl_seed_steps": 100,
        "mbrl_updates_per_step": 8,
    },
    "full": {
        "budget_steps": 50000,
        "num_envs": 4096,
        "mbrl_num_envs": 64,
        "mbrl_seed_steps": 500,
        "mbrl_updates_per_step": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO, SAC, and MBRL training for comparison.")
    parser.add_argument("--preset", choices=PRESETS, default=None, help="Fill unset training scale arguments.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--budget_steps", type=int, default=None, help="Approximate env-step budget per method.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--mbrl_num_envs", type=int, default=None, help="Override num_envs for MBRL if 4096 is too heavy.")
    parser.add_argument("--mbrl_seed_steps", type=int, default=None)
    parser.add_argument("--mbrl_updates_per_step", type=int, default=None)
    parser.add_argument("--command_x", type=float, default=0.6, help="Fixed forward velocity command in m/s.")
    parser.add_argument("--command_y", type=float, default=0.0, help="Fixed lateral velocity command in m/s.")
    parser.add_argument("--command_yaw", type=float, default=0.0, help="Fixed yaw velocity command in rad/s.")
    parser.add_argument("--wander", action="store_true", default=False, help="Train on a range of nonzero walking commands.")
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest path. Defaults to logs/comparisons/<timestamp>/manifest.json.",
    )
    args = parser.parse_args()
    preset = PRESETS[args.preset or "debug"]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def latest_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    return max(dirs, key=lambda path: path.stat().st_mtime) if dirs else None


def common_app_args(args: argparse.Namespace) -> list[str]:
    cmd_args: list[str] = []
    if args.device:
        cmd_args.extend(["--device", args.device])
    if args.headless:
        cmd_args.append("--headless")
    if args.disable_fabric:
        cmd_args.append("--disable_fabric")
    return cmd_args


def velocity_command_args(args: argparse.Namespace) -> list[str]:
    if args.wander:
        return ["--wander"]
    return [
        "--command_x",
        str(args.command_x),
        "--command_y",
        str(args.command_y),
        "--command_yaw",
        str(args.command_yaw),
    ]


def command_for(method: str, args: argparse.Namespace) -> tuple[list[str], Path]:
    python = sys.executable
    app_args = common_app_args(args)

    if method == "ppo":
        # skrl PPO interprets max_iterations as iterations, and each iteration uses rollouts env steps.
        max_iterations = max(1, math.ceil(args.budget_steps / PPO_ROLLOUTS))
        return (
            [
                python,
                "scripts/skrl/train.py",
                "--task",
                "Flat-Unitree-Go2-train-v0",
                "--algorithm",
                "PPO",
                "--num_envs",
                str(args.num_envs),
                "--seed",
                str(args.seed),
                "--max_iterations",
                str(max_iterations),
                *velocity_command_args(args),
                *app_args,
            ],
            PROJECT_ROOT / "logs" / "skrl" / "go2_flat_ppo",
        )

    if method == "sac":
        # The local skrl trainer maps max_iterations directly to timesteps for configs without rollouts.
        return (
            [
                python,
                "scripts/skrl/train.py",
                "--task",
                "SAC-Unitree-Go2-v0",
                "--algorithm",
                "SAC",
                "--num_envs",
                str(args.num_envs),
                "--seed",
                str(args.seed),
                "--max_iterations",
                str(args.budget_steps),
                *velocity_command_args(args),
                *app_args,
            ],
            PROJECT_ROOT / "logs" / "skrl" / "go2_flat_sac",
        )

    if method == "mbrl":
        mbrl_num_envs = args.mbrl_num_envs if args.mbrl_num_envs is not None else args.num_envs
        return (
            [
                python,
                "scripts/mbrl/train.py",
                "--task",
                "Flat-Unitree-Go2-train-v0",
                "--num_envs",
                str(mbrl_num_envs),
                "--seed",
                str(args.seed),
                "--train_steps",
                str(args.budget_steps),
                "--seed_steps",
                str(args.mbrl_seed_steps),
                "--updates_per_step",
                str(args.mbrl_updates_per_step),
                *velocity_command_args(args),
                *app_args,
            ],
            PROJECT_ROOT / "logs" / "mbrl",
        )

    raise ValueError(f"Unsupported method: {method}")


def run_method(method: str, args: argparse.Namespace) -> dict[str, object]:
    command, log_root = command_for(method, args)
    before = latest_dir(log_root)
    print(f"\n[compare] Running {method}: {' '.join(command)}", flush=True)

    if args.dry_run:
        return {"method": method, "command": command, "log_root": str(log_root), "run_dir": None, "returncode": None}

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SOURCE_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    after = latest_dir(log_root)
    run_dir = after if after != before else after
    return {
        "method": method,
        "command": command,
        "log_root": str(log_root),
        "run_dir": str(run_dir) if run_dir else None,
        "returncode": result.returncode,
    }


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manifest_path = args.manifest or PROJECT_ROOT / "logs" / "comparisons" / timestamp / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    for method in args.methods:
        run = run_method(method, args)
        runs.append(run)
        if run["returncode"] not in (0, None):
            print(f"[compare] Stopping after {method} failed with return code {run['returncode']}", flush=True)
            break

    manifest = {
        "created_at": timestamp,
        "project_root": str(PROJECT_ROOT),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "mbrl_num_envs": args.mbrl_num_envs,
        "budget_steps": args.budget_steps,
        "command": {
            "wander": args.wander,
            "x": args.command_x,
            "y": args.command_y,
            "yaw": args.command_yaw,
        },
        "methods": args.methods,
        "runs": runs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n[compare] Wrote manifest: {manifest_path}", flush=True)
    return 1 if any(run["returncode"] not in (0, None) for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())

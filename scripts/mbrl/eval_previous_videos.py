#!/usr/bin/env python3
"""Record comparable flat-play videos for previous MBRL Go2 runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("/home/rml2/anaconda3/envs/isaaclab/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python executable for Isaac Lab.")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs" / "mbrl")
    parser.add_argument("--include-latest", action="store_true", help="Also evaluate the newest run.")
    parser.add_argument("--checkpoint-name", type=str, default="model_best.pt", help="Checkpoint filename under each run's checkpoints directory.")
    parser.add_argument("--command-x", type=float, default=0.4)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--planner-velocity-objective-weight",
        type=float,
        default=None,
        help="Override every checkpoint's planner velocity objective. Default uses each checkpoint setting.",
    )
    return parser.parse_args()


def run_sort_key(run_dir: Path) -> str:
    return run_dir.name


def main() -> int:
    args = parse_args()
    runs = sorted(args.logs_dir.glob("go2_walk_*"), key=run_sort_key)
    runs = [run for run in runs if (run / "checkpoints" / args.checkpoint_name).exists()]
    if not args.include_latest and runs:
        runs = runs[:-1]

    if not runs:
        print("[ERROR] No runs with checkpoints/model_best.pt found.", file=sys.stderr)
        return 1

    python = args.python
    play_script = PROJECT_ROOT / "scripts" / "mbrl" / "play.py"
    failures: list[str] = []

    for run in runs:
        checkpoint = run / "checkpoints" / args.checkpoint_name
        video_dir = run / "videos" / "mismatch_nominal"
        video_dir.mkdir(parents=True, exist_ok=True)
        generated = video_dir / "rl-video-step-0.mp4"
        if generated.exists():
            generated.rename(video_dir / f"preexisting_{generated.stat().st_mtime_ns}_rl-video-step-0.mp4")

        command = [
            str(python),
            str(play_script),
            "--headless",
            "--video",
            "--video_length",
            str(args.video_length),
            "--checkpoint",
            str(checkpoint),
            "--num_envs",
            "1",
            "--num_episodes",
            "1",
            "--max_steps",
            str(args.max_steps),
            "--command_x",
            str(args.command_x),
            "--command_y",
            str(args.command_y),
            "--command_yaw",
            str(args.command_yaw),
        ]
        checkpoint_stem = Path(args.checkpoint_name).stem.replace("model_", "")
        suffix = f"{checkpoint_stem}_default"
        if args.planner_velocity_objective_weight is not None:
            command.extend(["--planner_velocity_objective_weight", str(args.planner_velocity_objective_weight)])
            suffix = f"{checkpoint_stem}_velobj{args.planner_velocity_objective_weight:g}".replace(".", "p")

        print(f"[RUN] {run.name}", flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
        log_path = video_dir / f"flat_best_x{args.command_x:g}_{suffix}_play.log"
        log_path.write_text(result.stdout + result.stderr)

        if result.returncode != 0:
            failures.append(f"{run.name}: play failed with code {result.returncode}; see {log_path}")
            print(f"[FAIL] {run.name}: code {result.returncode}", flush=True)
            continue
        if not generated.exists():
            failures.append(f"{run.name}: play succeeded but {generated} was not created; see {log_path}")
            print(f"[FAIL] {run.name}: missing video", flush=True)
            continue

        out_video = video_dir / f"flat_best_x{args.command_x:g}_{suffix}_eval.mp4"
        if out_video.exists():
            out_video.unlink()
        shutil.move(str(generated), str(out_video))
        print(f"[OK] {out_video}", flush=True)

    if failures:
        print("[SUMMARY] Failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("[SUMMARY] All requested videos recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

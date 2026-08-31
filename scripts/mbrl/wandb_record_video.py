#!/usr/bin/env python3
"""Record an MBRL checkpoint rollout and upload the MP4 to a W&B run."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("/home/rml2/anaconda3/envs/isaaclab/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python executable for Isaac Lab.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to record.")
    parser.add_argument("--project", default="ldm-quad-mbrl", help="W&B project name.")
    parser.add_argument("--entity", default=None, help="Optional W&B entity/team.")
    parser.add_argument("--run_id", default=None, help="Existing W&B run id to upload into. Inferred from checkpoint run dir when omitted.")
    parser.add_argument("--run_name", default=None, help="Optional W&B run name.")
    parser.add_argument("--media_name", default=None, help="W&B media key. Defaults to Videos / <checkpoint>_<command>.")
    parser.add_argument("--step", type=int, default=None, help="Optional W&B step for the video.")
    parser.add_argument("--command_x", type=float, default=0.2)
    parser.add_argument("--command_y", type=float, default=0.0)
    parser.add_argument("--command_yaw", type=float, default=0.0)
    parser.add_argument("--mismatch", default="nominal", choices=["nominal", "low_friction", "compliant", "rough", "slope", "mass", "motor_weakness", "push"])
    parser.add_argument("--video_length", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--upload_only", type=Path, default=None, help="Upload an existing MP4 instead of recording a new one.")
    return parser.parse_args()


def infer_run_id(checkpoint: Path) -> str | None:
    run_dir = checkpoint.expanduser().resolve().parent.parent
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.is_dir():
        return None
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda path: path.stat().st_mtime)
    if not run_dirs:
        return None
    return run_dirs[-1].name.rsplit("-", 1)[-1]


def default_media_name(args: argparse.Namespace) -> str:
    checkpoint_name = args.checkpoint.stem
    command = f"x{args.command_x:g}_y{args.command_y:g}_yaw{args.command_yaw:g}".replace(".", "p").replace("-", "m")
    mismatch = args.mismatch.replace("-", "_")
    return f"Videos / {checkpoint_name}_{command}_{mismatch}"


def record_video(args: argparse.Namespace) -> Path:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    run_dir = checkpoint.parent.parent
    video_dir = run_dir / "videos" / f"mismatch_{args.mismatch}"
    video_dir.mkdir(parents=True, exist_ok=True)
    generated = video_dir / "rl-video-step-0.mp4"
    if generated.exists():
        backup = video_dir / f"preexisting_{generated.stat().st_mtime_ns}_rl-video-step-0.mp4"
        generated.rename(backup)

    command = [
        str(args.python),
        str(PROJECT_ROOT / "scripts" / "mbrl" / "play.py"),
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
        "--mismatch",
        args.mismatch,
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = video_dir / f"wandb_record_{checkpoint.stem}_{timestamp}.log"
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"play.py failed with code {result.returncode}; see {log_path}")
    if not generated.is_file():
        raise FileNotFoundError(f"play.py finished but did not create {generated}; see {log_path}")

    out_video = video_dir / f"{checkpoint.stem}_x{args.command_x:g}_{timestamp}.mp4".replace(".", "p")
    if out_video.exists():
        out_video.unlink()
    shutil.move(str(generated), str(out_video))
    return out_video


def upload_video(args: argparse.Namespace, video_path: Path) -> None:
    import wandb

    run_id = args.run_id or infer_run_id(args.checkpoint)
    if not run_id:
        raise ValueError("Could not infer W&B run id. Pass --run_id explicitly.")

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=run_id,
        name=args.run_name,
        resume="allow",
        config={"uploaded_video": str(video_path)},
    )
    media_name = args.media_name or default_media_name(args)
    payload = {media_name: wandb.Video(str(video_path), fps=args.fps, format="mp4")}
    if args.step is None:
        run.log(payload)
    else:
        run.log(payload, step=args.step)
    run.finish()


def main() -> int:
    args = parse_args()
    video_path = args.upload_only.expanduser().resolve() if args.upload_only else record_video(args)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    print(f"[W&B Video] uploading {video_path}", flush=True)
    upload_video(args, video_path)
    print(f"[W&B Video] uploaded {video_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

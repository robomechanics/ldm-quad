#!/usr/bin/env python
"""Verify the yaw-deadband relabel formula against the ENV's actual reward term.

The Phase B relabel rewrites 1,000,000 stored rewards using

    delta = w * dt * ( exp(-max(|wz - cmd| - d, 0)^2 / std^2) - exp(-(wz - cmd)^2 / std^2) )

If that formula disagrees with what the environment actually computes, every row is silently
wrong and nothing errors. This builds a live env, calls the real reward functions on real sim
states, and compares. Run it BEFORE trusting a relabelled buffer.

  python scripts/mbrl/verify_yaw_deadband_relabel.py --steps 200 --deadband 0.20
"""
import argparse
import os
import sys

# ldm_quad is not installed; play.py/train.py put source/ldm_quad on sys.path the same way.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "source", "ldm_quad")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Flat-Unitree-Go2-train-v0")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--deadband", type=float, default=0.20)
ap.add_argument("--std", type=float, default=0.28)
ap.add_argument("--weight", type=float, default=8.0)
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--tol", type=float, default=1e-5)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import ldm_quad.tasks  # noqa: F401,E402
import ldm_quad.tasks.manager_based.ldm_quad.mdp as ldm_mdp  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg)
env.reset()
u = env.unwrapped
dt = float(u.step_dt)
print(f"[verify] task={args.task} num_envs={args.num_envs} step_dt={dt}")

max_err = 0.0
n = 0
for i in range(args.steps):
    act = torch.zeros((args.num_envs, u.action_space.shape[-1]), device=u.device).uniform_(-0.3, 0.3)
    env.step(act)
    # the env's OWN functions, on the live state
    stock = ldm_mdp.track_ang_vel_z_exp(u, std=args.std, command_name="base_velocity")
    dead = ldm_mdp.track_ang_vel_z_exp_deadband(
        u, std=args.std, deadband=args.deadband, command_name="base_velocity"
    )
    env_delta = args.weight * dt * (dead - stock)
    # the RELABEL formula, from the same quantities the relabel reads
    wz = u.scene["robot"].data.root_ang_vel_b[:, 2]
    cmd = u.command_manager.get_command("base_velocity")[:, 2]
    err = (cmd - wz).abs()
    f_old = torch.exp(-err.square() / args.std**2)
    f_new = torch.exp(-(err - args.deadband).clamp_min(0.0).square() / args.std**2)
    formula_delta = args.weight * dt * (f_new - f_old)

    e = float((env_delta - formula_delta).abs().max())
    max_err = max(max_err, e)
    n += args.num_envs
    if i < 3:
        print(f"  step {i}: env_delta[0]={float(env_delta[0]):+.6f} "
              f"formula_delta[0]={float(formula_delta[0]):+.6f} max_abs_err={e:.2e}")

print(f"\n[verify] compared {n} transitions")
print(f"[verify] MAX ABS ERROR = {max_err:.3e}   tol = {args.tol:.1e}")
print("[verify] PASS" if max_err <= args.tol else "[verify] *** FAIL -- relabel formula does NOT match the env ***")
env.close()
app.close()
raise SystemExit(0 if max_err <= args.tol else 1)

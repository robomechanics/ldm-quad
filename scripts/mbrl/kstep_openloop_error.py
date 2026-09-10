#!/usr/bin/env python
"""Open-loop k-step prediction error of the world model, per head, on real buffer sequences.

WHY
---
Eval traces only give ONE-step error at wall-clock spacing. The quantity that actually governs
MPPI's choice is error accumulated along an H-step AUTOREGRESSIVE rollout (z_{t+1} = next(z_t, a)),
where errors compound and correlate. Approximating it from a 5-step-spaced one-step series gives an
upper bound on how much horizon integration averages error down; this measures it directly.

Reports, for k = 1..H:
  - physical-head error on (vx, vy, vyaw): RMSE at step k
  - reward-head absolute error at step k
  - correlation of per-rollout error between step k and k+1 ALONG the rollout (the number that
    decides whether integration over H helps)

CPU-only, reads artifacts already on disk, so it does not contend with a running eval or training.

  python scripts/mbrl/kstep_openloop_error.py --checkpoint <model.pt> --replay <replay_latest.pt>
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import math
import os
import statistics as st
import sys

import torch


def _load(name: str, relpath: str):
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", relpath))
    spec = ilu.spec_from_file_location(name, path)
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod          # @dataclass resolves its module via sys.modules
    spec.loader.exec_module(mod)
    return mod


_wm = _load("_wm_k", "source/ldm_quad/ldm_quad/mbrl/world_model.py")
_rp = _load("_rp_k", "source/ldm_quad/ldm_quad/mbrl/replay.py")


def build_model(ck_args: dict, obs_dim: int, action_dim: int, n_phys: int):
    g = ck_args.get
    return _wm.LatentWorldModel(
        obs_dim=obs_dim, action_dim=action_dim,
        latent_dim=g("latent_dim", 256), hidden_dim=g("hidden_dim", 512),
        depth=g("model_depth", 3), num_q=g("num_q", 5), discount=g("discount", 0.99),
        tau=g("target_tau", 0.01), rho=g("rho", 0.5), entropy_coef=g("entropy_coef", 1e-4),
        num_bins=g("num_bins", 101), vmin=g("vmin", -10.0), vmax=g("vmax", 10.0),
        simnorm_dim=g("simnorm_dim", 8), q_dropout=g("q_dropout", 0.01),
        physical_feature_indices=tuple(range(n_phys)) if n_phys else (),
        command_indices=[int(x) for x in str(g("command_skip_indices", "") or "").split(",") if x.strip()],
        loss_weights=_wm.WorldModelLossWeights(),
    )


def corr(a, b):
    if len(a) < 4: return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    da, db = st.stdev(a), st.stdev(b)
    if da == 0 or db == 0: return float("nan")
    return sum((x-ma)*(y-mb) for x, y in zip(a, b)) / ((len(a)-1)*da*db)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"]
    ck_args = dict(ck.get("args", {}) or {})
    obs_dim = sd["encoder.0.0.weight"].shape[1]
    latent_dim = sd["continue_head.0.weight"].shape[1]
    action_dim = sd["reward_head.0.weight"].shape[1] - latent_dim
    _ph = [int(k.split(".")[1]) for k in sd if k.startswith("physical_head.") and k.endswith(".bias")]
    n_phys = int(sd[f"physical_head.{max(_ph)}.bias"].shape[0]) if _ph else 0
    ck_args["latent_dim"] = latent_dim
    phys_idx = [int(x) for x in str(ck_args.get("latent_physical_indices", "0,1,5")).split(",")]

    model = build_model(ck_args, obs_dim, action_dim, n_phys)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"model: obs_dim={obs_dim} latent={latent_dim} action={action_dim} "
          f"physical={n_phys} idx={phys_idx} command_dim={model.command_dim}")

    buf = _rp.ReplayBuffer(capacity=1, obs_dim=obs_dim, action_dim=action_dim, device="cpu")
    state = torch.load(a.replay, map_location="cpu", weights_only=False)
    if "replay" in state: state = state["replay"]
    cap = int(state.get("capacity", 0))
    buf = _rp.ReplayBuffer(capacity=cap, obs_dim=obs_dim, action_dim=action_dim, device="cpu")
    buf.load_state_dict(state)
    print(f"replay: size={len(buf)} stride={getattr(buf,'_last_batch_size',None)} "
          f"valid_seq(H={a.horizon})={buf.valid_sequence_count(a.horizon)}")

    batch = buf.sample_sequences(a.batch, a.horizon, device="cpu")
    obs = batch["obs"]            # [H+1, B, obs_dim]
    actions = batch["actions"]    # [H,   B, action_dim]
    rewards = batch["rewards"]    # [H,   B, 1]
    pidx = torch.as_tensor(phys_idx, dtype=torch.long)

    phys_err, rew_err = [], []    # per k: list over batch of per-sequence error
    # SIGNED error too: "the reward head over-predicts" was an inference from the direction of the
    # target gap, not a measurement, because model_reward_abs_error is unsigned.
    rew_signed, rew_tgt = [], []
    with torch.no_grad():
        z = model.encode(obs[0])
        for k in range(a.horizon):
            r_pred = model.reward(z, actions[k]).squeeze(-1)
            _signed = r_pred - rewards[k].squeeze(-1)
            rew_signed.append(_signed)
            rew_tgt.append(rewards[k].squeeze(-1))
            rew_err.append(_signed.abs())
            z = model.next(z, actions[k])
            if n_phys:
                tgt = obs[k+1].index_select(-1, pidx)
                phys_err.append((model.physical_features(z) - tgt).square().mean(dim=-1))

    print()
    print(f"{'k':>3} {'phys RMSE':>11} {'reward |err|':>13}   (open-loop from step 0, n={a.batch})")
    for k in range(a.horizon):
        pr = math.sqrt(float(phys_err[k].mean())) if phys_err else float("nan")
        re = float(rew_err[k].mean())
        print(f"{k+1:>3} {pr:>11.4f} {re:>13.4f}")

    print()
    print(f"{'k':>3} {'reward SIGNED':>14} {'target mean':>12} {'bias/target':>12}")
    for k in range(a.horizon):
        sg = float(rew_signed[k].mean()); tg = float(rew_tgt[k].mean())
        print(f"{k+1:>3} {sg:>+14.4f} {tg:>12.4f} {sg/tg*100:>11.1f}%")
    print()
    print("correlation of per-sequence error between consecutive rollout steps")
    print("(this is the number that decides whether integrating over H averages error down):")
    for name, series in (("physical", phys_err), ("reward", rew_err)):
        if not series: continue
        cs = [corr(series[k].tolist(), series[k+1].tolist()) for k in range(len(series)-1)]
        print(f"  {name:>8}: " + "  ".join(f"k{k+1}->{k+2} {c:+.3f}" for k, c in enumerate(cs)))
        mean_rho = st.mean([c for c in cs if c == c])
        H = a.horizon
        vf = sum(mean_rho**abs(i-j) for i in range(H) for j in range(H))/H**2
        print(f"  {'':8}  mean rho={mean_rho:+.3f} -> effective n over H={H} is {1/vf:.1f} "
              f"(independent 8.0), SNR gain {math.sqrt(1/vf):.2f}x")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""#6a gate: a command-skip model grafted from a no-skip checkpoint must be numerically IDENTICAL.

If this does not print ~0 for every head, the A/B is measuring re-initialisation shock instead
of what the skip-connection learns, and must not be launched. Run on CPU; no IsaacSim needed.

  python scripts/mbrl/verify_command_skip.py --checkpoint <model_*.pt> [--command_indices 9,10,11]
"""
from __future__ import annotations

import argparse
import sys

import torch

# Import world_model.py DIRECTLY by path: ldm_quad/__init__.py pulls in the task registry,
# which imports omni.kit and therefore needs a live IsaacSim. This check is pure torch.
import importlib.util as _ilu  # noqa: E402
import os as _os  # noqa: E402

_WM = _os.path.join(_os.path.dirname(__file__), "..", "..",
                    "source", "ldm_quad", "ldm_quad", "mbrl", "world_model.py")
_spec = _ilu.spec_from_file_location("_wm_standalone", _os.path.normpath(_WM))
_wm = _ilu.module_from_spec(_spec)
sys.modules["_wm_standalone"] = _wm   # @dataclass resolves its module via sys.modules
_spec.loader.exec_module(_wm)
LatentWorldModel = _wm.LatentWorldModel
WorldModelLossWeights = _wm.WorldModelLossWeights
expand_state_dict_for_command_skip = _wm.expand_state_dict_for_command_skip


def build(ck_args: dict, obs_dim: int, action_dim: int, command_indices) -> LatentWorldModel:
    g = ck_args.get
    return LatentWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=g("latent_dim", 256),
        hidden_dim=g("hidden_dim", 512),
        depth=g("model_depth", 3),
        num_q=g("num_q", 5),
        discount=g("discount", 0.99),
        tau=g("target_tau", 0.01),
        rho=g("rho", 0.5),
        entropy_coef=g("entropy_coef", 1e-4),
        bc_coef=g("tdmpc2_bc_coef", 0.0),
        num_bins=g("num_bins", 101),
        vmin=g("vmin", -10.0),
        vmax=g("vmax", 10.0),
        simnorm_dim=g("simnorm_dim", 8),
        q_dropout=g("q_dropout", 0.01),
        physical_feature_indices=g("_physical_indices", ()),
        command_indices=command_indices,
        loss_weights=WorldModelLossWeights(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--command_indices", default="9,10,11")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"]
    ck_args = ck.get("args", {}) or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)

    obs_dim = sd["encoder.0.0.weight"].shape[1]
    latent_dim = sd["continue_head.0.weight"].shape[1]
    action_dim = sd["reward_head.0.weight"].shape[1] - latent_dim
    # output width is the LAST Linear of the head, not the first (that one is hidden_dim)
    _ph = [int(k.split(".")[1]) for k in sd if k.startswith("physical_head.") and k.endswith(".bias")]
    n_phys = int(sd[f"physical_head.{max(_ph)}.bias"].shape[0]) if _ph else 0
    print(f"checkpoint: obs_dim={obs_dim} latent_dim={latent_dim} action_dim={action_dim} "
          f"n_physical={n_phys} num_q={sum(1 for k in sd if k.startswith('q_heads.') and k.endswith('.0.weight'))}")

    ck_args = dict(ck_args)
    ck_args["latent_dim"] = latent_dim
    ck_args["_physical_indices"] = tuple(range(n_phys)) if n_phys else ()

    cmd_idx = tuple(int(x) for x in a.command_indices.split(",") if x != "")
    print(f"command_indices={cmd_idx}")

    old = build(ck_args, obs_dim, action_dim, None)
    old.load_state_dict(sd, strict=True)
    old.eval()

    new = build(ck_args, obs_dim, action_dim, cmd_idx)
    new.load_state_dict(expand_state_dict_for_command_skip(sd, new), strict=True)
    new.eval()

    def compare(dtype: torch.dtype) -> float:
        """Worst max|diff| across every head, at the given precision."""
        o = build(ck_args, obs_dim, action_dim, None).to(dtype)
        o.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
        o.eval()
        n = build(ck_args, obs_dim, action_dim, cmd_idx).to(dtype)
        n.load_state_dict(
            {k: v.to(dtype) for k, v in expand_state_dict_for_command_skip(sd, n).items()}, strict=True
        )
        n.eval()

        torch.manual_seed(0)
        obs = torch.randn(a.batch, obs_dim, dtype=dtype)
        act = (torch.rand(a.batch, action_dim, dtype=dtype) * 2.0 - 1.0)

        worst = 0.0
        with torch.no_grad():
            z_o, z_n = o.encode(obs), n.encode(obs)
            checks = {
                "encode (latent part)": (z_o, z_n[:, :latent_dim]),
                "next   (latent part)": (o.next(z_o, act), n.next(z_n, act)[:, :latent_dim]),
                "reward": (o.reward(z_o, act), n.reward(z_n, act)),
                "reward_logits": (o.reward_logits(z_o, act), n.reward_logits(z_n, act)),
                "continue_logits": (o.continue_logits(z_o), n.continue_logits(z_n)),
                "pi (deterministic)": (o.pi(z_o, deterministic=True), n.pi(z_n, deterministic=True)),
                # return_type="all" keeps every head: Q(return_type="min") draws a RANDOM pair of
                # heads via torch.randperm, so comparing it across two model instances measures
                # the draw, not the graft.
                "Q all heads": (o.Q(z_o, act, return_type="all"), n.Q(z_n, act, return_type="all")),
                "Q all heads (target)": (o.Q(z_o, act, target=True, return_type="all"),
                                         n.Q(z_n, act, target=True, return_type="all")),
            }
            if n_phys:
                checks["physical_features"] = (o.physical_features(z_o), n.physical_features(z_n))
            checks["command carried through next()"] = (
                obs[:, list(cmd_idx)], n.next(z_n, act)[:, latent_dim:]
            )
            for name, (x, y) in checks.items():
                x = x[0] if isinstance(x, tuple) else x
                y = y[0] if isinstance(y, tuple) else y
                d = float((x - y).abs().max())
                worst = max(worst, d)
                print(f"  {name:32s} max|diff| = {d:.3e}")
        return worst

    print("\n--- float32 (what training actually runs) ---")
    w32 = compare(torch.float32)
    print("\n--- float64 (proves the graft is algebraically exact, not merely close) ---")
    w64 = compare(torch.float64)

    new_p = sum(p.numel() for p in build(ck_args, obs_dim, action_dim, cmd_idx).parameters())
    old_p = sum(p.numel() for p in build(ck_args, obs_dim, action_dim, None).parameters())
    print(f"\nparams: {old_p:,} -> {new_p:,}  (+{new_p - old_p:,}) "
          f"= {new_p - old_p} new weights, all zero-initialised")
    print(f"worst float32 {w32:.3e} | worst float64 {w64:.3e}")

    # float32 slack is real: widening a Linear changes the matmul reduction order. float64 is
    # the honest algebraic check -- if the graft were wrong it would fail at BOTH precisions.
    if w64 > 1e-10:
        raise SystemExit(f"FAIL: float64 diff {w64:.3e} -- the graft is WRONG, do not launch #6a.")
    if w32 > 1e-3:
        raise SystemExit(f"FAIL: float32 diff {w32:.3e} exceeds round-off -- investigate.")
    print("\nPASS: grafted model is identical up to float round-off; safe to launch #6a.")


if __name__ == "__main__":
    main()

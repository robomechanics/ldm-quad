#!/usr/bin/env python3
"""Context window length K vs stationary identification (the other side of the Level 3 trade-off).

One fixed sample of sequences (same seed) is drawn with Kmax = max(K) history slots. For each K the
context is computed from the K MOST RECENT slots only -- exactly what ContextController does with
context_len=K (pad_mask marks slots older than the K newest), and what Level 3's rolling96 arm did
beyond the trained 48: the history encoder is a set transformer with no positional embedding,
mean-pooled over valid tokens, so any window length is accepted (K > 48 is outside training).

Per K, two subsets of the same sample:
  full    sequences with >= K valid steps in the window (a full window; n reported per K)
  common  sequences with >= Kmax valid steps, scored at every K (apples-to-apples)
Metrics (eval_context_offline.py's definitions, reused from it):
  probe R^2 of motor gain / foot friction from the context (ridge, held-out env ids; the members'
  contexts concatenated for an ensemble), and the raw-history-mean probe over the SAME K window;
  open-loop physical |error| at k = 1, 4, 8, 16 for the true context vs the null context (= the
  frozen model) and the % reduction; the k=8 reduction per dynamics regime; the ensemble member
  disagreement at k=8; the mean pairwise cosine between contexts of the same episode (jitter proxy).

Usage:
  python scripts/mbrl/eval_window_sweep.py --checkpoint <model_final.pt> \
      --replay heldout=logs/mbrl/heldout_dyn_stageL_s1234/replay.pt --out <prefix>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os

import torch

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_eco", os.path.join(_here, "eval_context_offline.py"))
eco = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eco)

KS_DEFAULT = (8, 16, 24, 32, 48, 64, 96, 128)
OPEN_LOOP_K = (1, 4, 8, 16)
# regimes for this study (gain trained on [0.6, 1.0], friction on [0.25, 1.0] for v2/v3)
REGIMES = (("gain_extrap[0.5,0.6)", 0, 0.0, 0.6), ("gain_low[0.6,0.75)", 0, 0.6, 0.75),
           ("gain_high[0.75,1.0]", 0, 0.75, 9.0), ("fric_low[0.25,0.4)", 1, 0.25, 0.4),
           ("fric_mid[0.4,0.6)", 1, 0.4, 0.6), ("fric_high[0.6,1.0]", 1, 0.6, 9.0))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--replay", nargs="+", required=True, help="TAG=PATH")
    p.add_argument("--ks", type=int, nargs="+", default=list(KS_DEFAULT))
    p.add_argument("--n", type=int, default=16384, help="sequences sampled (before the per-K filters)")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout_frac", type=float, default=0.25)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", required=True)
    return p.parse_args()


def rows(d: dict, keep: torch.Tensor) -> dict:
    return eco.filter_rows(d, keep)


@torch.no_grad()
def contexts_for(model, d: dict, k: int) -> torch.Tensor:
    a, t, m = d["history_actions"][:, -k:], d["history_transitions"][:, -k:], d["history_pad_mask"][:, -k:]
    ens = getattr(model, "context_ensemble", 1) > 1
    n = a.shape[0]
    return torch.cat([model.encode_context(a[s:s + 1024], t[s:s + 1024], m[s:s + 1024]) for s in range(0, n, 1024)],
                     dim=1 if ens else 0)


def feats(c: torch.Tensor) -> torch.Tensor:
    return (c.permute(1, 0, 2).reshape(c.shape[1], -1) if c.dim() == 3 else c).double()


def raw_mean(d: dict, k: int) -> torch.Tensor:
    valid = (~d["history_pad_mask"][:, -k:]).unsqueeze(-1).float()
    den = valid.sum(1).clamp_min(1.0)
    return torch.cat([(d["history_transitions"][:, -k:] * valid).sum(1) / den,
                      (d["history_actions"][:, -k:] * valid).sum(1) / den], dim=-1).double()


def probe(x: torch.Tensor, y: torch.Tensor, env: torch.Tensor, test_envs: torch.Tensor, seed: int) -> dict:
    test = torch.isin(env, test_envs)
    train = ~test
    if test.sum() < 20 or train.sum() < 20:
        return {"motor_gain": math.nan, "foot_friction": math.nan}
    pred, _ = eco.ridge_r2(x, y, train, test, env, seed)
    return dict(zip(("motor_gain", "foot_friction"), eco._r2(pred, y[test]).tolist()))


def within_episode_cosine(c: torch.Tensor, env: torch.Tensor, episode: torch.Tensor) -> tuple[float, int]:
    """Mean cosine over all pairs of sequences from the same (env, episode)."""
    v = torch.nn.functional.normalize(feats(c).float(), dim=-1)
    key = env.long() * 1_000_003 + episode.long()
    total, pairs = 0.0, 0
    for u in key.unique():
        g = v[key == u]
        n = g.shape[0]
        if n < 2:
            continue
        s = g.sum(0)
        total += float(s @ s - n)  # sum over ordered pairs i != j of cos(i, j)
        pairs += n * (n - 1)
    return (total / pairs if pairs else math.nan), pairs // 2


def scores(model, d: dict, c: torch.Tensor, horizon: int) -> dict:
    n = d["env"].numel()
    ol = eco.open_loop(model, d, {"true": c, "null": model.null_context_batch(n).clone()}, horizon)
    return {"true": ol["true"]["phys"], "null": ol["null"]["phys"], "disagree": ol["true"]["disagree"]}


def summarise(sc: dict, sel: torch.Tensor, dyn: torch.Tensor, ens: bool) -> dict:
    t, z = sc["true"][sel], sc["null"][sel]
    out = {"n": int(sel.sum())}
    for k in OPEN_LOOP_K:
        tm, zm = float(t[:, k - 1].mean()), float(z[:, k - 1].mean())
        out[f"k{k}"] = {"true": tm, "null": zm, "reduction_pct": 100 * (1 - tm / zm)}
    reg = {}
    for label, col, lo, hi in REGIMES:
        m = (dyn[sel][:, col] >= lo) & (dyn[sel][:, col] < hi)
        if m.sum() >= 30:
            reg[label] = {"n": int(m.sum()), "reduction_pct_k8": 100 * (1 - float(t[m, 7].mean()) / float(z[m, 7].mean()))}
        else:
            reg[label] = {"n": int(m.sum()), "reduction_pct_k8": math.nan}
    out["regimes_k8"] = reg
    out["disagreement_k8"] = float(sc["disagree"][sel][:, 7].mean()) if ens else math.nan
    return out


def run_buffer(model, args, tag: str, path: str) -> dict:
    kmax = max(args.ks)
    buf = eco.load_buffer(path)
    d = eco.sample(buf, args.n, args.horizon, kmax, args.seed)
    del buf
    keep = ~torch.isnan(d["dyn"]).any(1) & ((~d["history_pad_mask"]).sum(1) > 0)
    d = rows(d, keep)
    nvalid = {k: (~d["history_pad_mask"][:, -k:]).sum(1) for k in args.ks}
    common = nvalid[kmax] >= kmax
    g = torch.Generator().manual_seed(args.seed)
    envs = d["env"].unique()
    test_envs = envs[torch.randperm(envs.numel(), generator=g)[: max(1, round(envs.numel() * args.holdout_frac))]]
    ens = getattr(model, "context_ensemble", 1) > 1
    res = {"replay": path, "sampled": int(keep.sum()), "kmax": kmax, "n_common": int(common.sum()),
           "test_envs": test_envs.tolist(), "per_k": {}}
    for k in args.ks:
        full = nvalid[k] >= k
        sub = rows(d, full)
        c = contexts_for(model, sub, k)
        y = sub["dyn"].double()
        cf, rf = feats(c), raw_mean(sub, k)
        com_in_sub = common[full]  # common rows, indexed within the full-window subset
        sc = scores(model, sub, c, args.horizon)
        e_full = summarise(sc, torch.ones(int(full.sum()), dtype=torch.bool), sub["dyn"], ens)
        e_com = summarise(sc, com_in_sub, sub["dyn"], ens)
        e_full["probe"] = {"context": probe(cf, y, sub["env"], test_envs, args.seed),
                           "raw_history_mean": probe(rf, y, sub["env"], test_envs, args.seed)}
        cm = com_in_sub
        e_com["probe"] = {"context": probe(cf[cm], y[cm], sub["env"][cm], test_envs, args.seed),
                          "raw_history_mean": probe(rf[cm], y[cm], sub["env"][cm], test_envs, args.seed)}
        e_full["cosine_within_episode"], e_full["cosine_pairs"] = within_episode_cosine(c, sub["env"], sub["episode"])
        cc = c[:, cm] if c.dim() == 3 else c[cm]
        e_com["cosine_within_episode"], e_com["cosine_pairs"] = within_episode_cosine(cc, sub["env"][cm], sub["episode"][cm])
        res["per_k"][k] = {"full": e_full, "common": e_com}
        pf, pc = e_full["probe"]["context"], e_com["probe"]["context"]
        print(f"[{tag}] K={k:3d} n={e_full['n']:5d} R2 gain {pf['motor_gain']:+.3f} fric {pf['foot_friction']:+.3f} "
              f"(raw {e_full['probe']['raw_history_mean']['motor_gain']:+.3f}/{e_full['probe']['raw_history_mean']['foot_friction']:+.3f}) "
              f"k8 red {e_full['k8']['reduction_pct']:5.1f}% fric_low {e_full['regimes_k8']['fric_low[0.25,0.4)']['reduction_pct_k8']:5.1f}% "
              f"dis8 {e_full['disagreement_k8']:.4f} cos {e_full['cosine_within_episode']:.3f} | common n={e_com['n']} "
              f"R2 {pc['motor_gain']:+.3f}/{pc['foot_friction']:+.3f} k8 {e_com['k8']['reduction_pct']:5.1f}%", flush=True)
    return res


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, ckpt = eco.build_model(args.checkpoint, "cpu")
    model.eval()
    out = {"checkpoint": args.checkpoint, "env_steps": int(ckpt.get("train_state", {}).get("env_steps", -1))
           if isinstance(ckpt.get("train_state"), dict) else None,
           "ensemble": int(getattr(model, "context_ensemble", 1)), "trained_history_len": int(model.history_len),
           "ks": args.ks, "n_sampled": args.n, "seed": args.seed, "buffers": {}}
    for spec in args.replay:
        tag, path = eco._tagged(spec)
        out["buffers"][tag] = run_buffer(model, args, tag, path)
    with open(args.out + ".json", "w") as f:
        json.dump(out, f, indent=1, default=lambda o: None)
    print(f"wrote {args.out}.json")


if __name__ == "__main__":
    main()

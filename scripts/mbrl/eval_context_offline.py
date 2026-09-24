#!/usr/bin/env python3
"""Level 1 offline evaluation of the SIT history context (torch only, no simulator).

Samples same-env, same-episode sequences from a replay buffer that recorded the true dynamics
parameters (dyn_params = [motor_gain, foot_friction]) and asks two questions of a
history-context checkpoint:

A. Parameter probe: does the context encode the dynamics? Ridge regression context ->
   parameter, fitted on 75% of env ids and scored (R^2) on the held-out 25% of envs. Compared
   against a raw-history-mean baseline (no encoder) and a label-shuffled floor.
B. Open-loop prediction: does the context make the world model predict better? Latent rollout
   under the recorded actions for k = 1..horizon with three context arms on identical
   sequences -- true (own history), null (zero), shuffled (another env's context) -- scored on
   the physical head's (vx, vy, wz) against the observation, plus latent consistency MSE and
   one-step reward error. Binned by motor-gain / friction regime and by history length.
C. Sanity: a stageL graft (all context weights zero) must give identical arms; context norm
   and within-episode vs across-env cosine similarity.

Usage:
  python scripts/mbrl/eval_context_offline.py --checkpoint <ckpt> \
      --replay train=<run>/checkpoints/replay_latest.pt heldout=logs/mbrl/heldout_dyn_stageL_s1234/replay.pt \
      --out logs/mbrl/<run>/level1/<name>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASELINE = os.path.join(ROOT, "logs", "mbrl", "best_walker", "stageL_omni_326k.pt")
HIST_BINS = ((1, 8), (9, 24), (25, 48))
# Training draws gain U(0.6, 1.0) and friction U(0.25, 0.8); the *_extrap bins lie outside it
# (only a wider held-out buffer populates them). Empty bins are skipped.
GAIN_BINS = (("gain_extrap[0.5,0.6)", 0.0, 0.6), ("gain_low[0.6,0.75)", 0.6, 0.75), ("gain_high[0.75,1.0]", 0.75, 9.0))
FRIC_BINS = (("fric_extrap[0.2,0.25)", 0.0, 0.25), ("fric_low[0.25,0.4)", 0.25, 0.4),
             ("fric_high[0.4,0.8]", 0.4, 0.8 + 1e-6), ("fric_extrap(0.8,0.9]", 0.8 + 1e-6, 9.0))
ARMS = ("true", "null", "shuffled")


def _load(name: str):
    path = os.path.join(ROOT, "source", "ldm_quad", "ldm_quad", "mbrl", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_ldmq_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wm = _load("world_model")
rp = _load("replay")
ck = _load("checkpoint")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--replay", required=True, nargs="+",
        help="one or more replay files with dyn_params, each optionally tagged as TAG=PATH "
             "(e.g. train=.../replay_latest.pt heldout=.../replay.pt). The FIRST is the reference for the "
             "cross-buffer probe (ridge fitted there, scored on the others).",
    )
    p.add_argument("--history_len", type=int, default=None, help="default: the checkpoint's history_len")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout_frac", type=float, default=0.25)
    p.add_argument("--control_checkpoint", default=BASELINE, help="context-free ckpt grafted as the zero-context control")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (keep low next to a training run)")
    p.add_argument("--out", required=True, help="output prefix: writes <out>.json and <out>_per_k.csv")
    return p.parse_args()


# ----------------------------------------------------------------------------- model
def _indices(text) -> list[int]:
    return [int(i) for i in str(text or "").split(",") if str(i).strip()]


def build_model(path: str, device: str, graft_context_dim: int = 0, history_len: int = 48,
                graft_components: str = "all"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a, sd = ckpt["args"], ckpt["model"]
    latent = a.get("latent_dim", 128)
    kwargs = dict(
        obs_dim=sd["encoder.0.0.weight"].shape[1], action_dim=sd["q_heads.0.0.weight"].shape[1] - latent,
        latent_dim=latent, hidden_dim=a["hidden_dim"], depth=a["model_depth"], num_q=a.get("num_q", 5),
        discount=a["discount"], tau=a.get("target_tau", 0.01), rho=a.get("rho", 0.5),
        entropy_coef=a.get("entropy_coef", 1e-4), num_bins=a.get("num_bins", 101), vmin=a.get("vmin", -10.0),
        vmax=a.get("vmax", 10.0), simnorm_dim=a.get("simnorm_dim", 8), q_dropout=a.get("q_dropout", 0.01),
        physical_feature_indices=_indices(a.get("latent_physical_indices", "")),
        command_indices=_indices(a.get("command_skip_indices", "")),
    )
    ctx_dim = int(a.get("history_context_dim") or 0) or graft_context_dim
    if ctx_dim:
        kwargs.update(
            context_dim=ctx_dim, history_len=int(a.get("history_len") or history_len),
            history_d_model=a.get("history_d_model", 64), history_nhead=a.get("history_nhead", 4),
            history_layers=a.get("history_layers", 1), history_ff=a.get("history_ff", 256),
            history_dropout=a.get("history_dropout", 0.1),
            context_components=a.get("context_components") or graft_components,
        )
    model = wm.LatentWorldModel(**kwargs)
    if ck.is_context_free_state_dict(sd) and ctx_dim:
        ck.graft_context_free_state_dict(model, sd)
    else:
        model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), ckpt


# ----------------------------------------------------------------------------- data
def sample(buf, n: int, horizon: int, k: int, seed: int) -> dict[str, torch.Tensor]:
    """Same-env, same-episode sequences (the training sampler's validity rules) plus the
    identity of each start so the probe can split by env."""
    stride = max(int(buf._last_batch_size), 1)
    starts = buf._valid_sequence_starts(horizon)
    g = torch.Generator().manual_seed(seed)
    start_t = starts[torch.randint(0, starts.numel(), (n,), generator=g)]
    idx = (start_t.unsqueeze(0) + torch.arange(horizon + 1).unsqueeze(1) * stride) % buf.capacity
    hist = buf._gather_history(start_t, k, stride, "cpu")
    return {
        "obs": buf.obs[idx], "actions": buf.actions[idx[:-1]], "rewards": buf.rewards[idx[:-1]],
        "dyn": buf.dyn_params[start_t], "env": buf.env_ids[start_t], "episode": buf.episode_ids[start_t],
        **hist,
    }


def filter_rows(d: dict, keep: torch.Tensor) -> dict:
    out = {}
    for key, v in d.items():
        out[key] = v[:, keep] if key in ("obs", "actions", "rewards") else v[keep]
    return out


# ----------------------------------------------------------------------------- probe
def _ridge_fit(x, y, alpha):
    xm, xs = x.mean(0), x.std(0).clamp_min(1e-6)
    ym = y.mean(0)
    xz = (x - xm) / xs
    a = xz.T @ xz + alpha * torch.eye(x.shape[1], dtype=x.dtype)
    w = torch.linalg.solve(a, xz.T @ (y - ym))
    return lambda q: ((q - xm) / xs) @ w + ym


def _r2(pred, y):
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot


def ridge_r2(x, y, train, test, env, seed):
    """Alpha from an env-disjoint inner split of the training envs, then refit on all of train."""
    g = torch.Generator().manual_seed(seed + 1)
    tr_envs = env[train].unique()
    inner_val = tr_envs[torch.randperm(tr_envs.numel(), generator=g)[: max(1, tr_envs.numel() // 4)]]
    iv = train & torch.isin(env, inner_val)
    it = train & ~iv
    best = max(
        (10.0 ** e for e in range(-3, 4)),
        key=lambda al: float(_r2(_ridge_fit(x[it], y[it], al)(x[iv]), y[iv]).mean()),
    )
    f = _ridge_fit(x[train], y[train], best)
    return f(x[test]), best


# ----------------------------------------------------------------------------- rollouts
@torch.no_grad()
def open_loop(model, d, contexts: dict[str, torch.Tensor], horizon: int, chunk: int = 1024):
    phys_idx = torch.as_tensor(model.physical_feature_indices, dtype=torch.long)
    lat = model.latent_dim
    n = d["obs"].shape[1]
    out = {arm: {"phys": torch.zeros(n, horizon), "floor": torch.zeros(n, horizon),
                 "latent": torch.zeros(n, horizon), "reward1": torch.zeros(n)} for arm in contexts}
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        obs, act, rew = d["obs"][:, sl], d["actions"][:, sl], d["rewards"][:, sl]
        for arm, ctx_all in contexts.items():
            c = ctx_all[sl]
            z = model.encode(obs[0], context=c)
            out[arm]["reward1"][sl] = (model.reward(z, act[0], context=c).view(-1) - rew[0].view(-1)).abs()
            for k in range(1, horizon + 1):
                z = model.next(z, act[k - 1], context=c)
                tgt = obs[k].index_select(-1, phys_idx)
                zk = model.encode(obs[k], context=c)
                out[arm]["phys"][sl, k - 1] = (model.physical_features(z) - tgt).abs().mean(-1)
                out[arm]["floor"][sl, k - 1] = (model.physical_features(zk) - tgt).abs().mean(-1)
                out[arm]["latent"][sl, k - 1] = ((z[..., :lat] - zk[..., :lat]) ** 2).mean(-1)
    return out


def shuffled_contexts(c, env, seed):
    """Each sequence gets the context of a random sequence from a DIFFERENT env."""
    g = torch.Generator().manual_seed(seed + 2)
    src = torch.randint(0, c.shape[0], (c.shape[0],), generator=g)
    for _ in range(100):
        clash = env[src] == env
        if not clash.any():
            break
        src[clash] = torch.randint(0, c.shape[0], (int(clash.sum()),), generator=g)
    assert not (env[src] == env).any(), "could not find a cross-env context"
    return c[src]


# ----------------------------------------------------------------------------- main
def _tagged(spec: str) -> tuple[str, str]:
    if "=" in spec and not os.path.exists(spec):
        tag, path = spec.split("=", 1)
        return tag, path
    return os.path.basename(os.path.dirname(os.path.abspath(spec))), spec


def load_buffer(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sd = payload["replay"] if "replay" in payload else payload
    buf = rp.ReplayBuffer(int(sd["capacity"]), obs_dim=sd["obs"].shape[1], action_dim=sd["actions"].shape[1])
    buf.load_state_dict(sd)
    return buf


def regime_groups(gain, fric, n_valid) -> dict[str, torch.Tensor]:
    groups = {"all": torch.ones(gain.numel(), dtype=torch.bool)}
    for label, lo, hi in GAIN_BINS:
        groups[label] = (gain >= lo) & (gain < hi)
    for label, lo, hi in FRIC_BINS:
        groups[label] = (fric >= lo) & (fric < hi)
    for lo, hi in HIST_BINS:
        groups[f"hist_{lo}-{hi}"] = (n_valid >= lo) & (n_valid <= hi)
    return {k: m for k, m in groups.items() if m.any()}


def evaluate_buffer(model, args, path: str, k_len: int) -> dict:
    buf = load_buffer(path)
    d = sample(buf, args.batches * args.batch_size, args.horizon, k_len, args.seed)
    del buf
    n_valid = (~d["history_pad_mask"]).sum(1)
    keep = (n_valid > 0) & ~torch.isnan(d["dyn"]).any(1)
    dropped = int((~keep).sum())
    d = filter_rows(d, keep)
    n_valid = n_valid[keep]
    n = d["env"].numel()
    gain, fric = d["dyn"][:, 0], d["dyn"][:, 1]
    with torch.no_grad():
        c = torch.cat([
            model.encode_context(d["history_actions"][s:s + 1024], d["history_transitions"][s:s + 1024],
                                 d["history_pad_mask"][s:s + 1024])
            for s in range(0, n, 1024)
        ])
    null = model.history_encoder.null().expand(n, -1).clone()
    c_shuf = shuffled_contexts(c, d["env"], args.seed)

    # ---- A: probe, split by env id within this buffer
    g = torch.Generator().manual_seed(args.seed)
    envs = d["env"].unique()
    test_envs = envs[torch.randperm(envs.numel(), generator=g)[: max(1, round(envs.numel() * args.holdout_frac))]]
    test = torch.isin(d["env"], test_envs)
    train = ~test
    y = d["dyn"].double()
    valid = (~d["history_pad_mask"]).unsqueeze(-1).float()
    raw = torch.cat([
        (d["history_transitions"] * valid).sum(1) / valid.sum(1),
        (d["history_actions"] * valid).sum(1) / valid.sum(1),
    ], dim=-1).double()
    y_perm = y.clone()
    y_perm[train] = y[train][torch.randperm(int(train.sum()), generator=g)]
    probe = {}
    for name, x, yy in (("context", c.double(), y), ("raw_history_mean", raw, y), ("shuffled_labels", c.double(), y_perm)):
        pred, alpha = ridge_r2(x, yy, train, test, d["env"], args.seed)
        entry = {"alpha": alpha, "overall": dict(zip(("motor_gain", "foot_friction"), _r2(pred, y[test]).tolist()))}
        for lo, hi in HIST_BINS:
            m = (n_valid[test] >= lo) & (n_valid[test] <= hi)
            if m.sum() > 10:
                entry[f"hist_{lo}-{hi}"] = dict(zip(("motor_gain", "foot_friction"), _r2(pred[m], y[test][m]).tolist()))
        probe[name] = entry

    # ---- B: open loop, three arms on identical sequences
    ol = open_loop(model, d, {"true": c, "null": null, "shuffled": c_shuf}, args.horizon)
    per_group = {}
    for gname, m in regime_groups(gain, fric, n_valid).items():
        per_group[gname] = {
            "n": int(m.sum()),
            **{arm: {"phys_per_k": ol[arm]["phys"][m].mean(0).tolist(),
                     "floor_per_k": ol[arm]["floor"][m].mean(0).tolist(),
                     "latent_per_k": ol[arm]["latent"][m].mean(0).tolist(),
                     "reward_abs_err_k1": float(ol[arm]["reward1"][m].mean())} for arm in ARMS},
        }

    # ---- C2: context statistics
    sub = torch.arange(min(n, 2048))
    cn = torch.nn.functional.normalize(c[sub], dim=-1)
    cos = cn @ cn.T
    same = (d["env"][sub].unsqueeze(0) == d["env"][sub].unsqueeze(1)) & (d["episode"][sub].unsqueeze(0) == d["episode"][sub].unsqueeze(1))
    same.fill_diagonal_(False)
    diff_env = d["env"][sub].unsqueeze(0) != d["env"][sub].unsqueeze(1)
    ctx_stats = {
        "norm_mean": float(c.norm(dim=-1).mean()),
        "cos_same_env_episode": float(cos[same].mean()) if same.any() else float("nan"),
        "cos_different_env": float(cos[diff_env].mean()),
        "null_norm": float(model.history_encoder.null().norm()),
    }

    allp = per_group["all"]
    true_k = torch.tensor(allp["true"]["phys_per_k"])
    beats_null = bool((true_k < torch.tensor(allp["null"]["phys_per_k"])).all())
    beats_shuf = bool((true_k < torch.tensor(allp["shuffled"]["phys_per_k"])).all())
    r2_gain = probe["context"]["overall"]["motor_gain"]
    r2_gain_raw = probe["raw_history_mean"]["overall"]["motor_gain"]
    result = {
        "replay": path, "sequences": n, "dropped_fully_padded_or_nan": dropped,
        "dyn_range": {"motor_gain": [float(gain.min()), float(gain.max())], "foot_friction": [float(fric.min()), float(fric.max())]},
        "test_envs": test_envs.tolist(), "probe_r2": probe, "open_loop": per_group, "context_stats": ctx_stats,
        "pass": {"true_lt_null_all_k": beats_null, "true_lt_shuffled_all_k": beats_shuf, "r2_gain": r2_gain,
                 "r2_gain_raw_baseline": r2_gain_raw,
                 "passed": beats_null and beats_shuf and r2_gain > 0.1 and r2_gain > r2_gain_raw},
    }
    # kept for the cross-buffer probe and the control; stripped before writing JSON
    result["_features"] = {"c": c.double(), "raw": raw, "y": y, "gain": gain, "fric": fric, "n_valid": n_valid, "d": d}
    return result


def cross_buffer_probe(ref: dict, other: dict) -> dict:
    """Ridge fitted on the reference buffer (all its sequences), scored on another buffer."""
    out = {}
    fr, fo = ref["_features"], other["_features"]
    for name, key in (("context", "c"), ("raw_history_mean", "raw")):
        f = _ridge_fit(fr[key], fr["y"], 1.0)
        pred = f(fo[key])
        entry = {"overall_r2": dict(zip(("motor_gain", "foot_friction"), _r2(pred, fo["y"]).tolist()))}
        err = (pred - fo["y"]).abs()
        for gname, m in regime_groups(fo["gain"], fo["fric"], fo["n_valid"]).items():
            entry[f"mae_{gname}"] = dict(zip(("motor_gain", "foot_friction"), err[m].mean(0).tolist()))
        out[name] = entry
    return out


def print_buffer(tag: str, r: dict, horizon: int) -> None:
    probe, per_group = r["probe_r2"], r["open_loop"]
    print(f"\n--- buffer [{tag}] {r['replay']}")
    print(f"sequences {r['sequences']} (dropped {r['dropped_fully_padded_or_nan']}), "
          f"gain {r['dyn_range']['motor_gain'][0]:.2f}-{r['dyn_range']['motor_gain'][1]:.2f}, "
          f"friction {r['dyn_range']['foot_friction'][0]:.2f}-{r['dyn_range']['foot_friction'][1]:.2f}, "
          f"held-out envs {len(r['test_envs'])}")
    print("A. probe held-out-env R^2  motor_gain  foot_friction   (gain by valid history 1-8 / 9-24 / 25-48)")
    for name, e in probe.items():
        bins = " / ".join(f"{e[f'hist_{lo}-{hi}']['motor_gain']:+.2f}" if f"hist_{lo}-{hi}" in e else "  -  " for lo, hi in HIST_BINS)
        print(f"   {name:18s}      {e['overall']['motor_gain']:+.3f}      {e['overall']['foot_friction']:+.3f}      ({bins})")
    ks = [1, 2, 4, 8, 12, 16] if horizon >= 16 else list(range(1, horizon + 1))
    print("B. open-loop physical |err| (vx,vy,wz)   " + "  ".join(f"k={k:<3d}" for k in ks))
    for gname, gv in per_group.items():
        print(f"   {gname} (n={gv['n']})")
        for arm in ARMS:
            print(f"     {arm:9s}" + " " * 26 + "  ".join(f"{gv[arm]['phys_per_k'][k - 1]:.4f}" for k in ks))
    allp = per_group["all"]
    print("   encoder floor (true)" + " " * 16 + "  ".join(f"{allp['true']['floor_per_k'][k - 1]:.4f}" for k in ks))
    print("   latent consistency MSE, all")
    for arm in ARMS:
        print(f"     {arm:9s}" + " " * 26 + "  ".join(f"{allp[arm]['latent_per_k'][k - 1]:.5f}" for k in ks))
    print("   reward |err| k=1: " + "  ".join(f"{a} {allp[a]['reward_abs_err_k1']:.4f}" for a in ARMS))
    cs = r["context_stats"]
    print(f"C. context norm {cs['norm_mean']:.3f}  null norm {cs['null_norm']:.3f}  cos same env+episode "
          f"{cs['cos_same_env_episode']:.3f} vs different env {cs['cos_different_env']:.3f}")
    pr = r["pass"]
    print(f"PASS RULE [{tag}]: true<null all k={pr['true_lt_null_all_k']}  true<shuffled all k={pr['true_lt_shuffled_all_k']}  "
          f"R2 gain {pr['r2_gain']:+.3f} (>0.1 and > raw {pr['r2_gain_raw_baseline']:+.3f})  => {'PASS' if pr['passed'] else 'FAIL'}")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, ckpt = build_model(args.checkpoint, args.device)
    if model.context_dim == 0:
        raise SystemExit("checkpoint has no history encoder")
    k_len = args.history_len or model.history_len
    buffers = {}
    for spec in args.replay:
        tag, path = _tagged(spec)
        buffers[tag] = evaluate_buffer(model, args, path, k_len)
    tags = list(buffers)
    ref = buffers[tags[0]]
    cross = {t: cross_buffer_probe(ref, buffers[t]) for t in tags[1:]}

    # ---- C1: zero-context control on the reference buffer: the three arms must coincide
    control = None
    if args.control_checkpoint and os.path.exists(args.control_checkpoint):
        # same context structure as the evaluated model (dynamics_only checkpoints have fewer projections)
        cm, _ = build_model(args.control_checkpoint, args.device, graft_context_dim=model.context_dim, history_len=k_len,
                            graft_components=model.context_components)
        d = ref["_features"]["d"]
        n = d["env"].numel()
        m = torch.arange(n) < min(n, 2048)
        dsub = filter_rows(d, m)
        with torch.no_grad():
            csub = cm.encode_context(dsub["history_actions"], dsub["history_transitions"], dsub["history_pad_mask"])
        olc = open_loop(cm, dsub, {"true": csub, "null": cm.history_encoder.null().expand(int(m.sum()), -1).clone(),
                                   "shuffled": shuffled_contexts(csub, dsub["env"], args.seed)}, args.horizon)
        maxdiff = max(float((olc[a]["phys"] - olc["true"]["phys"]).abs().max()) for a in ARMS)
        control = {"checkpoint": args.control_checkpoint, "buffer": tags[0], "n": int(m.sum()),
                   "arms_max_abs_diff": maxdiff, "identical": maxdiff == 0.0}

    # ---- context term vs the base pre-activation it is added to (is W*c starting to dominate?)
    with torch.no_grad():
        d = ref["_features"]["d"]
        n = d["env"].numel()
        m2 = torch.arange(n) < min(n, 2048)
        obs0, a0, c2 = d["obs"][0][m2], d["actions"][0][m2], ref["_features"]["c"][m2].float()
        z = model.encode(obs0, context=c2)
        za = torch.cat([z, a0], -1)
        term_ratio = {}
        for name, seq, x in (("encoder", model.encoder, obs0), ("dynamics", model.dynamics, za),
                             ("reward_head", model.reward_head, za), ("policy_head", model.policy_head, z),
                             ("q_heads.0", model.q_heads[0], za)):
            inner = seq[0] if isinstance(seq[0], torch.nn.Sequential) else seq  # latent_mlp nests the MLP
            if inner.context_weight is None:
                continue  # not conditioned in this checkpoint (context_components=dynamics_only)
            base = inner[0](x).norm(dim=-1).mean()
            term = torch.nn.functional.linear(c2, inner.context_weight).norm(dim=-1).mean()
            term_ratio[name] = float(term / base)
    sdm = ckpt["model"]
    weights = {k: float(v.norm()) for k, v in sdm.items() if k.endswith("context_weight") and not k.startswith(("target_", "detach_"))}
    weights["history_encoder.null_context_abs_max"] = float(sdm["history_encoder.null_context"].abs().max()) \
        if "history_encoder.null_context" in sdm else float("nan")
    total_w = sum(v ** 2 for k, v in weights.items() if k.endswith("context_weight")) ** 0.5

    for r in buffers.values():
        r.pop("_features")
    result = {
        "checkpoint": args.checkpoint, "env_steps": ckpt.get("train_state", {}).get("env_steps"),
        "horizon": args.horizon, "history_len": k_len, "buffers": buffers, "cross_buffer_probe": cross,
        "control": control, "context_weight_norms": weights, "context_weight_total_norm": total_w,
        "context_term_over_base_preact": term_ratio,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(f"{args.out}.json", "w") as f:
        json.dump(result, f, indent=1)
    with open(f"{args.out}_per_k.csv", "w") as f:
        f.write("buffer,group,n,k," + ",".join(f"{a}_{m}" for a in ARMS for m in ("phys", "floor", "latent")) + "\n")
        for tag, r in buffers.items():
            for gname, gv in r["open_loop"].items():
                for k in range(args.horizon):
                    vals = [gv[a][f"{m}_per_k"][k] for a in ARMS for m in ("phys", "floor", "latent")]
                    f.write(f"{tag},{gname},{gv['n']},{k + 1}," + ",".join(f"{v:.6f}" for v in vals) + "\n")

    print(f"\n=== Level 1 context eval: {os.path.basename(args.checkpoint)} (env_steps {result['env_steps']}), "
          f"horizon {args.horizon}, K {k_len} ===")
    for tag, r in buffers.items():
        print_buffer(tag, r, args.horizon)
    for tag, cb in cross.items():
        print(f"\n--- cross-buffer probe: ridge fitted on [{tags[0]}], scored on [{tag}]")
        for name, e in cb.items():
            print(f"   {name:18s} R^2 gain {e['overall_r2']['motor_gain']:+.3f}  friction {e['overall_r2']['foot_friction']:+.3f}")
            maes = [(k[4:], v) for k, v in e.items() if k.startswith("mae_") and not k.startswith("mae_hist")]
            print("      MAE gain / friction by regime: " + "  ".join(f"{g}={v['motor_gain']:.3f}/{v['foot_friction']:.3f}" for g, v in maes))
    print("\nweights")
    if control:
        print(f"   zero-context control ({os.path.basename(control['checkpoint'])} graft, [{control['buffer']}]): "
              f"arms identical = {control['identical']} (max diff {control['arms_max_abs_diff']:.2e})")
    print("   context_weight norms: " + "  ".join(f"{k.replace('.context_weight', '')}={v:.2f}" for k, v in weights.items() if k.endswith("context_weight")))
    print("   ||W c|| / ||base pre-activation||: " + "  ".join(f"{k}={v:.3f}" for k, v in term_ratio.items()))
    print(f"   total online context_weight norm {total_w:.1f}   null_context |max| = {weights['history_encoder.null_context_abs_max']}")
    print("\nPASS: " + "  ".join(f"[{t}] {'PASS' if r['pass']['passed'] else 'FAIL'}" for t, r in buffers.items()))
    print(f"wrote {args.out}.json and {args.out}_per_k.csv")


if __name__ == "__main__":
    main()

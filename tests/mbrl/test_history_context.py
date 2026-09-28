"""Level 0 wiring tests for the SIT history-context latent world model.

Torch only, no simulator. world_model.py / replay.py / history.py are loaded by file
path because importing the ldm_quad package pulls in ldm_quad.tasks -> Isaac Sim.

Run: /home/rml2/anaconda3/envs/isaaclab/bin/python -m pytest tests/mbrl/test_history_context.py -v
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys

import pytest
import torch
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(name: str):
    path = os.path.join(ROOT, "source", "ldm_quad", "ldm_quad", "mbrl", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_ldmq_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


LatentWorldModel = _load("world_model").LatentWorldModel
ReplayBuffer = _load("replay").ReplayBuffer
RollingHistory = _load("history").RollingHistory
_ckpt = _load("checkpoint")

OBS_DIM = 12
ACTION_DIM = 4
CONTEXT_DIM = 16
HISTORY_LEN = 8
BATCH = 6


def context_weights(model: LatentWorldModel) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.named_parameters() if k.endswith("context_weight")}


def make_model(context_dim: int = CONTEXT_DIM, seed: int = 0, randomize: bool = True) -> LatentWorldModel:
    torch.manual_seed(seed)
    model = LatentWorldModel(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        latent_dim=32,
        hidden_dim=64,
        depth=2,
        num_bins=21,
        simnorm_dim=8,
        context_dim=context_dim,
        history_len=HISTORY_LEN if context_dim > 0 else 0,
        history_d_model=32,
        history_nhead=4,
        history_ff=64,
    )
    if randomize:
        with torch.no_grad():
            # Reward/Q final layers and the context projections are zero-initialised, which
            # would make outputs trivially context-independent. Randomise them so the
            # sensitivity tests measure wiring, not initialisation.
            for head in [model.reward_head, *model.q_heads]:
                final = [m for m in head if isinstance(m, nn.Linear)][-1]
                nn.init.normal_(final.weight, std=0.1)
                nn.init.normal_(final.bias, std=0.1)
            for w in context_weights(model).values():
                nn.init.normal_(w, std=0.5)
            # A nonzero null context so "equals null()" is a real check, not 0 == 0.
            if model.history_encoder is not None:
                model.history_encoder.null_context.normal_()
    model.eval()  # dropout off -> deterministic
    return model


def random_history(batch: int = BATCH, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, HISTORY_LEN, ACTION_DIM, generator=g),
        torch.randn(batch, HISTORY_LEN, OBS_DIM, generator=g),
    )


# --------------------------------------------------------------------------- 1
def test_context_sensitivity():
    model = make_model()
    acts_a, trans_a = random_history(seed=1)
    acts_b, trans_b = random_history(seed=2)
    no_pad = torch.zeros(BATCH, HISTORY_LEN, dtype=torch.bool)

    with torch.no_grad():
        ctx_a = model.encode_context(acts_a, trans_a, no_pad)
        ctx_a2 = model.encode_context(acts_a.clone(), trans_a.clone(), no_pad)
        ctx_b = model.encode_context(acts_b, trans_b, no_pad)
        # Same actions, different transitions only -> must still move the context.
        ctx_c = model.encode_context(acts_a, trans_b, no_pad)

    assert ctx_a.shape == (BATCH, CONTEXT_DIM)
    assert torch.equal(ctx_a, ctx_a2), "identical histories must give identical contexts"
    assert (ctx_a - ctx_b).abs().amax(dim=-1).min() > 1e-4, "different histories gave the same context"
    assert (ctx_a - ctx_c).abs().amax(dim=-1).min() > 1e-4, "context ignores transition tokens"

    # Same obs, different contexts -> different latent.
    obs = torch.randn(BATCH, OBS_DIM)
    with torch.no_grad():
        assert not torch.allclose(model.encode(obs, context=ctx_a), model.encode(obs, context=ctx_b))

    # Mixed padding: row 0 fully padded, row 1 fully valid, others partially padded.
    pad = torch.rand(BATCH, HISTORY_LEN, generator=torch.Generator().manual_seed(3)) < 0.5
    pad[0] = True
    pad[1] = False
    with torch.no_grad():
        ctx_m = model.encode_context(acts_a, trans_a, pad)
        null = model.history_encoder.null()
    assert torch.isfinite(ctx_m).all(), "NaN/inf with mixed padding"
    assert torch.allclose(ctx_m[0], null), "fully padded row must equal history_encoder.null()"
    assert null.norm() <= 1.0 + 1e-6
    assert torch.allclose(ctx_m[1], ctx_a[1], atol=1e-6), "no-pad row changed by other rows' padding"
    for ctx in (ctx_a, ctx_b, ctx_c, ctx_m):
        assert (ctx.norm(dim=-1) <= 1.0 + 1e-5).all(), "context escaped the unit ball"

    # Padded slots must not leak: changing tokens at padded positions leaves context unchanged.
    trans_poison = trans_a.clone()
    trans_poison[pad] = 1e3
    with torch.no_grad():
        ctx_p = model.encode_context(acts_a, trans_poison, pad)
    assert torch.allclose(ctx_p, ctx_m, atol=1e-6), "padded history tokens leak into the context"

    # context=None on a context model means the null context.
    with torch.no_grad():
        assert torch.allclose(
            model.encode(obs), model.encode(obs, context=null.expand(BATCH, -1))
        )


# --------------------------------------------------------------------------- 2
def test_prediction_sensitivity():
    model = make_model()
    acts_a, trans_a = random_history(seed=1)
    acts_b, trans_b = random_history(seed=2)
    with torch.no_grad():
        ctx_a = model.encode_context(acts_a, trans_a)
        ctx_b = model.encode_context(acts_b, trans_b)
        z = model.encode(torch.randn(BATCH, OBS_DIM), context=ctx_a)
        a = torch.randn(BATCH, ACTION_DIM).clamp(-1, 1)

        pairs = {
            "next": (model.next(z, a, context=ctx_a), model.next(z, a, context=ctx_b)),
            "reward_logits": (model.reward_logits(z, a, context=ctx_a), model.reward_logits(z, a, context=ctx_b)),
            "Q_logits": (model.Q_logits(z, a, context=ctx_a), model.Q_logits(z, a, context=ctx_b)),
            "policy_mean": (model._policy_stats(z, context=ctx_a)[0], model._policy_stats(z, context=ctx_b)[0]),
        }
    for name, (out_a, out_b) in pairs.items():
        assert torch.isfinite(out_a).all() and torch.isfinite(out_b).all(), name
        assert not torch.allclose(out_a, out_b), f"{name} ignores the context"

    # continue/physical heads read z only (by design): same z -> same output regardless of context.
    with torch.no_grad():
        z_next_a = model.next(z, a, context=ctx_a)
        assert torch.equal(model.continue_logits(z_next_a), model.continue_logits(z_next_a.clone()))

    # context_dim=0: same signatures work with context=None and match the no-kwarg call.
    plain = make_model(context_dim=0)
    assert plain.history_encoder is None
    with torch.no_grad():
        obs = torch.randn(BATCH, OBS_DIM)
        zp = plain.encode(obs, context=None)
        assert torch.equal(zp, plain.encode(obs))
        assert torch.equal(plain.next(zp, a, context=None), plain.next(zp, a))
        assert torch.equal(plain.reward(zp, a, context=None), plain.reward(zp, a))
        assert torch.equal(plain.Q(zp, a, return_type="avg", context=None), plain.Q(zp, a, return_type="avg"))
        assert torch.equal(plain.pi(zp, context=None), plain.pi(zp))
        assert plain.encode_context(acts_a, trans_a) is None


# --------------------------------------------------------------------------- 3
# One first layer per conditioned MLP; the target/detach copies carry their own.
NEW_CONTEXT_KEYS = {
    "encoder.0.context_weight",
    "target_encoder.0.context_weight",
    "dynamics.0.context_weight",
    "reward_head.context_weight",
    "policy_head.context_weight",
    *(f"{p}.{i}.context_weight" for p in ("q_heads", "target_q_heads", "detach_q_heads") for i in range(2)),
}


def warm_start(base_sd: dict, **kwargs) -> LatentWorldModel:
    model = LatentWorldModel(**kwargs)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    assert unexpected == []
    new = {k for k in missing if not k.startswith("history_encoder.")}
    assert new == {k for k in model.state_dict() if k.endswith("context_weight")}, new
    model.sync_detached_qs()
    return model.eval()


def model_outputs(model: LatentWorldModel, obs, a, context):
    with torch.no_grad():
        z = model.encode(obs, context=context)
        return {
            "encode": z,
            "encode_target": model.encode(obs, context=context, target=True),
            "next": model.next(z, a, context=context),
            "reward_logits": model.reward_logits(z, a, context=context),
            "Q_logits": model.Q_logits(z, a, context=context),
            "Q_logits_target": model.Q_logits(z, a, target=True, context=context),
            "policy_mean": model._policy_stats(z, context=context)[0],
        }


def test_warm_start_equivalence():
    base = make_model(context_dim=0, seed=0, randomize=True)
    ctx_model = warm_start(
        base.state_dict(),
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=32, hidden_dim=64, depth=2, num_bins=21,
        simnorm_dim=8, context_dim=CONTEXT_DIM, history_len=HISTORY_LEN,
        history_d_model=32, history_nhead=4, history_ff=64,
    )
    assert {k for k in context_weights(ctx_model)} | {
        k for k in ctx_model.state_dict() if "target_" in k and k.endswith("context_weight")
    } >= {k for k in NEW_CONTEXT_KEYS if not k.startswith(("target_", "detach_"))}
    assert set(ctx_model.state_dict()) - set(base.state_dict()) - {
        k for k in ctx_model.state_dict() if k.startswith("history_encoder.")
    } == NEW_CONTEXT_KEYS

    obs = torch.randn(BATCH, OBS_DIM)
    a = torch.randn(BATCH, ACTION_DIM).clamp(-1, 1)
    ref = model_outputs(base, obs, a, None)
    acts, trans = random_history()
    with torch.no_grad():
        inferred = ctx_model.encode_context(acts, trans)
    contexts = {
        "none(null)": None,
        "random": torch.nn.functional.normalize(torch.randn(BATCH, CONTEXT_DIM), dim=-1),
        "inferred": inferred,
    }
    for name, ctx in contexts.items():
        got = model_outputs(ctx_model, obs, a, ctx)
        for key in ref:
            assert torch.equal(got[key], ref[key]), f"{key} differs from the context-free model under {name} context"

    # The projection is actually wired in: a nonzero weight must change the outputs.
    ctx = contexts["random"]
    for param_name in ["encoder.0.context_weight", "dynamics.0.context_weight", "reward_head.context_weight",
                       "q_heads.0.context_weight", "policy_head.context_weight"]:
        w = dict(ctx_model.named_parameters())[param_name]
        with torch.no_grad():
            w.normal_()
        got = model_outputs(ctx_model, obs, a, ctx)
        with torch.no_grad():
            w.zero_()
        changed = [k for k in ref if not torch.equal(got[k], ref[k])]
        assert changed, f"{param_name} is not wired into any output"


BASELINE_CKPT = os.path.join(ROOT, "logs", "mbrl", "best_walker", "stageL_omni_326k.pt")


def _parse_indices(text: str) -> list[int]:
    return [int(i) for i in str(text).split(",") if str(i).strip()]


def baseline_kwargs(ckpt: dict) -> dict:
    args, sd = ckpt["args"], ckpt["model"]
    obs_dim = sd["encoder.0.0.weight"].shape[1]
    latent_dim = args.get("latent_dim", 128)
    action_dim = sd["q_heads.0.0.weight"].shape[1] - latent_dim
    # Mirrors play.py's reconstruction from checkpoint["args"].
    kwargs = dict(
        obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim, hidden_dim=args["hidden_dim"],
        depth=args["model_depth"], num_q=args.get("num_q", 5), discount=args["discount"],
        tau=args.get("target_tau", 0.01), rho=args.get("rho", 0.5), entropy_coef=args.get("entropy_coef", 1e-4),
        num_bins=args.get("num_bins", 101), vmin=args.get("vmin", -10.0), vmax=args.get("vmax", 10.0),
        simnorm_dim=args.get("simnorm_dim", 8), q_dropout=args.get("q_dropout", 0.01),
        physical_feature_indices=_parse_indices(args.get("latent_physical_indices", "")),
        command_indices=_parse_indices(args.get("command_skip_indices", "")),
    )
    return kwargs


@pytest.fixture(scope="module")
def baseline_ckpt():
    if not os.path.exists(BASELINE_CKPT):
        pytest.skip("baseline checkpoint not present")
    return torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)


def test_warm_start_real_checkpoint(baseline_ckpt):
    sd = baseline_ckpt["model"]
    kwargs = baseline_kwargs(baseline_ckpt)
    obs_dim, action_dim = kwargs["obs_dim"], kwargs["action_dim"]
    base = LatentWorldModel(**kwargs)
    base.load_state_dict(sd, strict=True)
    base.eval()
    ctx_model = warm_start(sd, **kwargs, context_dim=CONTEXT_DIM, history_len=48)
    # enc, dyn, reward, policy, target_enc + q/target_q/detach_q per head
    assert len(context_weights(ctx_model)) == 5 + 3 * kwargs["num_q"]
    assert len(list(ctx_model.policy_parameters())) == len(list(base.policy_parameters())) + 1

    torch.manual_seed(0)
    obs = torch.randn(64, obs_dim)
    a = torch.rand(64, action_dim) * 2 - 1
    ref = model_outputs(base, obs, a, None)
    for ctx in (None, torch.nn.functional.normalize(torch.randn(64, CONTEXT_DIM), dim=-1)):
        got = model_outputs(ctx_model, obs, a, ctx)
        for key in ref:
            assert torch.equal(got[key], ref[key]), key
        with torch.no_grad():
            z = ctx_model.encode(obs, context=ctx)
            assert torch.equal(ctx_model.physical_features(z), base.physical_features(ref["encode"]))
            assert torch.equal(ctx_model.continue_logits(z), base.continue_logits(ref["encode"]))


# --------------------------------------------------------------------------- 4
NUM_ENVS = 4
ROLLOUT_STEPS = 60


def episode_length(env: int) -> int:
    return 9 + 4 * env  # 9, 13, 17, 21: boundaries at known, env-specific steps


def fill_synthetic(capacity: int, device: str = "cpu") -> ReplayBuffer:
    """Vectorised rollout where each row's action encodes (env, episode, step, global t)."""
    buf = ReplayBuffer(capacity, obs_dim=OBS_DIM, action_dim=ACTION_DIM, device=device)
    episode = torch.zeros(NUM_ENVS, dtype=torch.long)
    step = torch.zeros(NUM_ENVS, dtype=torch.long)
    lengths = torch.tensor([episode_length(e) for e in range(NUM_ENVS)])
    for t in range(ROLLOUT_STEPS):
        env = torch.arange(NUM_ENVS)
        ident = torch.stack([env, episode, step, torch.full_like(env, t)], dim=-1).float()
        obs = torch.zeros(NUM_ENVS, OBS_DIM)
        obs[:, :4] = ident
        next_obs = obs.clone()
        next_obs[:, 2] += 1
        next_obs[:, 4] = 100.0 + t  # unique per-step delta to check transition gathering
        done = step + 1 >= lengths
        buf.add_batch(
            obs,
            ident,
            torch.zeros(NUM_ENVS, 1),
            next_obs,
            (~done).float().unsqueeze(-1),
            resets=done,
        )
        step = torch.where(done, torch.zeros_like(step), step + 1)
        episode = torch.where(done, episode + 1, episode)
    return buf


@pytest.mark.parametrize("capacity", [NUM_ENVS * ROLLOUT_STEPS * 2, NUM_ENVS * 37], ids=["no_wrap", "wrapped"])
def test_replay_history_causality(capacity):
    horizon = 3
    buf = fill_synthetic(capacity)
    torch.manual_seed(0)
    batch = buf.sample_sequences(128, horizon, device="cpu", history_len=HISTORY_LEN)

    obs = batch["obs"]  # [H+1, B, dim]
    h_act = batch["history_actions"]  # [B, K, 4] = (env, episode, step, t)
    h_trans = batch["history_transitions"]
    pad = batch["history_pad_mask"]
    assert h_act.shape == (128, HISTORY_LEN, ACTION_DIM)
    assert pad.shape == (128, HISTORY_LEN) and pad.dtype == torch.bool

    oldest_t = ROLLOUT_STEPS - capacity // NUM_ENVS  # earliest global step still in the buffer
    checked_valid = 0
    for b in range(obs.shape[1]):
        env, ep, start_step, start_t = obs[0, b, :4].long().tolist()
        target_ts = set(obs[1:, b, 3].long().tolist())
        for i in range(HISTORY_LEN):
            expected_step = start_step - HISTORY_LEN + i
            expected_t = start_t - HISTORY_LEN + i
            should_be_valid = expected_step >= 0 and expected_t >= oldest_t
            assert bool(pad[b, i]) == (not should_be_valid), (
                f"sample {b} slot {i}: step {expected_step}, t {expected_t}, pad={bool(pad[b, i])}"
            )
            if pad[b, i]:
                continue
            h_env, h_ep, h_step, h_t = h_act[b, i].long().tolist()
            assert h_env == env and h_ep == ep, "history crosses env/episode"
            assert h_step == expected_step and h_step < start_step
            assert h_t < start_t and h_t not in target_ts, "history includes a prediction target"
            # Token is the transition *into* step h_step+1; the latest one lands on obs[0]
            # (same information RollingHistory has at deployment), never beyond it.
            assert h_trans[b, i, 2].item() == 1.0 and h_trans[b, i, 4].item() == 100.0 + h_t
            checked_valid += 1
        # Tokens never come from the sequence rows themselves.
        valid_ts = h_act[b, ~pad[b], 3].long()
        assert (valid_ts < start_t).all()
    assert checked_valid > 0 and pad.any(), "fixture did not exercise both valid and padded slots"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_replay_history_on_cuda_buffer():
    """train.py defaults --replay_device auto -> the training GPU."""
    buf = fill_synthetic(NUM_ENVS * ROLLOUT_STEPS * 2, device="cuda")
    batch = buf.sample_sequences(32, 3, device="cuda", history_len=HISTORY_LEN)
    assert batch["history_pad_mask"].shape == (32, HISTORY_LEN)


# --------------------------------------------------------------------------- 5
def test_rolling_history_reset():
    model = make_model()
    n = 3
    hist = RollingHistory(n, HISTORY_LEN, ACTION_DIM, OBS_DIM, torch.device("cpu"))
    ref = RollingHistory(n, HISTORY_LEN, ACTION_DIM, OBS_DIM, torch.device("cpu"))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        null = model.history_encoder.null()
        assert torch.allclose(hist.context(model), null.expand(n, -1)), "empty history must be null context"

        for _ in range(HISTORY_LEN + 3):  # wrap the ring buffer
            a, tr = torch.randn(n, ACTION_DIM, generator=g), torch.randn(n, OBS_DIM, generator=g)
            hist.append(a, tr)
            ref.append(a, tr)

        a, tr = torch.randn(n, ACTION_DIM, generator=g), torch.randn(n, OBS_DIM, generator=g)
        done = torch.tensor([False, True, False])
        hist.append(a, tr, done)
        ref.append(a, tr)

        assert not hist.valid[1].any(), "done env's window must be fully invalid"
        assert hist.valid[[0, 2]].all()
        ctx, ctx_ref = hist.context(model), ref.context(model)
        assert torch.allclose(ctx[1], null), "done env's context must equal the null context"
        assert torch.allclose(ctx[[0, 2]], ctx_ref[[0, 2]]), "reset leaked into other envs"

        # Next step: the reset env has exactly one valid token, others stay full.
        hist.append(torch.randn(n, ACTION_DIM, generator=g), torch.randn(n, OBS_DIM, generator=g))
        assert hist.valid[1].sum() == 1
        assert hist.valid[[0, 2]].all()
        assert not torch.allclose(hist.context(model)[1], null)


# --------------------------------------------------------------------------- extra
def test_loss_then_policy_loss_end_to_end():
    """Replay batch with history_* keys -> loss() -> policy_loss(), the train.py update order.

    Gradients from the world-model loss must reach the history encoder, and policy_loss must
    pick up the segment context cached by loss() (train.py never passes context explicitly).
    """
    model = make_model()
    model.train()
    buf = fill_synthetic(NUM_ENVS * ROLLOUT_STEPS * 2)
    batch = buf.sample_sequences(32, 3, device="cpu", history_len=HISTORY_LEN)
    batch["obs"] = batch["obs"] / 10.0  # keep the synthetic id-encoded obs in a sane range

    loss, metrics, rollout_zs = model.loss(batch)
    assert torch.isfinite(loss) and all(v == v for v in metrics.values())
    loss.backward()
    grads = [p.grad for p in model.history_encoder.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "no gradient reaches the history encoder"

    cached = model._last_context
    assert cached is not None and cached.shape == (32, CONTEXT_DIM) and not cached.requires_grad
    torch.manual_seed(0)
    pl_cached, _ = model.policy_loss(rollout_zs)
    torch.manual_seed(0)
    pl_explicit, _ = model.policy_loss(rollout_zs, context=cached)
    assert torch.allclose(pl_cached, pl_explicit), "policy_loss did not use the context cached by loss()"


@pytest.mark.parametrize("context_dim", [CONTEXT_DIM, 0])
def test_loss_smoke(context_dim):
    """loss() + policy_loss() on a fake [H+1, B, dim] batch, with and without history."""
    model = make_model(context_dim=context_dim, randomize=False)
    model.train()
    H, B = 3, 16
    batch = {
        "obs": torch.randn(H + 1, B, OBS_DIM),
        "actions": torch.rand(H, B, ACTION_DIM) * 2 - 1,
        "rewards": torch.randn(H, B, 1),
        "continues": torch.ones(H, B, 1),
    }
    if context_dim:
        acts, trans = random_history(batch=B)
        batch |= {"history_actions": acts, "history_transitions": trans,
                  "history_pad_mask": torch.rand(B, HISTORY_LEN) < 0.3}
    loss, metrics, zs = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    pl, _ = model.policy_loss(zs)
    assert torch.isfinite(pl)
    pl.backward()
    if context_dim:
        params = dict(model.named_parameters())
        online = {k: v for k, v in context_weights(model).items() if not k.startswith(("target_", "detach_"))}
        assert all(v.requires_grad and v.grad is not None for v in online.values())
        assert not any(v.requires_grad for k, v in context_weights(model).items() if k not in online)
        # Zero-init projections still get a gradient (dL/dh @ context), so they can learn.
        # Reward/Q heads are the exception at a FRESH init: their final layer is zero, so
        # dL/dh is zero for the whole first layer, base weights included.
        for k, v in online.items():
            base_first = params[k.replace("context_weight", "0.weight")]
            assert (v.grad.abs().sum() > 0) == (base_first.grad.abs().sum() > 0), k
        for k in ("encoder.0.context_weight", "dynamics.0.context_weight", "policy_head.context_weight"):
            assert online[k].grad.abs().sum() > 0, k
    else:
        assert context_weights(model) == {} and model.history_encoder is None


# --------------------------------------------------------------------------- train.py graft
def _train_optimizers(model: LatentWorldModel):
    """Same param groups as train.py builds for --model_type latent."""
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.encoder_parameters()), "lr": 3e-4},
            {"params": list(model.non_encoder_model_parameters()), "lr": 3e-4},
        ],
        lr=3e-4,
    )
    policy_optimizer = torch.optim.Adam(model.policy_parameters(), lr=3e-4, eps=1e-5)
    return optimizer, policy_optimizer


def test_checkpoint_graft_real(baseline_ckpt):
    sd = baseline_ckpt["model"]
    kwargs = baseline_kwargs(baseline_ckpt)
    assert _ckpt.is_context_free_state_dict(sd)
    base = LatentWorldModel(**kwargs)
    base.load_state_dict(sd, strict=True)
    base.eval()
    model = LatentWorldModel(**kwargs, context_dim=CONTEXT_DIM, history_len=48)
    new_keys = _ckpt.graft_context_free_state_dict(model, sd)
    assert set(new_keys) == {k for k in model.state_dict() if _ckpt.is_context_graft_key(k)}
    assert not _ckpt.is_context_free_state_dict(model.state_dict())
    model.eval()
    torch.manual_seed(0)
    obs = torch.randn(32, kwargs["obs_dim"])
    a = torch.rand(32, kwargs["action_dim"]) * 2 - 1
    ref, got = model_outputs(base, obs, a, None), model_outputs(model, obs, a, None)
    for key in ref:
        assert torch.equal(got[key], ref[key]), key


@pytest.mark.parametrize(
    "corrupt",
    ["drop_base_key", "extra_key", "has_context_key"],
)
def test_checkpoint_graft_rejects_mismatch(baseline_ckpt, corrupt):
    sd = dict(baseline_ckpt["model"])
    kwargs = baseline_kwargs(baseline_ckpt)
    if corrupt == "drop_base_key":
        sd.pop("reward_head.0.bias")
    elif corrupt == "extra_key":
        sd["some_new_head.0.weight"] = torch.zeros(1)
    else:  # partially context-aware checkpoint: must not be treated as a clean graft
        sd["policy_head.context_weight"] = torch.zeros(kwargs["hidden_dim"], CONTEXT_DIM)
    model = LatentWorldModel(**kwargs, context_dim=CONTEXT_DIM, history_len=48)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        _ckpt.graft_context_free_state_dict(model, sd)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "model mutated by a rejected graft"


def test_optimizer_remap_real(baseline_ckpt):
    kwargs = baseline_kwargs(baseline_ckpt)
    base = LatentWorldModel(**kwargs)
    base.load_state_dict(baseline_ckpt["model"])
    base_opt, base_popt = _train_optimizers(base)
    base_opt.load_state_dict(baseline_ckpt["optimizer"])
    base_popt.load_state_dict(baseline_ckpt["policy_optimizer"])

    model = LatentWorldModel(**kwargs, context_dim=CONTEXT_DIM, history_len=48)
    new_keys = _ckpt.graft_context_free_state_dict(model, baseline_ckpt["model"])
    opt, popt = _train_optimizers(model)
    # A plain load is exactly what train.py used to do; it must fail, not misassign.
    with pytest.raises(ValueError):
        _train_optimizers(model)[0].load_state_dict(baseline_ckpt["optimizer"])

    named = dict(model.named_parameters())
    added = [named[k] for k in new_keys if k in named]
    kept, fresh, dropped = _ckpt.remap_optimizer_state_for_added_params(opt, baseline_ckpt["optimizer"], added)
    pkept, pfresh, pdropped = _ckpt.remap_optimizer_state_for_added_params(
        popt, baseline_ckpt["policy_optimizer"], added
    )
    assert dropped == pdropped == 0
    assert kept == len(baseline_ckpt["optimizer"]["state"]) and pkept == len(baseline_ckpt["policy_optimizer"]["state"])
    assert pfresh == 1  # policy_head.context_weight
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    assert fresh == sum(1 for p in added if id(p) in in_opt)  # target/detach copies are not optimised
    assert fresh == len(list(model.history_encoder.parameters())) + 3 + kwargs["num_q"]  # enc, dyn, reward + q heads

    # Every base tensor carries exactly the moments the context-free optimizer has for it.
    base_named = dict(base.named_parameters())
    for o_new, o_base in ((opt, base_opt), (popt, base_popt)):
        for group in o_new.param_groups:
            for p in group["params"]:
                name = next(n for n, q in named.items() if q is p)
                if name in new_keys:
                    assert p not in o_new.state or not o_new.state[p]
                    continue
                st, st_ref = o_new.state[p], o_base.state[base_named[name]]
                assert torch.equal(st["exp_avg"], st_ref["exp_avg"]), name
                assert torch.equal(st["exp_avg_sq"], st_ref["exp_avg_sq"]), name
    assert [g["lr"] for g in opt.param_groups] == [g["lr"] for g in base_opt.param_groups]

    # Two real updates: context projections move off zero on step 1; the history encoder
    # only receives gradient once they are nonzero (step 2).
    model.train()
    H, B = 3, 16
    acts, trans = torch.randn(B, 48, kwargs["action_dim"]), torch.randn(B, 48, kwargs["obs_dim"])
    batch = {
        "obs": torch.randn(H + 1, B, kwargs["obs_dim"]),
        "actions": torch.rand(H, B, kwargs["action_dim"]) * 2 - 1,
        "rewards": torch.randn(H, B, 1),
        "continues": torch.ones(H, B, 1),
        "history_actions": acts, "history_transitions": trans,
        "history_pad_mask": torch.zeros(B, 48, dtype=torch.bool),
    }
    hist_grad = []
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = model.loss(batch)
        loss.backward()
        hist_grad.append(sum(float(p.grad.abs().sum()) for p in model.history_encoder.parameters() if p.grad is not None))
        opt.step()
    assert hist_grad[0] == 0.0 and hist_grad[1] > 0.0, hist_grad
    assert named["encoder.0.context_weight"].abs().sum() > 0


# --------------------------------------------------------------------------- adapter mode
def test_adapter_mode_real(baseline_ckpt):
    """--sit_train_mode adapter: graft stageL, 3 updates, base (incl. target/detach) bit-frozen."""
    kwargs = baseline_kwargs(baseline_ckpt)
    model = LatentWorldModel(**kwargs, context_dim=CONTEXT_DIM, history_len=48)
    new_keys = _ckpt.graft_context_free_state_dict(model, baseline_ckpt["model"])
    trainable, frozen = model.freeze_base_for_adapter()
    adapter_ids = {id(p) for p in model.adapter_parameters()} | {id(p) for p in model.policy_adapter_parameters()}
    assert trainable == sum(p.numel() for p in model.parameters() if id(p) in adapter_ids)
    assert all(p.requires_grad == (id(p) in adapter_ids) for p in model.parameters())

    # Same construction and load remap as train.py.
    (enc, nonenc), pol = _ckpt.latent_optimizer_param_groups(model, "adapter")
    opt = torch.optim.Adam([{"params": enc, "lr": 3e-4}, {"params": nonenc, "lr": 3e-4}], lr=3e-4)
    popt = torch.optim.Adam(pol, lr=3e-4, eps=1e-5)
    named = dict(model.named_parameters())
    added = {id(named[k]) for k in new_keys if k in named}
    saved_groups, saved_policy = _ckpt.latent_optimizer_param_groups(model, "full", exclude=added)
    c = _ckpt.remap_optimizer_state(opt, baseline_ckpt["optimizer"], saved_groups)
    pc = _ckpt.remap_optimizer_state(popt, baseline_ckpt["policy_optimizer"], [saved_policy])
    assert c["kept"] == pc["kept"] == 0
    assert c["dropped_absent"] == len(baseline_ckpt["optimizer"]["state"])
    assert pc["dropped_absent"] == len(baseline_ckpt["policy_optimizer"]["state"])
    assert c["fresh"] == len(enc) + len(nonenc) and pc["fresh"] == 1 == len(pol)
    assert not opt.state and not popt.state  # fresh Adam for the adapter

    frozen_ref = {n: p.detach().clone() for n, p in named.items()
                  if not n.endswith("context_weight") and not n.startswith("history_encoder.")}
    adapter_ref = {n: p.detach().clone() for n, p in named.items() if id(p) in adapter_ids}
    target_ctx = [n for n in named if n.startswith(("target_encoder.", "target_q_heads.")) and n.endswith("context_weight")]

    torch.manual_seed(0)
    H, B, K = 3, 16, 48
    batch = {
        "obs": torch.randn(H + 1, B, kwargs["obs_dim"]),
        "actions": torch.rand(H, B, kwargs["action_dim"]) * 2 - 1,
        "planner_mean": torch.full((H, B, kwargs["action_dim"]), float("nan")),
        "planner_std": torch.full((H, B, kwargs["action_dim"]), float("nan")),
        "rewards": torch.randn(H, B, 1),
        "continues": torch.ones(H, B, 1),
        "history_actions": torch.rand(B, K, kwargs["action_dim"]) * 2 - 1,
        "history_transitions": torch.randn(B, K, kwargs["obs_dim"]) * 0.1,
        "history_pad_mask": torch.rand(B, K) < 0.2,
    }
    # Episode-start samples have an all-padded window: they must use the pinned zero null.
    batch["history_pad_mask"][:2] = True
    q_scale_at_load = model.q_scale.value.clone()
    model.train()
    for _ in range(3):  # the train.py update, verbatim order
        loss, _, zs = model.loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model_parameters(), 20.0)
        opt.step()
        model.sync_detached_qs()
        pl, _ = model.policy_loss(zs, planner_mean=batch["planner_mean"], planner_std=batch["planner_std"])
        popt.zero_grad(set_to_none=True)
        pl.backward()
        torch.nn.utils.clip_grad_norm_(model.policy_parameters(), 20.0)
        popt.step()
        model.soft_update_targets()

    named = dict(model.named_parameters())
    changed_base = [n for n, ref in frozen_ref.items() if not torch.equal(named[n], ref)]
    assert changed_base == [], changed_base  # includes target_* and detach_* base tensors
    unmoved = [n for n, ref in adapter_ref.items() if torch.equal(named[n], ref)]
    assert unmoved == [], unmoved
    assert all(named[n].abs().sum() > 0 for n in target_ctx), "target context weights should EMA-track"

    # What adapter mode does and does NOT guarantee for the walker:
    model.eval()
    base = LatentWorldModel(**kwargs)
    base.load_state_dict(baseline_ckpt["model"])
    base.eval()
    obs = torch.randn(256, kwargs["obs_dim"])
    with torch.no_grad():
        ref = base.pi(base.encode(obs), deterministic=True)
        zeroed = LatentWorldModel(**kwargs, context_dim=CONTEXT_DIM, history_len=48)
        zeroed.load_state_dict(model.state_dict())
        for n, p in zeroed.named_parameters():
            if n.endswith("context_weight"):
                p.zero_()
        zeroed.eval()
        # (a) with the projections removed the base network is exactly stageL
        assert torch.equal(zeroed.pi(zeroed.encode(obs), deterministic=True), ref)
        # (b) null_context is a fixed zero buffer, so an empty history IS stageL, even after
        #     training -- the null-context arm is exactly the frozen walker.
        null_act = model.pi(model.encode(obs), deterministic=True)
        # (c) a real history context does change the policy (that is the adapter's job).
        ctx = torch.nn.functional.normalize(torch.randn(256, CONTEXT_DIM), dim=-1)
        ctx_act = model.pi(model.encode(obs, context=ctx), deterministic=True, context=ctx)
    assert torch.equal(model.history_encoder.null_context, torch.zeros(CONTEXT_DIM))
    assert "history_encoder.null_context" not in dict(model.named_parameters())
    assert torch.equal(null_act, ref)
    assert not torch.equal(ctx_act, ref)
    # D2: the actor-loss Q scale is frozen at its loaded value in adapter mode.
    assert model.q_scale.frozen and torch.equal(model.q_scale.value, q_scale_at_load)


# --------------------------------------------------------------------------- dynamics randomisation
_dyn = _load("dynamics_rand")


class _FakeTerm:
    action_dim = ACTION_DIM

    def __init__(self, n):
        self._scale = 0.4  # JointAction stores a float unless the cfg gives per-joint scales


class _FakeView:
    def __init__(self, n, shapes=27):
        self.mat = torch.ones(n, shapes, 3)
        self.mat[..., 2] = 0.0
        self.set_calls = []

    def get_material_properties(self):
        return self.mat.clone()

    def set_material_properties(self, materials, env_ids):
        self.set_calls.append(env_ids.clone())
        self.mat[env_ids] = materials[env_ids]


class _FakeEnv:
    def __init__(self, n):
        self.term, self.view = _FakeTerm(n), _FakeView(n)
        term, view = self.term, self.view

        class _AM:
            def get_term(self, name):
                return term

        class _Robot:
            root_physx_view = view

        self.action_manager = _AM()
        self.scene = {"robot": _Robot()}
        self.unwrapped = self


def test_dynamics_randomizer_off_is_inert():
    env = _FakeEnv(6)
    dr = _dyn.DynamicsRandomizer(6, "cpu")
    dr.attach(env)
    dr.resample(torch.ones(6, dtype=torch.bool))
    assert not dr.enabled and dr.metrics() == {}
    assert env.term._scale == 0.4 and env.view.set_calls == []  # simulator untouched
    assert dr.params().shape == (6, len(_dyn.PARAM_NAMES)) and torch.isnan(dr.params()).all()


def test_dynamics_randomizer_axes():
    n = 16
    env = _FakeEnv(n)
    dr = _dyn.DynamicsRandomizer(n, "cpu", motor_gain_range=(0.6, 1.0), friction_range=(0.25, 0.8), seed=0)
    dr.attach(env)
    p = dr.params()
    g, f = p[:, 0], p[:, 1]
    assert ((g >= 0.6) & (g <= 1.0)).all() and ((f >= 0.25) & (f <= 0.8)).all()
    assert g.unique().numel() > 1 and f.unique().numel() > 1  # per env, not one global draw
    # motor gain lives in the action term: scale = 0.4 * g per env, every joint
    assert torch.allclose(env.term._scale, 0.4 * g.unsqueeze(-1).expand(n, ACTION_DIM))
    # friction written to every shape, static and dynamic, restitution untouched
    assert torch.allclose(env.view.mat[..., 0], f.unsqueeze(-1).expand(n, 27))
    assert torch.allclose(env.view.mat[..., 1], f.unsqueeze(-1).expand(n, 27))
    assert (env.view.mat[..., 2] == 0).all()
    assert torch.allclose(dr.read_back_friction(), f)
    # resample only the done envs
    done = torch.zeros(n, dtype=torch.bool)
    done[[2, 9]] = True
    before = dr.params().clone()
    dr.resample(done)
    after = dr.params()
    assert torch.equal(after[~done], before[~done])
    assert not torch.equal(after[done], before[done])
    assert torch.equal(env.view.set_calls[-1], torch.tensor([2, 9]))
    assert torch.allclose(env.term._scale[:, 0], 0.4 * after[:, 0])
    m = dr.metrics()
    assert list(m) == [f"dyn_{a}_{s}" for a in _dyn.PARAM_NAMES for s in ("mean", "min", "max")]
    # pinned held-out value
    pinned = _dyn.DynamicsRandomizer(n, "cpu", motor_gain_range=(0.7, 0.7))
    pinned.attach(_FakeEnv(n))
    assert torch.allclose(pinned.motor_gain, torch.full((n,), 0.7)) and torch.isnan(pinned.foot_friction).all()


def test_replay_dyn_params():
    buf = ReplayBuffer(NUM_ENVS * ROLLOUT_STEPS * 2, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    step = torch.zeros(NUM_ENVS, dtype=torch.long)
    episode = torch.zeros(NUM_ENVS, dtype=torch.long)
    lengths = torch.tensor([episode_length(e) for e in range(NUM_ENVS)])
    for t in range(ROLLOUT_STEPS):
        env = torch.arange(NUM_ENVS)
        ident = torch.stack([env, episode, step, torch.full_like(env, t)], dim=-1).float()
        obs = torch.zeros(NUM_ENVS, OBS_DIM)
        obs[:, :4] = ident
        done = step + 1 >= lengths
        # parameter = a per-(env, episode) code, constant within an episode
        dyn = torch.stack([env * 10.0 + episode, -(env * 10.0 + episode)], dim=-1)
        buf.add_batch(obs, ident, torch.zeros(NUM_ENVS, 1), obs.clone(), (~done).float().unsqueeze(-1),
                      resets=done, dyn_params=dyn)
        step = torch.where(done, torch.zeros_like(step), step + 1)
        episode = torch.where(done, episode + 1, episode)
    batch = buf.sample_sequences(64, 3, device="cpu", history_len=HISTORY_LEN)
    assert batch["dyn_params"].shape == (64, 2)
    env_id, ep = batch["obs"][0, :, 0], batch["obs"][0, :, 1]
    assert torch.equal(batch["dyn_params"][:, 0], env_id * 10 + ep)  # the START transition's parameters
    # checkpoint round trip, and a legacy buffer without the key loads as NaN
    sd = buf.state_dict()
    fresh = ReplayBuffer(buf.capacity, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    fresh.load_state_dict(sd)
    assert torch.equal(fresh.dyn_params[: buf.size], buf.dyn_params[: buf.size])
    sd.pop("dyn_params")
    fresh.load_state_dict(sd)
    assert torch.isnan(fresh.dyn_params).all()
    # add_batch without dyn_params stores NaN (default-off runs)
    buf.add_batch(torch.zeros(NUM_ENVS, OBS_DIM), torch.zeros(NUM_ENVS, ACTION_DIM), torch.zeros(NUM_ENVS, 1),
                  torch.zeros(NUM_ENVS, OBS_DIM), torch.ones(NUM_ENVS, 1))
    last = (buf.ptr - NUM_ENVS) % buf.capacity
    assert torch.isnan(buf.dyn_params[last:last + NUM_ENVS]).all()


# --------------------------------------------------------------------------- eval context modes
_hist = _load("history")


def _feed(ctrl, steps, n, seed=0, done_at=None):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        done = None
        if done_at is not None and ctrl.t in done_at:
            done = torch.zeros(n, dtype=torch.bool)
            done[done_at[ctrl.t]] = True
        ctrl.append(torch.randn(n, ACTION_DIM, generator=g), torch.randn(n, OBS_DIM, generator=g), done)


def _ctrl(mode="rolling", n=3, **kw):
    return _hist.ContextController(n, HISTORY_LEN, ACTION_DIM, OBS_DIM, torch.device("cpu"), mode=mode, **kw)


def test_context_mode_null_is_zero():
    model = make_model(randomize=False)  # null_context is the pinned zero buffer
    ctrl = _ctrl("null")
    with torch.no_grad():
        _feed(ctrl, HISTORY_LEN + 2, 3)
        c = ctrl.context(model)
    assert torch.equal(c, torch.zeros(3, CONTEXT_DIM))


def test_context_mode_rolling_matches_rolling_history():
    model = make_model()
    ctrl, ref = _ctrl("rolling"), RollingHistory(3, HISTORY_LEN, ACTION_DIM, OBS_DIM, torch.device("cpu"))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _ in range(HISTORY_LEN + 3):
            a, tr = torch.randn(3, ACTION_DIM, generator=g), torch.randn(3, OBS_DIM, generator=g)
            ctrl.append(a, tr)
            ref.append(a, tr)
        assert torch.equal(ctrl.context(model), ref.context(model))


def test_context_mode_frozen_holds_and_resets_to_null():
    model = make_model()
    ctrl = _ctrl("frozen", freeze_step=5)
    with torch.no_grad():
        _feed(ctrl, 5, 3)
        c_freeze = ctrl.context(model)  # t == 5: the frozen value
        _feed(ctrl, 4, 3, seed=7, done_at={7: [1]})  # env 1 resets after the freeze
        c_later = ctrl.context(model)
    assert torch.equal(c_later[[0, 2]], c_freeze[[0, 2]])  # held despite new transitions
    assert torch.allclose(c_later[1], model.history_encoder.null())
    assert ctrl.metrics()["context_drift_mean"] >= 0.0


def test_context_mode_truncated_clears_every_window():
    model = make_model()
    ctrl = _ctrl("truncated", truncate_step=6)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        _feed(ctrl, 6, 3)
        assert not ctrl.history.valid.any()  # cleared at the truncation step
        assert torch.allclose(ctrl.context(model), model.history_encoder.null().expand(3, -1))
        a, tr = torch.randn(3, ACTION_DIM, generator=g), torch.randn(3, OBS_DIM, generator=g)
        ctrl.append(a, tr)
        c = ctrl.context(model)
        only = model.encode_context(a.unsqueeze(1), tr.unsqueeze(1), torch.zeros(3, 1, dtype=torch.bool))
    assert torch.allclose(c, only, atol=1e-6)  # rebuilt from post-truncation transitions only


def test_context_len_pads_older_slots():
    model = make_model()
    k = 3
    ctrl = _ctrl("rolling", context_len=k)
    g = torch.Generator().manual_seed(3)
    acts, trans = [], []
    with torch.no_grad():
        for _ in range(HISTORY_LEN + 2):
            a, tr = torch.randn(3, ACTION_DIM, generator=g), torch.randn(3, OBS_DIM, generator=g)
            acts.append(a)
            trans.append(tr)
            ctrl.append(a, tr)
        assert (~ctrl.pad_mask()).sum(dim=1).tolist() == [k, k, k]
        c = ctrl.context(model)
        recent = model.encode_context(
            torch.stack(acts[-k:], dim=1), torch.stack(trans[-k:], dim=1), torch.zeros(3, k, dtype=torch.bool)
        )
    assert torch.allclose(c, recent, atol=1e-6)  # set encoder: order-free, older slots ignored
    with pytest.raises(ValueError):
        _ctrl("rolling", context_len=HISTORY_LEN + 1)


def test_dyn_switch_pins_axis():
    n = 8
    env = _FakeEnv(n)
    dr = _dyn.DynamicsRandomizer(n, "cpu", motor_gain_range=(1.0, 1.0), friction_range=(1.0, 1.0))
    dr.attach(env)
    dr.switch(friction=0.3)
    assert torch.allclose(dr.read_back_friction(), torch.full((n,), 0.3))
    assert torch.allclose(dr.motor_gain, torch.ones(n))  # untouched axis not redrawn
    dr.resample(torch.tensor([2]))  # a post-switch reset keeps the switched value
    assert torch.allclose(dr.foot_friction, torch.full((n,), 0.3))
    dr.switch(motor_gain=0.6)
    assert torch.allclose(env.term._scale, torch.full((n, ACTION_DIM), 0.4 * 0.6))
    with pytest.raises(ValueError):
        _dyn.DynamicsRandomizer(n, "cpu").switch(friction=0.3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_replay_fed_cuda_tensors_is_exact():
    """Regression: a CPU buffer fed CUDA tensors used non_blocking D2H copies and stored garbage."""
    buf = ReplayBuffer(4096, obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cpu")
    for _ in range(20):
        obs = torch.randn(64, OBS_DIM, device="cuda") @ torch.eye(OBS_DIM, device="cuda")
        act = torch.randn(64, ACTION_DIM, device="cuda")
        dyn = torch.rand(64, 2, device="cuda")
        start = buf.ptr
        buf.add_batch(obs, act, torch.randn(64, 1, device="cuda"), obs * 2, torch.ones(64, 1, device="cuda"),
                      dyn_params=dyn)
        assert torch.equal(buf.obs[start:start + 64], obs.cpu())
        assert torch.equal(buf.actions[start:start + 64], act.cpu())
        assert torch.equal(buf.dyn_params[start:start + 64], dyn.cpu())


def test_restrict_context_dynamics_only():
    model = make_model()  # context weights randomised
    before = {n: p.clone() for n, p in model.named_parameters()}
    zeroed = model.restrict_context("dynamics_only")
    after = dict(model.named_parameters())
    heads = ("reward_head", "q_heads", "target_q_heads", "detach_q_heads", "policy_head")
    expect = {n for n in before if n.endswith("context_weight") and n.split(".")[0] in heads}
    assert set(zeroed) == expect and len(expect) == 2 + 2 * 3  # reward + policy + 2 Q heads x (online, target, detach)
    for n, p in after.items():
        if n in expect:
            assert torch.equal(p, torch.zeros_like(p)), n
        else:
            assert torch.equal(p, before[n]), n  # encoder/dynamics context and every base weight untouched
    # objective is context-free, rollout is context-conditioned
    obs, a = torch.randn(BATCH, OBS_DIM), torch.randn(BATCH, ACTION_DIM).clamp(-1, 1)
    acts, trans = random_history()
    with torch.no_grad():
        c = model.encode_context(acts, trans)
        z = model.encode(obs, context=c)
        assert torch.equal(model.reward_logits(z, a, context=c), model.reward_logits(z, a, context=None))
        assert torch.equal(model.Q_logits(z, a, context=c), model.Q_logits(z, a, context=None))
        assert torch.equal(model._policy_stats(z, context=c)[0], model._policy_stats(z, context=None)[0])
        assert not torch.equal(model.next(z, a, context=c), model.next(z, a, context=None))
        assert not torch.equal(model.encode(obs, context=c), model.encode(obs, context=None))
    assert model.restrict_context("none") == []
    with pytest.raises(ValueError):
        model.restrict_context("reward_only")
    strict = make_model()
    z_names = strict.restrict_context("dynamics_strict")
    assert {n.split(".")[0] for n in z_names} == {"encoder", "target_encoder", "reward_head", "q_heads",
                                                    "target_q_heads", "detach_q_heads", "policy_head"}
    with torch.no_grad():
        c = strict.encode_context(acts, trans)
        assert torch.equal(strict.encode(obs, context=c), strict.encode(obs, context=None))
        z = strict.encode(obs)
        assert not torch.equal(strict.next(z, a, context=c), strict.next(z, a, context=None))


# --------------------------------------------------------------------------- v2: context in dynamics only
def test_dynamics_only_construction_keys():
    m = LatentWorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=32, hidden_dim=64, depth=2,
                         num_bins=21, context_dim=CONTEXT_DIM, history_len=HISTORY_LEN,
                         history_d_model=32, history_ff=64, context_components="dynamics_only")
    ctx_keys = {k for k in m.state_dict() if k.endswith("context_weight")}
    assert ctx_keys == {"dynamics.0.context_weight"}
    assert list(m.policy_adapter_parameters()) == []
    assert {id(p) for p in m.adapter_parameters()} == (
        {id(m.dynamics[0].context_weight)} | {id(p) for p in m.history_encoder.parameters()})
    with pytest.raises(ValueError):
        LatentWorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, context_dim=4, history_len=4,
                         context_components="reward_only")


def test_dynamics_only_warm_start_and_adapter_real(baseline_ckpt):
    kwargs = baseline_kwargs(baseline_ckpt)
    base = LatentWorldModel(**kwargs)
    base.load_state_dict(baseline_ckpt["model"])
    base.eval()
    m = LatentWorldModel(**kwargs, context_dim=CONTEXT_DIM, history_len=48, context_components="dynamics_only")
    new_keys = _ckpt.graft_context_free_state_dict(m, baseline_ckpt["model"])
    assert {k for k in new_keys if not k.startswith("history_encoder.")} == {"dynamics.0.context_weight"}
    m.eval()
    torch.manual_seed(0)
    obs = torch.randn(32, kwargs["obs_dim"])
    a = torch.rand(32, kwargs["action_dim"]) * 2 - 1
    ref = model_outputs(base, obs, a, None)
    for ctx in (None, torch.nn.functional.normalize(torch.randn(32, CONTEXT_DIM), dim=-1)):
        got = model_outputs(m, obs, a, ctx)
        for key in ref:
            assert torch.equal(got[key], ref[key]), key  # warm start is exact

    trainable, frozen = m.freeze_base_for_adapter()
    assert trainable == m.dynamics[0].context_weight.numel() + sum(p.numel() for p in m.history_encoder.parameters())
    (enc, nonenc), pol = _ckpt.latent_optimizer_param_groups(m, "adapter")
    assert pol == [] and nonenc == [m.dynamics[0].context_weight]
    opt = torch.optim.Adam([{"params": enc, "lr": 3e-4}, {"params": nonenc, "lr": 3e-4}], lr=3e-4)
    named = dict(m.named_parameters())
    added = {id(named[k]) for k in new_keys if k in named}
    saved_groups, _ = _ckpt.latent_optimizer_param_groups(m, "full", exclude=added)
    c = _ckpt.remap_optimizer_state(opt, baseline_ckpt["optimizer"], saved_groups)
    assert c["kept"] == 0 and c["fresh"] == len(enc) + len(nonenc)

    frozen_ref = {n: p.detach().clone() for n, p in named.items()
                  if not n.endswith("context_weight") and not n.startswith("history_encoder.")}
    H, B, K = 3, 16, 48
    batch = {
        "obs": torch.randn(H + 1, B, kwargs["obs_dim"]),
        "actions": torch.rand(H, B, kwargs["action_dim"]) * 2 - 1,
        "rewards": torch.randn(H, B, 1), "continues": torch.ones(H, B, 1),
        "history_actions": torch.rand(B, K, kwargs["action_dim"]) * 2 - 1,
        "history_transitions": torch.randn(B, K, kwargs["obs_dim"]) * 0.1,
        "history_pad_mask": torch.zeros(B, K, dtype=torch.bool),
    }
    m.train()
    hist_grad = []
    for _ in range(2):
        loss, _, zs = m.loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        hist_grad.append(sum(float(p.grad.abs().sum()) for p in m.history_encoder.parameters() if p.grad is not None))
        opt.step()
        m.sync_detached_qs()
        with torch.no_grad():
            m.policy_loss(zs)  # train.py's no-optimizer path: metrics only
        m.soft_update_targets()
    assert hist_grad[0] == 0.0 and hist_grad[1] > 0.0, hist_grad  # only via d(z, a, c), once W moves
    named = dict(m.named_parameters())
    assert [n for n, r in frozen_ref.items() if not torch.equal(named[n], r)] == []
    assert m.dynamics[0].context_weight.abs().sum() > 0


# --------------------------------------------------------------------------- Level 3 passive predictor
_pe = _load("predictor_eval")


def _stream(steps, n=4, seed=0, done_at=None):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, OBS_DIM, generator=g)
    for t in range(steps):
        a = torch.rand(n, ACTION_DIM, generator=g) * 2 - 1
        nxt = obs + 0.1 * torch.randn(n, OBS_DIM, generator=g)
        done = torch.zeros(n, dtype=torch.bool)
        if done_at and t in done_at:
            done[done_at[t]] = True
        yield obs, a, nxt, done
        obs = nxt


def test_predictor_identical_context_arms_identical_numbers():
    base = LatentWorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=32, hidden_dim=64, depth=2, num_bins=21,
                            context_dim=CONTEXT_DIM, history_len=HISTORY_LEN, history_d_model=32, history_ff=64,
                            physical_feature_indices=[0, 1, 5]).eval()
    ev = _pe.PredictorEvaluator(base, ["rolling8", "truncated999", "frozen999"], 4, OBS_DIM, ACTION_DIM, torch.device("cpu"))
    for obs, a, nxt, done in _stream(20):
        out = ev.step(obs, a, nxt, done)
        for m in _pe.METRICS:
            v = [out[f"predictor_{arm}_{m}"] for arm in ("rolling8", "truncated999", "frozen999")]
            assert all((x == v[0]) or (x != x and v[0] != v[0]) for x in v), (m, v)
    assert ev.field_names()[0] == "predictor_rolling8_phys1"


def test_predictor_null_arm_equals_context_free_model_and_kstep_rollout():
    kw = dict(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=32, hidden_dim=64, depth=2, num_bins=21,
              physical_feature_indices=[0, 1, 5])
    torch.manual_seed(0)
    free = LatentWorldModel(**kw).eval()
    ctx = LatentWorldModel(**kw, context_dim=CONTEXT_DIM, history_len=HISTORY_LEN, history_d_model=32, history_ff=64,
                           context_components="dynamics_only")
    _ckpt.graft_context_free_state_dict(ctx, free.state_dict())
    with torch.no_grad():
        ctx.dynamics[0].context_weight.normal_()  # a TRAINED projection; the null context is still zero
    ctx.eval()
    ev_free = _pe.PredictorEvaluator(free, ["null"], 4, OBS_DIM, ACTION_DIM, torch.device("cpu"))
    ev_ctx = _pe.PredictorEvaluator(ctx, ["null", "rolling8"], 4, OBS_DIM, ACTION_DIM, torch.device("cpu"))
    hist = []
    for t, (obs, a, nxt, done) in enumerate(_stream(14, done_at={10: [2]})):
        hist.append((obs, a, nxt))
        of, oc = ev_free.step(obs, a, nxt, done), ev_ctx.step(obs, a, nxt, done)
        for m in ("phys1", "lat1", "phys4", "phys8"):
            x, y = of[f"predictor_null_{m}"], oc[f"predictor_null_{m}"]
            assert (x == y) or (x != x and y != y), (t, m, x, y)  # null arm == context-free model, exactly
        # direct play.py formula for one-step physical MSE: done rows (post-reset next_obs) skipped
        with torch.no_grad():
            z1 = free.next(free.encode(obs), a)
            e1 = (free.physical_features(z1) - nxt[:, [0, 1, 5]]).square().mean(-1)
            assert of["predictor_null_phys1"] == float(e1[~done].mean())
            if t == 10:  # env 2 is done here: the all-rows mean would differ
                assert done[2] and of["predictor_null_phys1"] != float(e1.mean())
        if t < 3:
            assert of["predictor_null_phys4"] != of["predictor_null_phys4"]  # NaN until k transitions exist
        if t == 6:  # manual 4-step open-loop rollout from obs_{t-3} on the executed actions
            with torch.no_grad():
                z = free.encode(hist[t - 3][0])
                for j in range(4):
                    z = free.next(z, hist[t - 3 + j][1])
                ref = (free.physical_features(z) - nxt[:, [0, 1, 5]]).square().mean(-1).mean()
            assert abs(of["predictor_null_phys4"] - float(ref)) < 1e-6
        if t == 12:  # env 2 reset at t=10: only 2 transitions in its new episode -> excluded from k=4
            with torch.no_grad():
                z = free.encode(hist[t - 3][0])
                for j in range(4):
                    z = free.next(z, hist[t - 3 + j][1])
                err = (free.physical_features(z) - nxt[:, [0, 1, 5]]).square().mean(-1)
            keep = torch.tensor([True, True, False, True])
            assert abs(of["predictor_null_phys4"] - float(err[keep].mean())) < 1e-6
    assert oc["predictor_rolling8_phys1"] != oc["predictor_null_phys1"]  # a live context changes the prediction


V2_FINAL = os.path.join(ROOT, "logs", "mbrl", "go2_walk_2026-09-24_00-07-35", "checkpoints", "model_final.pt")
HELDOUT = os.path.join(ROOT, "logs", "mbrl", "heldout_dyn_stageL_s1234", "replay.pt")


@pytest.mark.skipif(not all(os.path.exists(p) for p in (V2_FINAL, HELDOUT, BASELINE_CKPT)), reason="artifacts not present")
def test_predictor_on_real_checkpoints_and_trajectory():
    """Real v2 checkpoint + real stageL, rebuilt by build_predictor_model, scored on 30 consecutive
    vectorised steps of the held-out buffer: v2's null arm must equal stageL exactly."""
    sd = torch.load(HELDOUT, map_location="cpu", weights_only=False)["replay"]
    n_env, obs_dim, act_dim = int(sd["last_batch_size"]), sd["obs"].shape[1], sd["actions"].shape[1]
    v2 = _pe.build_predictor_model(V2_FINAL, obs_dim, act_dim, "cpu")
    stage_l = _pe.build_predictor_model(BASELINE_CKPT, obs_dim, act_dim, "cpu")
    assert v2.context_components == "dynamics_only" and v2.context_dim == 16 and stage_l.context_dim == 0
    arms = ["null", "rolling48", "frozen10", "truncated20", "rolling24", "rolling96"]
    ev = _pe.PredictorEvaluator(v2, arms, n_env, obs_dim, act_dim, torch.device("cpu"))
    ev_l = _pe.PredictorEvaluator(stage_l, ["null"], n_env, obs_dim, act_dim, torch.device("cpu"))
    torch.manual_seed(0)
    prev_done = torch.zeros(n_env, dtype=torch.bool)
    for t in range(30):
        rows = slice(t * n_env, (t + 1) * n_env)  # row i + n_env is the next step of the same env
        obs, act, nxt = sd["obs"][rows], sd["actions"][rows], sd["next_obs"][rows]
        # done = the env's NEXT row (same env, next step) starts a new episode: catches time-outs,
        # which continues <= 0 misses
        done = sd["episode_ids"][rows] != sd["episode_ids"][(t + 1) * n_env:(t + 2) * n_env]
        out, ref = ev.step(obs, act, nxt, done), ev_l.step(obs, act, nxt, done)
        for m in ("phys1", "lat1", "phys4", "phys8"):
            x, y = out[f"predictor_null_{m}"], ref[f"predictor_null_{m}"]
            assert (x == y) or (x != x and y != y), (t, m, x, y)
        for arm in arms:
            assert out[f"predictor_{arm}_phys1"] == out[f"predictor_{arm}_phys1"], (t, arm)  # finite, no NaN
            if t >= 7:
                assert out[f"predictor_{arm}_phys8"] == out[f"predictor_{arm}_phys8"], (t, arm)
        if 10 < t < 20:
            # held after its freeze step; only an env that reset since switches to the null context
            # (a jump of norm ~1 for that env, i.e. ~1/n_env in the mean)
            assert out["predictor_frozen10_ctx_drift"] <= float(prev_done.sum()) / n_env + 1e-6, (t, prev_done.sum())
        if t == 20:
            assert out["predictor_truncated20_ctx_norm"] == 0.0  # window cleared at the truncation step
        prev_done = done
    assert out["predictor_rolling48_phys1"] != out["predictor_null_phys1"]


def _l3_module():
    spec = importlib.util.spec_from_file_location("_l3s", os.path.join(ROOT, "scripts", "mbrl", "level3_summary.py"))
    l3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(l3)
    return l3


def _l3_write(root, cond, seed, arms, n_steps=600):
    """arms: name -> f(step) -> phys1 value (NaN allowed); other metrics derived from it."""
    d = root / f"{cond}__s{seed}"
    d.mkdir()
    (d / ".done").touch()
    with open(d / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"predictor_{a}_{m}" for a in arms for m in ("phys1", "lat1", "phys4", "phys8", "ctx_norm", "ctx_drift")])
        for step in range(n_steps):
            row = [step]
            for fn in arms.values():
                e = fn(step)
                row += [e, e / 100, e * 1.5, e * 2, 0.99, 0.1]
            w.writerow(row)


def _l3_settle(l3, root, cond="c", **kw):
    r = l3.analyse(str(root), switch=250, width=25, pre=100, peak=100, steady=150, tol=0.10, **kw)[cond]
    return {a: e["k1"]["settling_steps_per_seed"] for a, e in r["arms"].items()}, r


def test_level3_summary_known_answer(tmp_path):
    """Exponential decay after the switch: oracle (tau 10) settles at 25 steps, rolling (tau 40) at
    100, null (never decays) is '>run'. last_exit reports the END of the last window above
    1.1 x the oracle's steady error."""
    l3 = _l3_module()

    def err(tau, floor=0.012, pre=0.010, jump=0.03):
        return lambda step: pre if step < 250 else (floor + (jump - floor) * math.exp(-(step - 250) / tau) if tau else jump)

    for seed in range(3):
        _l3_write(tmp_path, "gain0.7", seed, {a: (lambda f, sc: (lambda s: f(s) * sc))(err(t), 1 + 0.01 * seed)
                                             for a, t in {"truncated250": 10.0, "rolling48": 40.0, "null": 0.0}.items()})
    settle, r = _l3_settle(l3, tmp_path, "gain0.7")
    assert r["oracle"] == "truncated250"
    assert settle["truncated250"] == [25.0] * 3 and settle["rolling48"] == [100.0] * 3, settle
    assert settle["null"] == [math.inf] * 3
    assert math.isinf(r["arms"]["null"]["k1"]["settling"][0]) and r["arms"]["null"]["k1"]["settling"][2] == 0
    assert abs(r["arms"]["null"]["k1"]["pre"][0] - 0.0101) < 1e-3
    assert abs(r["arms"]["null"]["k1"]["steady"][0] - 0.0303) < 1e-3


def test_level3_settling_edge_cases(tmp_path):
    l3 = _l3_module()
    base = 0.010  # oracle level everywhere -> thr = 0.011
    arms = {
        "truncated250": lambda s: base,
        # never leaves the band -> 0
        "flat": lambda s: base,
        # error RISES after the switch: first window [250,275) inside, [275,325) above, then back:
        # the trivially-inside first window must NOT count as settled
        "rise": lambda s: 0.020 if 275 <= s < 325 else base,
        # transient dips into the band but the steady span is above -> '>run'
        "dips": lambda s: base if s in range(300, 325) or s in range(460, 470) else 0.020,
        # the steady span is all NaN -> '>run'
        "nansteady": lambda s: float("nan") if s >= 450 else base,
        # a window with no finite rows inside the post-switch span counts as OUT: [350,375) -> 125
        "nanwin": lambda s: float("nan") if 350 <= s < 375 else base,
        # one noisy window [400,425): last_exit -> 175, suffix_mean -> 0 (diluted by the tail)
        "noisy": lambda s: 0.015 if 400 <= s < 425 else base,
    }
    _l3_write(tmp_path, "c", 0, arms)
    settle, _ = _l3_settle(l3, tmp_path)
    assert settle["flat"] == [0.0] and settle["truncated250"] == [0.0]
    assert settle["rise"] == [75.0], settle["rise"]
    assert settle["dips"] == [math.inf]
    assert settle["nansteady"] == [math.inf]
    assert settle["nanwin"] == [125.0]
    assert settle["noisy"] == [175.0]
    settle_sm, _ = _l3_settle(l3, tmp_path, rule="suffix_mean")
    assert settle_sm["noisy"] == [0.0] and settle_sm["dips"] == [math.inf] and settle_sm["nanwin"] == [125.0]


def test_level3_settling_guards_and_missing_oracle(tmp_path):
    l3 = _l3_module()
    f = lambda s: 0.01
    _l3_write(tmp_path, "short", 0, {"truncated250": f, "null": f}, n_steps=380)
    with pytest.raises(ValueError, match="starts at/before the switch"):
        l3.analyse(str(tmp_path), switch=250, width=25, pre=100, peak=100, steady=150, tol=0.10)
    other = tmp_path / "b"
    other.mkdir()
    _l3_write(other, "odd", 0, {"truncated250": f, "null": f}, n_steps=610)
    with pytest.raises(ValueError, match="not a multiple"):
        l3.analyse(str(other), switch=250, width=25, pre=100, peak=100, steady=150, tol=0.10)
    # a seed that lacks the oracle arm: no KeyError; its settling is '>run' (NaN threshold)
    third = tmp_path / "c3"
    third.mkdir()
    _l3_write(third, "c", 0, {"truncated250": f, "null": f})
    _l3_write(third, "c", 1, {"null": f})
    settle, _ = _l3_settle(l3, third)
    assert settle["null"] == [0.0, math.inf], settle


class _ScriptedContextModel:
    """encode_context returns a scripted raw context (same for every env); null() is zero."""

    def __init__(self, dim=4):
        self.value = torch.zeros(dim)

        class _HE:
            def null(_self):
                return torch.zeros(dim)

        self.history_encoder = _HE()

    def encode_context(self, actions, transitions, pad_mask):
        return self.value.unsqueeze(0).expand(actions.shape[0], -1).clone()


def test_context_ema_tau1_is_rolling_exactly():
    model = make_model()
    a_ctrl, b_ctrl = _ctrl("rolling"), _ctrl("rolling", ema_tau=1.0)
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for t in range(HISTORY_LEN + 5):
            acts, trans = torch.randn(3, ACTION_DIM, generator=g), torch.randn(3, OBS_DIM, generator=g)
            done = torch.tensor([False, t == 7, False])
            assert torch.equal(a_ctrl.context(model), b_ctrl.context(model))
            a_ctrl.append(acts, trans, done)
            b_ctrl.append(acts, trans, done)


def test_context_ema_step_response_and_reset():
    tau, n = 0.05, 3
    model = _ScriptedContextModel()
    ctrl = _hist.ContextController(n, 8, ACTION_DIM, OBS_DIM, torch.device("cpu"), mode="rolling", ema_tau=tau)
    z = torch.zeros(n, ACTION_DIM), torch.zeros(n, OBS_DIM)
    for _ in range(5):  # raw context 0 -> EMA stays 0
        assert torch.equal(ctrl.context(model), torch.zeros(n, 4))
        ctrl.append(*z)
    model.value = torch.tensor([1.0, 0.0, 0.0, 0.0])  # step change in the raw context
    for k in range(1, 61):
        c = ctrl.context(model)
        expected = 1 - (1 - tau) ** k  # first-order response, time constant ~1/tau = 20 steps
        smooth_envs = [0, 2] if k > 30 else [0, 1, 2]  # env 1 resets at k=30 and restarts from the raw value
        assert torch.allclose(c[smooth_envs, 0], torch.full((len(smooth_envs),), expected), atol=1e-6), (k, c[:, 0], expected)
        if k > 30:
            assert torch.allclose(c[1], model.value), (k, c[1])
        assert (c.norm(dim=-1) <= 1 + 1e-6).all()  # convex combination stays in the unit ball
        ctrl.append(*z, torch.tensor([False, k == 30, False]))
    c = ctrl.context(model)  # env 1 reset at k=30 -> it re-started from the raw context
    assert torch.allclose(c[1], model.value)
    assert c[0, 0] < 0.96 and torch.allclose(c[0, 0], torch.tensor(1 - (1 - tau) ** 61), atol=1e-6)
    with pytest.raises(ValueError):
        _hist.ContextController(n, 8, ACTION_DIM, OBS_DIM, torch.device("cpu"), ema_tau=0.0)


# --------------------------------------------------------------------------- v3: context ensemble + risk-aware planner
import types as _types


def _load_planner_pkg():
    """planner.py uses package-relative imports; load it inside a synthetic package whose
    __init__ is never executed (the real one imports the policy prior)."""
    name = "_ldmq_pkg"
    if f"{name}.planner" in sys.modules:
        return sys.modules[f"{name}.planner"]
    pkg = _types.ModuleType(name)
    pkg.__path__ = [os.path.join(ROOT, "source", "ldm_quad", "ldm_quad", "mbrl")]
    sys.modules[name] = pkg
    return importlib.import_module(f"{name}.planner")


def _ens_model(m, seed=0, **kw):
    torch.manual_seed(seed)
    return LatentWorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=32, hidden_dim=64, depth=2, num_bins=21,
                            context_dim=CONTEXT_DIM, history_len=HISTORY_LEN, history_d_model=32, history_ff=64,
                            context_components="dynamics_only", context_ensemble=m, physical_feature_indices=[0, 1, 5], **kw)


def _planner(model, **kw):
    pl = _load_planner_pkg()
    lo, hi = -torch.ones(ACTION_DIM), torch.ones(ACTION_DIM)
    return pl.LatentMPPIPlanner(model, lo, hi, horizon=4, candidates=16, elites=4, iterations=2, num_pi_trajs=2,
                                use_continue_model=True, **kw)


def test_ensemble_keys_shapes_and_m1_is_v2():
    m5 = _ens_model(5)
    keys = [k for k in m5.state_dict() if k.endswith("context_weight")]
    assert keys == ["dynamics.0.context_weight"] and m5.dynamics[0].context_weight.shape == (5, 64, CONTEXT_DIM)
    assert any(k.startswith("history_encoder.4.") for k in m5.state_dict())
    assert torch.equal(m5.null_context_batch(3), torch.zeros(5, 3, CONTEXT_DIM))
    # M=1 builds exactly the v2 model (same keys, shapes and values from the same seed)
    a = _ens_model(1, seed=3)
    torch.manual_seed(3)
    b = LatentWorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=32, hidden_dim=64, depth=2, num_bins=21,
                         context_dim=CONTEXT_DIM, history_len=HISTORY_LEN, history_d_model=32, history_ff=64,
                         context_components="dynamics_only", physical_feature_indices=[0, 1, 5])  # v2, no ensemble arg
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    with pytest.raises(ValueError):
        LatentWorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, context_dim=4, history_len=4, context_ensemble=3)


def test_m1_planner_equals_reference_rollout():
    model = _ens_model(1).eval()
    with torch.no_grad():
        model.dynamics[0].context_weight.normal_()
    pl = _planner(model, planner_risk_lambda=1.0)  # lambda must not matter for M=1
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(3, OBS_DIM, generator=g)
    seqs = torch.rand(3, 16, 4, ACTION_DIM, generator=g) * 2 - 1
    ctx = torch.nn.functional.normalize(torch.randn(3, CONTEXT_DIM, generator=g), dim=-1)
    pl._context = ctx
    torch.manual_seed(1)
    got = pl.evaluate_sequences(obs, seqs)
    # reference: the pre-v3 algorithm written out
    torch.manual_seed(1)
    with torch.no_grad():
        z = model.encode(obs).unsqueeze(1).expand(-1, 16, -1).reshape(48, -1)
        c = ctx.unsqueeze(1).expand(-1, 16, -1).reshape(48, -1)
        ret, disc, alive = torch.zeros(48), torch.ones(48), torch.ones(48)
        for t in range(4):
            a = seqs[:, :, t].reshape(48, ACTION_DIM)
            r = model.reward(z, a, context=c).squeeze(-1)
            z = model.next(z, a, context=c)
            ret = ret + disc * alive * r
            alive = alive * model.continue_logits(z).sigmoid().squeeze(-1)
            disc = disc * pl.discount
        ta = model.pi(z, deterministic=False, context=c)
        ret = ret + disc * alive * model.Q(z, ta, return_type="min", context=c).squeeze(-1)
    assert torch.equal(got, ret.view(3, 16))


@pytest.mark.skipif(not os.path.exists(BASELINE_CKPT), reason="baseline checkpoint not present")
def test_ensemble_warm_start_exact(baseline_ckpt):
    kwargs = baseline_kwargs(baseline_ckpt)
    base = LatentWorldModel(**kwargs)
    base.load_state_dict(baseline_ckpt["model"])
    base.eval()
    m = LatentWorldModel(**kwargs, context_dim=CONTEXT_DIM, history_len=48, context_components="dynamics_only", context_ensemble=5)
    new = _ckpt.graft_context_free_state_dict(m, baseline_ckpt["model"])
    assert {k for k in new if not k.startswith("history_encoder.")} == {"dynamics.0.context_weight"}
    m.eval()
    torch.manual_seed(0)
    obs = torch.randn(8, kwargs["obs_dim"])
    a = torch.rand(8, kwargs["action_dim"]) * 2 - 1
    ctx = torch.nn.functional.normalize(torch.randn(5, 8, CONTEXT_DIM), dim=-1)
    with torch.no_grad():
        z = m.encode(obs)
        assert torch.equal(z, base.encode(obs))
        zn = m.next(z, a, context=ctx)  # [5, 8, D]
        ref = base.next(base.encode(obs), a)
        assert zn.shape[0] == 5 and all(torch.equal(zn[i], ref) for i in range(5))
    trainable, _ = m.freeze_base_for_adapter()
    assert trainable == 5 * (sum(p.numel() for p in m.history_encoder[0].parameters()) + 512 * CONTEXT_DIM)


def test_ensemble_members_get_gradient_and_diverge():
    m = _ens_model(3)
    m.freeze_base_for_adapter()
    opt = torch.optim.Adam(list(m.adapter_parameters()), lr=1e-2)
    H, B = 3, 32
    g = torch.Generator().manual_seed(0)
    batch = {"obs": torch.randn(H + 1, B, OBS_DIM, generator=g), "actions": torch.rand(H, B, ACTION_DIM, generator=g) * 2 - 1,
             "rewards": torch.randn(H, B, 1, generator=g), "continues": torch.ones(H, B, 1),
             "history_actions": torch.randn(B, HISTORY_LEN, ACTION_DIM, generator=g),
             "history_transitions": torch.randn(B, HISTORY_LEN, OBS_DIM, generator=g),
             "history_pad_mask": torch.zeros(B, HISTORY_LEN, dtype=torch.bool)}
    m.train()
    enc_grad = []
    for _ in range(3):
        loss, metrics, zs = m.loss(batch)
        assert torch.isfinite(loss) and zs.shape == (H + 1, B, 32)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        enc_grad.append([sum(float(p.grad.abs().sum()) for p in enc.parameters() if p.grad is not None) for enc in m.history_encoder])
        w = m.dynamics[0].context_weight
        assert all(float(w.grad[i].abs().sum()) > 0 for i in range(3))  # every member's projection is trained
        opt.step()
    assert enc_grad[0] == [0.0, 0.0, 0.0] and all(x > 0 for x in enc_grad[1])  # through W once it moves
    w = m.dynamics[0].context_weight.detach()
    assert not torch.equal(w[0], w[1]) and not torch.equal(w[1], w[2])  # members diverge (bootstrap + own encoders)


def test_risk_penalty_and_two_stage():
    model = _ens_model(3).eval()
    pl = _planner(model, planner_risk_lambda=2.0)
    member = torch.tensor([[[1.0, 5.0]], [[1.0, 1.0]], [[1.0, 3.0]]])  # [M=3, B=1, cand=2]
    score = pl._risk_score(member)
    assert score[0, 0] == 1.0  # zero disagreement: unchanged
    assert torch.allclose(score[0, 1], torch.tensor(3.0 - 2.0 * member[:, 0, 1].std(unbiased=False)))
    # identical members (same projection, same context) -> std 0 -> lambda irrelevant
    with torch.no_grad():
        model.dynamics[0].context_weight.copy_(torch.randn(1, 64, CONTEXT_DIM).expand(3, -1, -1))
    g = torch.Generator().manual_seed(1)
    obs = torch.randn(2, OBS_DIM, generator=g)
    seqs = torch.rand(2, 16, 4, ACTION_DIM, generator=g) * 2 - 1
    ctx1 = torch.nn.functional.normalize(torch.randn(1, 2, CONTEXT_DIM, generator=g), dim=-1)
    for p_ in (pl, _planner(model, planner_risk_lambda=0.0)):
        p_._context = ctx1.expand(3, -1, -1)
    torch.manual_seed(2)
    s_risk = pl.evaluate_sequences(obs, seqs)
    lam0 = _planner(model, planner_risk_lambda=0.0)
    lam0._context = ctx1.expand(3, -1, -1)
    torch.manual_seed(2)
    s_mean = lam0.evaluate_sequences(obs, seqs)
    assert torch.allclose(s_risk, s_mean, atol=1e-5)
    # two-stage with k >= candidates is exactly the full variant
    full = _planner(model, planner_risk_lambda=1.0)
    two = _planner(model, planner_risk_lambda=1.0, planner_two_stage_k=16)
    ctx = torch.nn.functional.normalize(torch.randn(3, 2, CONTEXT_DIM, generator=g), dim=-1)
    full._context = two._context = ctx
    torch.manual_seed(3)
    a = full.evaluate_sequences(obs, seqs)
    torch.manual_seed(3)
    b = two.evaluate_sequences(obs, seqs)
    assert torch.equal(a, b)
    # true two-stage (k < candidates): the re-scored top-k are the only candidates that can be elites
    two4 = _planner(model, planner_risk_lambda=1.0, planner_two_stage_k=6)
    two4._context = ctx
    s = two4.evaluate_sequences(obs, seqs)
    assert s.shape == (2, 16) and torch.isfinite(s).all()
    # a plan() call runs end to end with the ensemble context
    act = full.plan(obs, eval_mode=True, t0=True, context=ctx)
    assert act.shape == (2, ACTION_DIM) and full._last_disagreement is not None


def test_mean_field_equals_members_when_they_coincide():
    m = _ens_model(3).eval()
    with torch.no_grad():
        m.dynamics[0].context_weight.copy_(torch.randn(1, 64, CONTEXT_DIM).expand(3, -1, -1))
    g = torch.Generator().manual_seed(5)
    z = m.encode(torch.randn(4, OBS_DIM, generator=g))
    a = torch.rand(4, ACTION_DIM, generator=g) * 2 - 1
    c = torch.nn.functional.normalize(torch.randn(1, 4, CONTEXT_DIM, generator=g), dim=-1).expand(3, -1, -1)
    with torch.no_grad():
        full = m.next(z, a, context=c)
        mf = m.next_mean_field(z, a, c)
    assert torch.allclose(mf, full[0], atol=1e-6) and torch.allclose(full[0], full[2])
    with torch.no_grad():  # distinct members: mean-field is a (different) single surrogate, not member 0
        m.dynamics[0].context_weight.normal_()
        c2 = torch.nn.functional.normalize(torch.randn(3, 4, CONTEXT_DIM, generator=g), dim=-1)
        assert not torch.allclose(m.next_mean_field(z, a, c2), m.next(z, a, context=c2)[0])


def test_predictor_ensemble_scores_member_mean_prediction():
    """Passive predictor on a context ensemble: phys1/phys4 score the ENSEMBLE-MEAN physical
    prediction (hand-rolled reference), and the null arm of an untrained ensemble equals M=1."""
    model = _ens_model(3).eval()
    with torch.no_grad():
        model.dynamics[0].context_weight.normal_()
    ev = _pe.PredictorEvaluator(model, ["null", "rolling8"], 4, OBS_DIM, ACTION_DIM, torch.device("cpu"))
    ctrl = _hist.ContextController(4, 8, ACTION_DIM, OBS_DIM, torch.device("cpu"), mode="rolling", context_len=8)
    idx = torch.tensor([0, 1, 5])
    ctx_log, hist = [], []
    for t, (obs, a, nxt, done) in enumerate(_stream(10)):
        c = ctrl.context(model)
        assert c.shape == (3, 4, CONTEXT_DIM)
        out = ev.step(obs, a, nxt, done)
        with torch.no_grad():
            z1 = model.next(model.encode(obs, context=c), a, context=c)
            ref = (model.physical_features(z1).mean(0) - nxt.index_select(-1, idx)).square().mean()
            assert math.isclose(out["predictor_rolling8_phys1"], float(ref), rel_tol=1e-5)
            ctx_log.append(c)
            hist.append((obs, a))
            if t >= 3:
                o, c0 = hist[t - 3][0], ctx_log[t - 3]
                zk = model.encode(o, context=c0)
                for j in range(4):
                    zk = model.next(zk, hist[t - 3 + j][1], context=c0)
                ref4 = (model.physical_features(zk).mean(0) - nxt.index_select(-1, idx)).square().mean()
                assert math.isclose(out["predictor_rolling8_phys4"], float(ref4), rel_tol=1e-5)
        ctrl.append(a, nxt - obs, done)
        for m in _pe.METRICS:
            assert f"predictor_null_{m}" in out


def test_predictor_score_false_only_advances_and_all_done_is_nan():
    torch.manual_seed(0)
    model = LatentWorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=32, hidden_dim=64, depth=2, num_bins=21,
                             context_dim=CONTEXT_DIM, history_len=HISTORY_LEN, history_d_model=32, history_ff=64,
                             context_components="dynamics_only", physical_feature_indices=[0, 1, 5]).eval()
    with torch.no_grad():
        model.dynamics[0].context_weight.normal_()
    arms = ["null", "rolling8", "frozen5", "truncated6"]
    a_ev = _pe.PredictorEvaluator(model, arms, 4, OBS_DIM, ACTION_DIM, torch.device("cpu"))
    b_ev = _pe.PredictorEvaluator(model, arms, 4, OBS_DIM, ACTION_DIM, torch.device("cpu"))
    for t, (obs, a, nxt, done) in enumerate(_stream(14, done_at={9: [1]})):
        full = a_ev.step(obs, a, nxt, done)
        part = b_ev.step(obs, a, nxt, done, score=(t % 3 == 0))
        if t % 3:
            assert part == {}
        else:  # scoring every step or only on written rows gives the same numbers on those rows
            assert part.keys() == full.keys()
            assert all((part[k] == full[k]) or (part[k] != part[k] and full[k] != full[k]) for k in full), t
    obs, a, nxt, _ = next(_stream(1, seed=5))
    out = a_ev.step(obs, a, nxt, torch.ones(4, dtype=torch.bool))
    assert out["predictor_rolling8_phys1"] != out["predictor_rolling8_phys1"]  # every env done -> NaN


def test_context_ema_frozen_snapshots_smoothed_and_truncation_resets_ema():
    m = _ScriptedContextModel()
    fz = _hist.ContextController(2, 4, ACTION_DIM, OBS_DIM, torch.device("cpu"), mode="frozen", freeze_step=3, ema_tau=0.5)
    tr = _hist.ContextController(2, 4, ACTION_DIM, OBS_DIM, torch.device("cpu"), mode="truncated", truncate_step=3,
                                 ema_tau=0.5)
    a, x = torch.zeros(2, ACTION_DIM), torch.zeros(2, OBS_DIM)
    seq = [0.0, 1.0, 1.0, 1.0, 5.0, 5.0]
    got_f, got_t = [], []
    for t, v in enumerate(seq):
        m.value = torch.full((4,), v)
        got_f.append(float(fz.context(m)[0, 0]))
        got_t.append(float(tr.context(m)[0, 0]))
        fz.append(a, x)
        tr.append(a, x)
    # EMA 0.5: 0, .5, .75, .875 -> frozen at t=3 on the SMOOTHED .875 and held exactly after
    assert got_f == [0.0, 0.5, 0.75, 0.875, 0.875, 0.875], got_f
    # truncation after the append at t=2 (t counter hits 3): the t=3 context restarts from raw
    assert got_t[:3] == [0.0, 0.5, 0.75] and got_t[3] == 1.0 and got_t[4] == 3.0, got_t


def test_latent_model_from_args_rebuilds_ensemble_strictly():
    wm = _load("world_model")
    args = dict(latent_dim=32, hidden_dim=64, model_depth=2, discount=0.99, num_bins=21, latent_physical_indices="0,1,5",
                command_skip_indices="", history_context_dim=CONTEXT_DIM, history_len=HISTORY_LEN, history_d_model=32,
                history_ff=64, context_components="dynamics_only", context_ensemble=3)
    torch.manual_seed(0)
    src = wm.latent_model_from_args(args, OBS_DIM, ACTION_DIM)
    dst = wm.latent_model_from_args(dict(args), OBS_DIM, ACTION_DIM)
    dst.load_state_dict(src.state_dict(), strict=True)
    assert dst.context_ensemble == 3 and list(dst.physical_feature_indices) == [0, 1, 5]
    assert wm.parse_index_list([0, "1"]) == [0, 1] and wm.parse_index_list(None) == [] and wm.parse_index_list(" 2, 3 ,") == [2, 3]
    with pytest.raises(RuntimeError):  # an M=1 rebuild must not silently accept an M=3 state dict
        wm.latent_model_from_args({**args, "context_ensemble": 1}, OBS_DIM, ACTION_DIM).load_state_dict(src.state_dict())


def test_ensemble_next_with_null_context_and_member_dimmed_z():
    """context=None: memberless z [N, D] and member-dimmed z [M, N, D] both give [M, N, D]
    (previously the member-dimmed case expanded z twice to [M, M, N, D])."""
    model = _ens_model(3).eval()
    with torch.no_grad():
        model.dynamics[0].context_weight.normal_()
        z = model.encode(torch.randn(5, OBS_DIM))
        a = torch.rand(5, ACTION_DIM)
        z1 = model.next(z, a)  # memberless in
        assert z1.shape == (3, 5, z.shape[-1])
        z2 = model.next(z1, a)  # member-dimmed in, no context
        assert z2.shape == (3, 5, z.shape[-1])
        ref = model.next(z1, a, context=model.null_context_batch(5))
        assert torch.equal(z2, ref)
        # explicit override for a memberless z whose leading dim equals M
        zz = z[:3].unsqueeze(1).expand(3, 2, -1).contiguous()  # memberless [3, 2, D]
        out = model.next(zz, torch.rand(3, 2, ACTION_DIM), z_has_members=False)
        assert out.shape == (3, 3, 2, z.shape[-1])
        assert torch.equal(model.member_mean(z2), z2.mean(0))
        assert torch.equal(_ens_model(1).member_mean(z), z)


def test_ensemble_proposals_use_mean_field_and_two_stage_diagnostics():
    model = _ens_model(3).eval()
    with torch.no_grad():
        model.dynamics[0].context_weight.normal_()
    g = torch.Generator().manual_seed(4)
    obs = torch.randn(2, OBS_DIM, generator=g)
    ctx = torch.nn.functional.normalize(torch.randn(3, 2, CONTEXT_DIM, generator=g), dim=-1)
    pl = _planner(model, planner_risk_lambda=1.0, planner_two_stage_k=6)
    pl._context = ctx
    # proposal step = the 1x mean-field surrogate, not an average of member latents
    with torch.no_grad():
        z = model.encode(obs)
        a = torch.rand(2, ACTION_DIM, generator=g)
        assert torch.equal(pl._next_mean(z, a, ctx), model.next_mean_field(z, a, ctx))
        assert not torch.allclose(pl._next_mean(z, a, ctx), model.next(z, a, context=ctx).mean(0))
    # two-stage diagnostics: mean/std from stage 1 over ALL candidates, re-scored top-k separately
    act = pl.plan(obs, eval_mode=True, t0=True, context=ctx)
    d = pl.last_diagnostics
    assert act.shape == (2, ACTION_DIM)
    assert d["planner_candidate_return_mean"] == float(pl._last_stage1.mean())
    assert d["planner_rescored_return_mean"] == float(pl._last_rescored.mean())
    assert pl._last_stage1.shape == (2, 16) and pl._last_rescored.shape == (2, 6)
    # full scoring: no stage-1 record; rescored == the candidate scores
    full = _planner(model, planner_risk_lambda=1.0)
    full.plan(obs, eval_mode=True, t0=True, context=ctx)
    assert full._last_stage1 is None
    assert full.last_diagnostics["planner_rescored_return_mean"] == full.last_diagnostics["planner_candidate_return_mean"]
    # k < elites is rejected
    with pytest.raises(ValueError, match="must be >= elites"):
        _planner(model, planner_two_stage_k=3)

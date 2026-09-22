"""Level 0 wiring tests for the SIT history-context latent world model.

Torch only, no simulator. world_model.py / replay.py / history.py are loaded by file
path because importing the ldm_quad package pulls in ldm_quad.tasks -> Isaac Sim.

Run: /home/rml2/anaconda3/envs/isaaclab/bin/python -m pytest tests/mbrl/test_history_context.py -v
"""

from __future__ import annotations

import importlib.util
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

OBS_DIM = 12
ACTION_DIM = 4
CONTEXT_DIM = 16
HISTORY_LEN = 8
BATCH = 6


def make_model(context_dim: int = CONTEXT_DIM, seed: int = 0) -> LatentWorldModel:
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
    # Reward/Q final layers are zero-initialised (TD-MPC2), which would make their
    # outputs trivially context-independent. Randomise them so the tests measure wiring.
    for head in [model.reward_head, *model.q_heads]:
        final = [m for m in head if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(final.weight, std=0.1)
        nn.init.normal_(final.bias, std=0.1)
    # A nonzero null context so "equals null()" is a real check, not 0 == 0.
    if model.history_encoder is not None:
        with torch.no_grad():
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
@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "Context is concatenated into the first Linear of every conditioned MLP, so the input "
        "width changes and a context-free checkpoint cannot be loaded. Planned fix: a "
        "zero-initialised additive context projection into each first layer (not implemented)."
    ),
)
def test_warm_start_equivalence():
    base = make_model(context_dim=0, seed=0)
    ctx_model = make_model(context_dim=CONTEXT_DIM, seed=1)
    missing, unexpected = ctx_model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("history_encoder.") for k in missing), missing
    ctx_model.eval()

    obs = torch.randn(BATCH, OBS_DIM)
    a = torch.randn(BATCH, ACTION_DIM).clamp(-1, 1)
    with torch.no_grad():
        null = ctx_model.history_encoder.null().expand(BATCH, -1)
        z_base = base.encode(obs)
        z_ctx = ctx_model.encode(obs, context=null)
        assert torch.allclose(z_base, z_ctx)
        assert torch.allclose(base.next(z_base, a), ctx_model.next(z_ctx, a, context=null))
        assert torch.allclose(base.reward(z_base, a), ctx_model.reward(z_ctx, a, context=null))


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

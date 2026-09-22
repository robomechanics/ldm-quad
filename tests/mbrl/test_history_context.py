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

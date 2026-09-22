"""Warm-starting a history-context LatentWorldModel from a context-free checkpoint.

Torch-only on purpose (no package-relative imports), so tests can load this file by path
without pulling in Isaac Sim through the ldm_quad package.

The context model differs from a context-free one only by ADDED tensors: the zero-initialised
``*.context_weight`` projections (world_model.ContextSequential) and the ``history_encoder.*``
set transformer. Every base tensor keeps its name and shape, so the graft is a key-validated
non-strict load plus an optimizer-state remap that skips the added parameters.
"""

from __future__ import annotations

import torch


def is_context_graft_key(key: str) -> bool:
    """True for state_dict keys a history-context model adds on top of a context-free one."""
    return key.endswith("context_weight") or key.startswith("history_encoder.")


def is_context_free_state_dict(state: dict) -> bool:
    return not any(is_context_graft_key(k) for k in state)


def graft_context_free_state_dict(model: torch.nn.Module, state: dict) -> list[str]:
    """Load a context-free ``state`` into a history-context ``model``.

    The added tensors keep their initialisation (context projections at zero, so the grafted
    model reproduces the checkpoint exactly). Raises BEFORE touching the model unless the
    missing keys are exactly the model's context/history keys and nothing is unexpected, so a
    genuinely mismatched checkpoint is never silently accepted. Returns the keys left at init.
    """
    model_keys = set(model.state_dict())
    state_keys = set(state)
    missing = model_keys - state_keys
    unexpected = state_keys - model_keys
    expected_missing = {k for k in model_keys if is_context_graft_key(k)}
    if unexpected or missing != expected_missing:
        raise ValueError(
            "Checkpoint does not fit this history-context model as a context-free graft: "
            f"unexpected={sorted(unexpected)[:6]} "
            f"missing_non_context={sorted(missing - expected_missing)[:6]} "
            f"context_keys_present_in_checkpoint={sorted(expected_missing - missing)[:6]}"
        )
    result = model.load_state_dict(state, strict=False)
    assert set(result.missing_keys) == expected_missing and not result.unexpected_keys
    return sorted(expected_missing)


def remap_optimizer_state_for_added_params(
    optimizer: torch.optim.Optimizer,
    saved_state: dict,
    added_params: list[torch.nn.Parameter],
    allow_shape_mismatch: bool = False,
) -> tuple[int, int, int]:
    """Load optimizer state saved WITHOUT ``added_params`` into ``optimizer``, which has them.

    torch optimizers key saved state by position, so a plain load_state_dict either fails
    (group sizes differ) or silently assigns moments to the wrong tensors. Removing the added
    parameters from each live group recovers the saved order exactly; those tensors get no
    state (Adam restarts them at step 0 with correct bias correction) and every other tensor
    keeps its moments. With ``allow_shape_mismatch`` (the #6a command-skip graft widens first
    layers) state whose shape no longer matches is dropped instead of raising.

    Returns (kept, fresh, dropped) tensor counts.
    """
    saved_groups = saved_state["param_groups"]
    groups = optimizer.param_groups
    if len(saved_groups) != len(groups):
        raise ValueError(f"optimizer has {len(groups)} param groups, checkpoint has {len(saved_groups)}")
    added_ids = {id(p) for p in added_params}
    new_state: dict[int, dict] = {}
    new_groups = []
    index = kept = fresh = dropped = 0
    for group_i, (group, saved_group) in enumerate(zip(groups, saved_groups)):
        n_base = sum(1 for p in group["params"] if id(p) not in added_ids)
        if n_base != len(saved_group["params"]):
            raise ValueError(
                f"param group {group_i}: {n_base} non-added params but checkpoint saved "
                f"{len(saved_group['params'])}; the saved order cannot be recovered"
            )
        saved_ids = iter(saved_group["params"])
        ids = []
        for param in group["params"]:
            ids.append(index)
            if id(param) in added_ids:
                fresh += 1
            else:
                entry = saved_state["state"].get(next(saved_ids))
                if entry is not None:
                    exp_avg = entry.get("exp_avg")
                    if exp_avg is not None and tuple(exp_avg.shape) != tuple(param.shape):
                        if not allow_shape_mismatch:
                            raise ValueError(
                                f"param group {group_i} slot {len(ids) - 1}: saved state shape "
                                f"{tuple(exp_avg.shape)} != param shape {tuple(param.shape)}"
                            )
                        dropped += 1
                    else:
                        new_state[index] = entry
                        kept += 1
            index += 1
        new_group = {k: v for k, v in saved_group.items() if k != "params"}
        new_group["params"] = ids
        new_groups.append(new_group)
    optimizer.load_state_dict({"state": new_state, "param_groups": new_groups})
    return kept, fresh, dropped

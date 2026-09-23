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


def latent_optimizer_param_groups(
    model: torch.nn.Module, mode: str, exclude: set[int] | None = None
) -> tuple[list[list[torch.nn.Parameter]], list[torch.nn.Parameter]]:
    """Parameter lists, in optimizer order, for the latent world-model optimizer's two groups
    (encoder, non-encoder) and the policy optimizer, as built under ``--sit_train_mode mode``.

    Used both to BUILD the optimizers and to describe what a CHECKPOINT's optimizers held, so
    saved Adam state can be remapped by identity (mbrl/checkpoint.py remap_optimizer_state).
    ``exclude`` drops tensors by id (a context-free checkpoint never held the SIT tensors).
    """
    exclude = exclude or set()
    groups = [list(model.encoder_parameters()), list(model.non_encoder_model_parameters())]
    policy = list(model.policy_parameters())
    if mode == "adapter":
        adapter = {id(p) for p in model.adapter_parameters()} | {id(p) for p in model.policy_adapter_parameters()}
        groups = [[p for p in g if id(p) in adapter] for g in groups]
        policy = [p for p in policy if id(p) in adapter]
    elif mode != "full":
        raise ValueError(f"unknown sit_train_mode {mode!r}")
    groups = [[p for p in g if id(p) not in exclude] for g in groups]
    policy = [p for p in policy if id(p) not in exclude]
    return groups, policy


def remap_optimizer_state(
    optimizer: torch.optim.Optimizer,
    saved_state: dict,
    saved_param_order: list[list[torch.nn.Parameter]],
    allow_shape_mismatch: bool = False,
) -> dict[str, int]:
    """Load optimizer state that was saved for a DIFFERENT parameter set into ``optimizer``.

    torch optimizers key saved state by position, so a plain load_state_dict either fails
    (group sizes differ) or silently assigns moments to the wrong tensors. The caller supplies,
    per param group, the LIVE Parameter objects in the order the checkpoint's optimizer held
    them; state then moves by identity:

    * a live optimizer tensor that was in the saved order keeps its moments   -> ``kept``
    * a live optimizer tensor the checkpoint never optimised starts at step 0 -> ``fresh``
    * saved state for a tensor this optimizer no longer holds (frozen base)   -> ``dropped_absent``
    * saved state whose shape no longer matches (#6a widened first layers)    -> ``dropped_shape``
      (raises unless ``allow_shape_mismatch``)

    Group hyperparameters (lr, betas, ...) come from the checkpoint, as with a plain load.
    """
    saved_groups = saved_state["param_groups"]
    groups = optimizer.param_groups
    if not (len(saved_groups) == len(groups) == len(saved_param_order)):
        raise ValueError(
            f"param groups: optimizer {len(groups)}, checkpoint {len(saved_groups)}, "
            f"saved order {len(saved_param_order)}"
        )
    slot_of: dict[int, int] = {}
    for group_i, (order, saved_group) in enumerate(zip(saved_param_order, saved_groups)):
        if len(order) != len(saved_group["params"]):
            raise ValueError(
                f"param group {group_i}: saved order lists {len(order)} params but the checkpoint "
                f"saved {len(saved_group['params'])}; the saved order cannot be recovered"
            )
        for param, slot in zip(order, saved_group["params"]):
            slot_of[id(param)] = slot

    live_ids = {id(p) for g in groups for p in g["params"]}
    counts = {"kept": 0, "fresh": 0, "dropped_absent": 0, "dropped_shape": 0}
    for order in saved_param_order:
        for param in order:
            if id(param) not in live_ids and saved_state["state"].get(slot_of[id(param)]):
                counts["dropped_absent"] += 1

    new_state: dict[int, dict] = {}
    new_groups = []
    index = 0
    for group_i, (group, saved_group) in enumerate(zip(groups, saved_groups)):
        ids = []
        for param in group["params"]:
            ids.append(index)
            slot = slot_of.get(id(param))
            entry = saved_state["state"].get(slot) if slot is not None else None
            if slot is None:
                counts["fresh"] += 1
            elif entry:
                exp_avg = entry.get("exp_avg")
                if exp_avg is not None and tuple(exp_avg.shape) != tuple(param.shape):
                    if not allow_shape_mismatch:
                        raise ValueError(
                            f"param group {group_i} slot {len(ids) - 1}: saved state shape "
                            f"{tuple(exp_avg.shape)} != param shape {tuple(param.shape)}"
                        )
                    counts["dropped_shape"] += 1
                else:
                    new_state[index] = entry
                    counts["kept"] += 1
            index += 1
        new_group = {k: v for k, v in saved_group.items() if k != "params"}
        new_group["params"] = ids
        new_groups.append(new_group)
    optimizer.load_state_dict({"state": new_state, "param_groups": new_groups})
    return counts


def remap_optimizer_state_for_added_params(
    optimizer: torch.optim.Optimizer,
    saved_state: dict,
    added_params: list[torch.nn.Parameter],
    allow_shape_mismatch: bool = False,
) -> tuple[int, int, int]:
    """Special case of :func:`remap_optimizer_state`: the checkpoint's optimizer held exactly
    this optimizer's tensors minus ``added_params``, in the same order.

    Returns (kept, fresh, dropped_shape)."""
    added_ids = {id(p) for p in added_params}
    order = [[p for p in g["params"] if id(p) not in added_ids] for g in optimizer.param_groups]
    counts = remap_optimizer_state(optimizer, saved_state, order, allow_shape_mismatch)
    return counts["kept"], counts["fresh"], counts["dropped_shape"]

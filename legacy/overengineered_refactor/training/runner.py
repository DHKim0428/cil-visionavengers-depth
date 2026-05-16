from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.adapters import is_adapter_parameter

TRAINABLE_FORMAT = "da2_trainable_checkpoint_v1"
FULL_FORMAT = "full_model_checkpoint_v1"
LEGACY_DA2_FULL_FORMAT = "da2_full_model_checkpoint_v1"


def checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return final-schema checkpoint save settings with storage-conscious defaults."""
    current = config.get("checkpoint", {}) or {}
    return {
        "save_policy": "trainable_only",
        "keep_latest": True,
        "keep_best": True,
        "save_optimizer": True,
        **current,
    }


def split_trainable_state(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    adapter = {}
    base = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = adapter if is_adapter_parameter(name) else base
        target[name] = param.detach().cpu()
    return adapter, base


def build_checkpoint_payload(
    *,
    model: nn.Module,
    config: dict[str, Any],
    base_checkpoint: str | Path | None,
    epoch: int,
    global_step: int,
    best_sirmse: float,
    val_history: list[dict[str, float]],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    include_optimizer: bool = False,
    save_policy: str = "trainable_only",
) -> dict[str, Any]:
    common = {
        "epoch": epoch,
        "global_step": global_step,
        "best_sirmse": best_sirmse,
        "config": config,
        "base_checkpoint": None if base_checkpoint is None else str(base_checkpoint),
        "val_history": val_history,
    }
    if save_policy == "full_model":
        payload = {
            "format": FULL_FORMAT,
            "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            **common,
        }
    elif save_policy == "trainable_only":
        adapter_state, base_state = split_trainable_state(model)
        payload = {
            "format": TRAINABLE_FORMAT,
            "adapter": adapter_state,
            "trainable_base": base_state,
            "trainable": {**base_state, **adapter_state},
            **common,
        }
    else:
        raise ValueError(f"Unsupported checkpoint save_policy: {save_policy}")

    if include_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if include_optimizer and scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def is_trainable_checkpoint(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") == TRAINABLE_FORMAT


def is_full_model_checkpoint(payload: Any) -> bool:
    return isinstance(payload, dict) and (
        payload.get("format") in {FULL_FORMAT, LEGACY_DA2_FULL_FORMAT} or "model" in payload
    )


def checkpoint_has_config(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("config"), dict)


def restore_model_payload(model: nn.Module, payload: Any, *, strict: bool = True) -> None:
    if is_trainable_checkpoint(payload):
        state = payload.get("trainable")
        if state is None:
            state = {**payload.get("trainable_base", {}), **payload.get("adapter", {})}
        current = model.state_dict()
        missing = [name for name in state if name not in current]
        if missing:
            raise KeyError(f"Checkpoint contains parameters not present in model. First missing: {missing[0]}")
        current.update(state)
        model.load_state_dict(current, strict=strict)
        return
    if is_full_model_checkpoint(payload):
        model.load_state_dict(payload["model"], strict=strict)
        return
    raise ValueError("Unsupported checkpoint payload for model restoration")

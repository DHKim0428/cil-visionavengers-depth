from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterSummary:
    enabled: bool
    adapter_type: str
    target_mode: str
    modules_wrapped: int
    adapter_parameters: int
    wrapped_module_names: list[str]


def adapter_is_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("adapter", {}).get("enabled", False))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if base.groups != 1:
            raise ValueError("LoRAConv2d currently supports only groups=1 convolutions")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Conv2d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
            padding_mode=base.padding_mode,
        )
        self.lora_B = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_adapters_from_config(model: nn.Module, config: dict[str, Any]) -> AdapterSummary:
    adapter_cfg = config.get("adapter", {}) or {}
    if not adapter_cfg.get("enabled", False):
        return AdapterSummary(False, "none", "none", 0, 0, [])

    adapter_type = adapter_cfg.get("type", "lora")
    if adapter_type != "lora":
        raise ValueError(f"Unsupported adapter type: {adapter_type}")

    target_cfg = adapter_cfg.get("target", {}) or {}
    target_mode = target_cfg.get("mode", "trainable_scope")
    rank = int(adapter_cfg.get("rank", 8))
    alpha = float(adapter_cfg.get("alpha", rank))
    dropout = float(adapter_cfg.get("dropout", 0.0))

    wrapped = inject_lora(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_mode=target_mode,
        patterns=target_cfg.get("patterns", []),
    )
    adapter_params = sum(param.numel() for name, param in model.named_parameters() if is_adapter_parameter(name))
    return AdapterSummary(True, adapter_type, target_mode, len(wrapped), adapter_params, wrapped)


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_mode: str,
    patterns: list[str] | None = None,
) -> list[str]:
    replacements: list[tuple[str, nn.Module]] = []
    for module_name, module in list(model.named_modules()):
        if module_name == "" or _inside_lora_wrapper(module_name):
            continue
        if not _is_supported_target(module, target_mode):
            continue
        if not _matches_target(module_name, module, target_mode, patterns or []):
            continue
        replacements.append((module_name, _wrap_with_lora(module, rank, alpha, dropout)))

    for module_name, wrapped_module in replacements:
        _set_module(model, module_name, wrapped_module)
    return [name for name, _ in replacements]


def is_adapter_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def adapter_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for name, param in model.named_parameters() if is_adapter_parameter(name))


def base_trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad and not is_adapter_parameter(name)
    )


def _inside_lora_wrapper(module_name: str) -> bool:
    parts = module_name.split(".")
    return "lora_A" in parts or "lora_B" in parts


def _is_supported_target(module: nn.Module, target_mode: str) -> bool:
    if target_mode == "all_linear":
        return isinstance(module, nn.Linear)
    return isinstance(module, (nn.Linear, nn.Conv2d)) and not isinstance(module, (LoRALinear, LoRAConv2d))


def _matches_target(module_name: str, module: nn.Module, target_mode: str, patterns: list[str]) -> bool:
    if target_mode == "trainable_scope":
        return any(param.requires_grad for param in module.parameters(recurse=False))
    if target_mode == "decoder":
        return module_name.startswith("depth_head")
    if target_mode == "all_linear":
        return isinstance(module, nn.Linear)
    if target_mode == "regex":
        if not patterns:
            raise ValueError("adapter.target.mode=regex requires adapter.target.patterns")
        return any(re.search(pattern, module_name) for pattern in patterns)
    raise ValueError(
        f"Unsupported LoRA target mode '{target_mode}'. Expected trainable_scope, decoder, all_linear, or regex."
    )


def _wrap_with_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> nn.Module:
    if isinstance(module, nn.Linear):
        return LoRALinear(module, rank, alpha, dropout)
    if isinstance(module, nn.Conv2d):
        return LoRAConv2d(module, rank, alpha, dropout)
    raise TypeError(f"Unsupported LoRA target module: {type(module).__name__}")


def _set_module(root: nn.Module, module_name: str, module: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, module)

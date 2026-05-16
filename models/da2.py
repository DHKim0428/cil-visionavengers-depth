from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

DA2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

TRAINABLE_PREFIXES = {
    "decoder": ["depth_head"],
    "refinenets_output": ["depth_head.scratch.refinenet", "depth_head.scratch.output_conv"],
}


def default_checkpoint_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["da2_checkpoint_dir"]) / f"depth_anything_v2_{cfg['model']['encoder']}.pth"


def import_da2(repo: str | Path):
    repo = Path(repo)
    if not repo.exists():
        raise FileNotFoundError(f"Depth-Anything-V2 repo not found: {repo}. Run scripts/setup_da2.sh")
    for p in [repo, repo / "metric_depth"]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
    return DepthAnythingV2


def build_da2(cfg: dict[str, Any]) -> tuple[nn.Module, Path]:
    ckpt = Path(cfg.get("paths", {}).get("checkpoint") or default_checkpoint_path(cfg))
    if not ckpt.exists():
        raise FileNotFoundError(f"DA2 checkpoint not found: {ckpt}")

    encoder = cfg["model"]["encoder"]
    model = import_da2(cfg["paths"]["da2_repo"])(**DA2_CONFIGS[encoder])
    payload = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(payload.get("model", payload) if isinstance(payload, dict) else payload, strict=True)

    scope = cfg.get("base", {}).get("trainable_scope", "frozen")
    for name, p in model.named_parameters():
        if scope == "frozen":
            p.requires_grad = False
        elif scope == "full":
            p.requires_grad = True
        elif scope in TRAINABLE_PREFIXES:
            p.requires_grad = any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES[scope])
        else:
            raise ValueError(f"Unknown DA2 trainable_scope: {scope}")

    if cfg.get("adapter", {}).get("enabled", False):
        add_lora(model, cfg["adapter"])
    return model, ckpt


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * self.scale


def add_lora(model: nn.Module, cfg: dict[str, Any]) -> None:
    rank = int(cfg.get("rank", 8))
    alpha = float(cfg.get("alpha", rank))
    target = cfg.get("target", {}) or {}
    mode = target.get("mode", "trainable_scope")
    patterns = target.get("patterns", [])

    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name.endswith(".A") or name.endswith(".B"):
            continue
        if mode == "trainable_scope" and not any(p.requires_grad for p in module.parameters(recurse=False)):
            continue
        if mode == "decoder" and not name.startswith("depth_head"):
            continue
        if mode == "regex" and not any(re.search(p, name) for p in patterns):
            continue
        replacements.append((name, LoRALinear(module, rank, alpha)))

    for name, wrapped in replacements:
        parent_name, child = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, wrapped)


def load_training_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") == "trainable_only":
        state = model.state_dict()
        state.update(payload.get("trainable", {}))
        state.update(payload.get("adapter", {}))
        state.update(payload.get("trainable_base", {}))
        model.load_state_dict(state, strict=True)
    elif "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
    else:
        model.load_state_dict(payload, strict=True)
    return payload


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def parameter_counts(model: nn.Module) -> dict[str, float | int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable, "trainable_pct": 100.0 * trainable / total if total else 0.0}

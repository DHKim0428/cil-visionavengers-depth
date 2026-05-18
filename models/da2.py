from __future__ import annotations

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
    model_name = cfg["model"]
    if not isinstance(model_name, str) or not model_name.startswith("da2_"):
        raise ValueError(f"DA2 config expects model like da2_vits, got {model_name!r}")
    encoder = model_name.removeprefix("da2_")
    if encoder not in DA2_CONFIGS:
        raise ValueError(f"Unsupported DA2 model: {model_name}")

    ckpt = cfg.get("paths", {}).get("checkpoint")
    ckpt = Path(ckpt) if ckpt else Path(cfg["paths"]["da2_checkpoint_dir"]) / f"depth_anything_v2_{encoder}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"DA2 checkpoint not found: {ckpt}")

    model = import_da2(cfg["paths"]["da2_repo"])(**DA2_CONFIGS[encoder])
    payload = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(payload.get("model", payload) if isinstance(payload, dict) else payload, strict=True)

    scope = cfg.get("trainable", "frozen")
    for name, p in model.named_parameters():
        if scope == "frozen":
            p.requires_grad = False
        elif scope == "full":
            p.requires_grad = True
        elif scope in TRAINABLE_PREFIXES:
            p.requires_grad = any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES[scope])
        else:
            raise ValueError(f"Unknown DA2 trainable: {scope}")

    if cfg.get("adapter", {}).get("enabled", False):
        model = add_lora(model, cfg["adapter"])
    return model, ckpt


def add_lora(model: nn.Module, cfg: dict[str, Any]) -> nn.Module:
    if cfg.get("type") != "lora":
        raise ValueError(f"Unsupported adapter type: {cfg.get('type')}")
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA requires `peft`. Run `python -m pip install -r requirements.txt`.") from exc

    target_modules = cfg.get("target_modules")
    if not target_modules:
        prefixes = cfg.get("target_prefixes") or []
        module_types = set(cfg.get("target_module_types") or ["linear"])
        allowed_types = []
        if "linear" in module_types:
            allowed_types.append(nn.Linear)
        if "conv2d" in module_types:
            allowed_types.append(nn.Conv2d)
        if not allowed_types:
            raise ValueError(f"Unsupported LoRA target_module_types: {sorted(module_types)}")
        target_modules = [
            name
            for name, module in model.named_modules()
            if isinstance(module, tuple(allowed_types)) and any(name.startswith(prefix) for prefix in prefixes)
        ]
    if not target_modules:
        raise ValueError("LoRA adapter needs `target_modules` or non-empty `target_prefixes`.")

    return get_peft_model(model, LoraConfig(
        r=int(cfg.get("rank", 8)),
        lora_alpha=int(cfg.get("alpha", cfg.get("rank", 8))),
        lora_dropout=float(cfg.get("dropout", 0.0)),
        target_modules=target_modules,
        bias="none",
    ))


def load_training_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") == "trainable_only":
        state = model.state_dict()
        state.update(payload.get("trainable", {}))
        model.load_state_dict(state, strict=True)
    elif "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
    else:
        model.load_state_dict(payload, strict=True)
    return payload

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from models.da2 import build_da2, load_training_checkpoint
from models.da2_refine import build_da2_unet_refine
from models.unet import UNetBaseline


def load_model_for_inference(cfg: dict[str, Any], checkpoint: str | None, device: torch.device) -> torch.nn.Module:
    if cfg["model"] == "da2_unet_refine":
        if not checkpoint:
            raise ValueError("DA2 U-Net refinement inference requires --checkpoint")
        ckpt = Path(os.path.expanduser(os.path.expandvars(checkpoint)))
        payload = torch.load(ckpt, map_location="cpu")
        if isinstance(payload, dict) and payload.get("format") == "trainable_only":
            cfg.setdefault("paths", {})["checkpoint"] = payload["base_checkpoint"]
        model, _ = build_da2_unet_refine(cfg)
        load_training_checkpoint(model, ckpt)
        return model.to(device).eval()
    if checkpoint:
        cfg.setdefault("paths", {})["checkpoint"] = os.path.expanduser(os.path.expandvars(checkpoint))
    if str(cfg["model"]).startswith("da2_"):
        model_name = cfg["model"]
        ckpt = Path(cfg.get("paths", {}).get("checkpoint") or Path(cfg["paths"]["da2_checkpoint_dir"]) / f"depth_anything_v2_{model_name.removeprefix('da2_')}.pth")
        payload = torch.load(ckpt, map_location="cpu")
        if isinstance(payload, dict) and payload.get("format") == "trainable_only":
            cfg.setdefault("paths", {})["checkpoint"] = payload["base_checkpoint"]
            model, _ = build_da2(cfg)
            load_training_checkpoint(model, ckpt)
        elif isinstance(payload, dict) and payload.get("format") == "full_model":
            cfg.setdefault("paths", {}).pop("checkpoint", None)
            model, _ = build_da2(cfg)
            load_training_checkpoint(model, ckpt)
        else:
            model, _ = build_da2(cfg)
        return model.to(device).eval()
    if cfg["model"] == "unet":
        if not checkpoint:
            raise ValueError("U-Net inference requires --checkpoint")
        model = UNetBaseline()
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload)
        return model.to(device).eval()
    raise ValueError(f"Unknown model: {cfg['model']}")

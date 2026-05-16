from __future__ import annotations

import sys
from pathlib import Path

import torch


MODEL_CFGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

STRATEGY_LAYERS = {
    "decoder": ["depth_head"],
    "refinenets_output": [
        "depth_head.scratch.refinenet",
        "depth_head.scratch.output_conv",
    ],
}


def checkpoint_path_for_encoder(checkpoint_dir: str | Path, encoder: str) -> Path:
    return Path(checkpoint_dir) / f"depth_anything_v2_{encoder}.pth"


def import_depth_anything_v2(da2_repo: str | Path):
    repo = Path(da2_repo)
    if not repo.exists():
        raise FileNotFoundError(
            f"DA2 upstream repo not found at {repo}. Run: bash scripts/setup/setup_da2.sh"
        )
    metric_repo = repo / "metric_depth"
    for path in (repo, metric_repo):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

    return DepthAnythingV2


def extract_state_dict(checkpoint_obj):
    """Extract a DA2 model state_dict from official, best.pth, or latest.pth files."""
    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        return checkpoint_obj["model"]
    return checkpoint_obj


def create_da2_model(*, encoder: str, da2_repo: str | Path) -> torch.nn.Module:
    if encoder not in MODEL_CFGS:
        raise ValueError(f"Unsupported DA2 encoder '{encoder}'. Expected one of {sorted(MODEL_CFGS)}")
    DepthAnythingV2 = import_depth_anything_v2(da2_repo)
    return DepthAnythingV2(**MODEL_CFGS[encoder])


def load_da2_model(
    *,
    encoder: str,
    da2_repo: str | Path,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"DA2 checkpoint not found at {checkpoint}. Run setup or pass --checkpoint."
        )

    model = create_da2_model(encoder=encoder, da2_repo=da2_repo)
    checkpoint_obj = torch.load(checkpoint, map_location=map_location)
    model.load_state_dict(extract_state_dict(checkpoint_obj), strict=True)
    return model


def apply_base_trainable_scope(model: torch.nn.Module, scope: str) -> list[str]:
    """Set requires_grad for original DA2 parameters according to trainable scope."""
    if scope == "frozen":
        for param in model.parameters():
            param.requires_grad = False
    elif scope == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif scope in STRATEGY_LAYERS:
        prefixes = STRATEGY_LAYERS[scope]
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)
    else:
        raise ValueError(
            f"Unsupported base.trainable_scope '{scope}'. "
            "Expected frozen, full, decoder, or refinenets_output."
        )
    return [name for name, param in model.named_parameters() if param.requires_grad]


def parameter_summary(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    pct = 100.0 * trainable / total if total else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": pct,
    }

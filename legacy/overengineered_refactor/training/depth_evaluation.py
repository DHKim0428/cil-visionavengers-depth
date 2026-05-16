from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset.raw_cil import discover_rgb_filenames, load_rgb_depth
from training.metrics import (
    disparity_to_depth,
    resize_to_match,
    sanitize_direct_depth,
    sirmse_from_depth,
    valid_depth_mask,
)
from training.splits import make_or_load_split


@dataclass(frozen=True)
class DepthPrediction:
    """Model adapter output before canonical metric conversion."""

    values: torch.Tensor | np.ndarray
    kind: str  # "depth" or "disparity"
    label: str = "prediction"
    metadata: dict[str, Any] = field(default_factory=dict)


class DepthEvalAdapter(Protocol):
    """Adapter contract for the unified depth evaluator."""

    name: str
    prediction_kind: str

    def predict(self, image_rgb: np.ndarray, gt_depth: np.ndarray, sample_name: str) -> DepthPrediction:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


@dataclass
class DepthEvaluationResult:
    scores: list[float]
    sample_names: list[str]
    adapter_metadata: dict[str, Any]
    evaluator_version: str = "canonical_sirmse_v1"

    @property
    def count(self) -> int:
        return len(self.scores)

    def summary(self) -> dict[str, float | int | str]:
        if not self.scores:
            return {
                "evaluator_version": self.evaluator_version,
                "samples_evaluated": 0,
                "sirmse_mean": float("nan"),
                "sirmse_median": float("nan"),
                "sirmse_std": float("nan"),
                "sirmse_min": float("nan"),
                "sirmse_max": float("nan"),
            }
        arr = np.asarray(self.scores, dtype=np.float64)
        return {
            "evaluator_version": self.evaluator_version,
            "samples_evaluated": int(arr.size),
            "sirmse_mean": float(arr.mean()),
            "sirmse_median": float(np.median(arr)),
            "sirmse_std": float(arr.std()),
            "sirmse_min": float(arr.min()),
            "sirmse_max": float(arr.max()),
        }


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def tensor_from_prediction(values: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(device)
    return torch.from_numpy(np.asarray(values)).to(device)


def canonical_prediction_depth(
    prediction: DepthPrediction,
    gt_depth: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    values = tensor_from_prediction(prediction.values, device=device).float()
    values = resize_to_match(values, gt_depth)
    if prediction.kind == "disparity":
        return disparity_to_depth(values)
    if prediction.kind == "depth":
        return sanitize_direct_depth(values)
    raise ValueError(f"Unsupported prediction kind: {prediction.kind!r}. Expected 'depth' or 'disparity'.")


def select_eval_filenames(
    *,
    data_dir: str | Path,
    val_fraction: float,
    split_seed: int,
    split_file: str | Path | None = None,
    max_samples: int | None = None,
    fraction: float | None = None,
) -> list[str]:
    """Select canonical evaluation filenames and return the exact evaluated list."""
    all_names = discover_rgb_filenames(data_dir)
    _, val_names = make_or_load_split(
        sample_names=all_names,
        val_fraction=val_fraction,
        seed=split_seed,
        split_file=split_file,
    )
    filenames = list(val_names)
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        rng = np.random.default_rng(split_seed)
        count = max(1, int(len(filenames) * fraction))
        indices = np.sort(rng.choice(len(filenames), size=count, replace=False))
        filenames = [filenames[i] for i in indices]
    if max_samples is not None:
        filenames = filenames[:max_samples]
    return filenames


def evaluate_depth_adapter(
    *,
    adapter: DepthEvalAdapter,
    data_dir: str | Path,
    filenames: Iterable[str],
    device: torch.device,
    min_pixels: int = 10,
) -> DepthEvaluationResult:
    scores: list[float] = []
    evaluated_names: list[str] = []
    root = Path(data_dir)
    filename_list = list(filenames)
    for sample_name in tqdm(filename_list, desc=f"Evaluating {adapter.name}"):
        image_rgb, gt_np = load_rgb_depth(root, sample_name)
        prediction = adapter.predict(image_rgb, gt_np, sample_name)
        gt_t = torch.from_numpy(gt_np).to(device).float()
        valid_t = valid_depth_mask(gt_t)
        pred_depth = canonical_prediction_depth(prediction, gt_t, device=device)
        score = sirmse_from_depth(pred_depth, gt_t, valid_t, min_pixels=min_pixels)
        if score is None:
            continue
        scores.append(float(score.item()))
        evaluated_names.append(sample_name)
    return DepthEvaluationResult(
        scores=scores,
        sample_names=evaluated_names,
        adapter_metadata=adapter.metadata(),
    )


def write_evaluation_outputs(
    *,
    output_dir: str | Path,
    result: DepthEvaluationResult,
    config: dict[str, Any],
    checkpoint: str | Path | None,
    selected_filenames: list[str],
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = result.summary()
    payload = {
        "summary": summary,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "config_name": config.get("experiment", {}).get("name"),
        "model": config.get("model", {}),
        "data": config.get("data", {}),
        "adapter": result.adapter_metadata,
        "selected_sample_names": selected_filenames,
        "evaluated_sample_names": result.sample_names,
        "scores": result.scores,
    }
    with (out / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    lines = [
        f"Evaluator         : {summary['evaluator_version']}",
        f"Adapter           : {result.adapter_metadata.get('name')}",
        f"Prediction kind   : {result.adapter_metadata.get('prediction_kind')}",
        f"Samples selected  : {len(selected_filenames)}",
        f"Samples evaluated : {summary['samples_evaluated']}",
        f"siRMSE mean       : {summary['sirmse_mean']:.4f}",
        f"siRMSE median     : {summary['sirmse_median']:.4f}",
        f"siRMSE std        : {summary['sirmse_std']:.4f}",
        f"siRMSE min        : {summary['sirmse_min']:.4f}",
        f"siRMSE max        : {summary['sirmse_max']:.4f}",
        f"Checkpoint        : {checkpoint}",
    ]
    (out / "eval_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "sample_names.txt").write_text("\n".join(selected_filenames) + "\n", encoding="utf-8")
    return payload


def maybe_log_wandb_eval(config: dict[str, Any], output_dir: str | Path, summary: dict[str, Any], *, disabled: bool) -> None:
    if disabled:
        return
    logging_cfg = config.get("logging", {})
    if logging_cfg.get("backend", "wandb") != "wandb":
        return
    try:
        import wandb
    except ImportError:
        print("[warn] W&B requested but wandb is not installed. Run: python -m pip install -r requirements.txt")
        return
    if not getattr(wandb.api, "api_key", None):
        print("[warn] W&B requested but no login detected. Run `wandb login`; continuing local-only.")
        return
    run = wandb.init(
        entity=logging_cfg.get("entity"),
        project=logging_cfg.get("project"),
        name=f"eval_{config.get('experiment', {}).get('name', 'depth')}",
        tags=list(config.get("experiment", {}).get("tags", [])) + ["eval", "unified_eval"],
        config=config,
        dir=str(output_dir),
        job_type="eval",
    )
    run.log({f"eval/{key}": value for key, value in summary.items() if isinstance(value, (int, float))})
    run.summary.update(summary)
    run.finish()

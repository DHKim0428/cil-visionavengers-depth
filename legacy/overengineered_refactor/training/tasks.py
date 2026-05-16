from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.adapters import (
    adapter_parameter_count,
    apply_adapters_from_config,
    base_trainable_parameter_count,
)
from models.da2 import (
    apply_base_trainable_scope,
    checkpoint_path_for_encoder,
    load_da2_model,
    parameter_summary as da2_parameter_summary,
)
from models.unet import UNetBaseline, parameter_summary as unet_parameter_summary
from training.metrics import (
    disparity_to_depth,
    resize_to_match,
    sanitize_direct_depth,
    sirmse_from_depth,
    sirmse_from_disparity,
    sirmse_loss_from_depth_batch,
    sirmse_loss_from_disparity_batch,
)


class TrainingTask:
    """Small model-family hook surface consumed by the shared runner."""

    family: str
    prediction_kind: str
    model: torch.nn.Module
    checkpoint_reference: Path | None = None

    def predict_for_target(self, image: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, prediction: torch.Tensor, depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score(self, prediction: torch.Tensor, depth: torch.Tensor, valid_mask: torch.Tensor) -> float | None:
        raise NotImplementedError

    def prediction_to_depth(self, prediction: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def parameter_summary(self) -> dict[str, Any]:
        raise NotImplementedError

    def log_model_details(self, logger) -> None:
        summary = self.parameter_summary()
        logger.info(
            "Params      : total=%s trainable=%s frozen=%s trainable=%.2f%%",
            f"{summary['total']:,}",
            f"{summary['trainable']:,}",
            f"{summary['frozen']:,}",
            summary["trainable_pct"],
        )

    def summary_metadata(self) -> dict[str, Any]:
        return {"parameter_summary": self.parameter_summary(), "prediction_kind": self.prediction_kind}


class DA2TrainingTask(TrainingTask):
    family = "da2_relative"
    prediction_kind = "disparity"

    def __init__(self, *, model: torch.nn.Module, checkpoint_reference: Path, adapter_summary) -> None:
        self.model = model
        self.checkpoint_reference = checkpoint_reference
        self.adapter_summary = adapter_summary

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DA2TrainingTask":
        paths = config["paths"]
        checkpoint = (
            Path(paths["checkpoint"])
            if paths.get("checkpoint")
            else checkpoint_path_for_encoder(paths["da2_checkpoint_dir"], config["model"]["encoder"])
        )
        model = load_da2_model(
            encoder=config["model"]["encoder"],
            da2_repo=paths["da2_repo"],
            checkpoint_path=checkpoint,
            map_location="cpu",
        )
        apply_base_trainable_scope(model, config["base"]["trainable_scope"])
        adapter_summary = apply_adapters_from_config(model, config)
        return cls(model=model, checkpoint_reference=checkpoint, adapter_summary=adapter_summary)

    def predict_for_target(self, image: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
        return resize_to_match(self.model(image), target_depth, mode="bilinear", align_corners=False)

    def loss(self, prediction: torch.Tensor, depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return sirmse_loss_from_disparity_batch(prediction, depth, valid_mask)

    def score(self, prediction: torch.Tensor, depth: torch.Tensor, valid_mask: torch.Tensor) -> float | None:
        score = sirmse_from_disparity(prediction, depth, valid_mask)
        return None if score is None else float(score.item())

    def prediction_to_depth(self, prediction: torch.Tensor) -> torch.Tensor:
        return disparity_to_depth(prediction)

    def parameter_summary(self) -> dict[str, Any]:
        return da2_parameter_summary(self.model)

    def log_model_details(self, logger) -> None:
        super().log_model_details(logger)
        logger.info(
            "Trainable split: base=%s adapter=%s",
            f"{base_trainable_parameter_count(self.model):,}",
            f"{adapter_parameter_count(self.model):,}",
        )
        if self.adapter_summary.enabled:
            logger.info(
                "Adapter     : type=%s target=%s wrapped=%d params=%s",
                self.adapter_summary.adapter_type,
                self.adapter_summary.target_mode,
                self.adapter_summary.modules_wrapped,
                f"{self.adapter_summary.adapter_parameters:,}",
            )
            for module_name in self.adapter_summary.wrapped_module_names[:20]:
                logger.info("Adapter wraps: %s", module_name)
            if len(self.adapter_summary.wrapped_module_names) > 20:
                logger.info("Adapter wraps: ... and %d more modules", len(self.adapter_summary.wrapped_module_names) - 20)
        trainable_names = [name for name, param in self.model.named_parameters() if param.requires_grad]
        for name in trainable_names[:20]:
            logger.info("Trainable   : %s", name)
        if len(trainable_names) > 20:
            logger.info("Trainable   : ... and %d more tensors", len(trainable_names) - 20)

    def summary_metadata(self) -> dict[str, Any]:
        payload = super().summary_metadata()
        payload.update({
            "adapter_summary": self.adapter_summary.__dict__,
            "base_checkpoint": str(self.checkpoint_reference),
        })
        return payload


class UNetTrainingTask(TrainingTask):
    family = "unet"

    def __init__(self, *, model: UNetBaseline, prediction_kind: str) -> None:
        self.model = model
        self.prediction_kind = prediction_kind
        self.checkpoint_reference = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "UNetTrainingTask":
        prediction_kind = str(config["model"].get("prediction_kind", "disparity"))
        if prediction_kind not in {"disparity", "depth"}:
            raise ValueError("canonical U-Net model.prediction_kind must be one of: disparity, depth")
        return cls(model=UNetBaseline(prediction_kind=prediction_kind), prediction_kind=prediction_kind)

    def predict_for_target(self, image: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
        return resize_to_match(self.model(image), target_depth, mode="bilinear", align_corners=False)

    def loss(self, prediction: torch.Tensor, depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if self.prediction_kind == "disparity":
            return sirmse_loss_from_disparity_batch(prediction, depth, valid_mask)
        return sirmse_loss_from_depth_batch(prediction, depth, valid_mask)

    def score(self, prediction: torch.Tensor, depth: torch.Tensor, valid_mask: torch.Tensor) -> float | None:
        if self.prediction_kind == "disparity":
            score = sirmse_from_disparity(prediction, depth, valid_mask)
        else:
            score = sirmse_from_depth(prediction, depth, valid_mask)
        return None if score is None else float(score.item())

    def prediction_to_depth(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.prediction_kind == "disparity":
            return disparity_to_depth(prediction)
        return sanitize_direct_depth(prediction)

    def parameter_summary(self) -> dict[str, Any]:
        return unet_parameter_summary(self.model)


def build_training_task(config: dict[str, Any]) -> TrainingTask:
    family = config.get("model", {}).get("family")
    if family == "da2_relative":
        return DA2TrainingTask.from_config(config)
    if family == "unet":
        return UNetTrainingTask.from_config(config)
    raise ValueError(f"Unsupported model.family for training: {family!r}")

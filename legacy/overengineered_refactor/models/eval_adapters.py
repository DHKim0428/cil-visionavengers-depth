from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from dataset.tensors import normalize_imagenet, to_chw_image
from dataset.transform import PrepareForNet, Resize
from models.unet import UNetBaseline
from models.adapters import adapter_is_enabled, apply_adapters_from_config
from models.da2 import apply_base_trainable_scope, create_da2_model, extract_state_dict, load_da2_model
from training.checkpoints import (
    is_full_model_checkpoint,
    is_trainable_checkpoint,
    restore_model_payload,
)
from training.depth_evaluation import DepthPrediction


class BaseEvalAdapter:
    name = "base"
    prediction_kind = "depth"

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prediction_kind": self.prediction_kind,
        }


def build_da2_model_for_eval(
    *,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load official/full/trainable-only DA2 checkpoints for evaluation."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    if is_trainable_checkpoint(payload):
        base_checkpoint = payload.get("base_checkpoint")
        if not base_checkpoint:
            raise ValueError("Trainable-only checkpoint is missing base_checkpoint metadata")
        model = load_da2_model(
            encoder=config["model"]["encoder"],
            da2_repo=config["paths"]["da2_repo"],
            checkpoint_path=base_checkpoint,
            map_location="cpu",
        )
        apply_base_trainable_scope(model, config.get("base", {}).get("trainable_scope", "frozen"))
        apply_adapters_from_config(model, config)
        restore_model_payload(model, payload)
        return model.to(device).eval()

    if is_full_model_checkpoint(payload):
        model = create_da2_model(encoder=config["model"]["encoder"], da2_repo=config["paths"]["da2_repo"])
        apply_base_trainable_scope(model, config.get("base", {}).get("trainable_scope", "frozen"))
        apply_adapters_from_config(model, config)
        restore_model_payload(model, payload)
        return model.to(device).eval()

    if adapter_is_enabled(config):
        raise ValueError(
            "Config enables adapters, but checkpoint has no adapter payload. "
            "Evaluate a trainable-only adapter checkpoint or disable adapter.enabled."
        )
    model = create_da2_model(encoder=config["model"]["encoder"], da2_repo=config["paths"]["da2_repo"])
    model.load_state_dict(extract_state_dict(payload), strict=True)
    return model.to(device).eval()


class DA2RawInferAdapter(BaseEvalAdapter):
    name = "da2_raw_infer"
    prediction_kind = "disparity"

    def __init__(self, *, model: torch.nn.Module, input_size: int, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.input_size = int(input_size)
        self.device = device

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray, gt_depth: np.ndarray, sample_name: str) -> DepthPrediction:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        pred_disp = self.model.infer_image(image_bgr, self.input_size)
        return DepthPrediction(
            values=pred_disp,
            kind="disparity",
            label="DA2 raw infer disparity",
            metadata={"input_size": self.input_size},
        )

    def metadata(self) -> dict[str, Any]:
        out = super().metadata()
        out.update({"input_size": self.input_size, "preprocess": "da2_infer_image"})
        return out


class DA2TensorAdapter(BaseEvalAdapter):
    name = "da2_tensor"
    prediction_kind = "disparity"

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        input_size: int,
        preprocess: str,
        device: torch.device,
    ) -> None:
        if preprocess not in {"dpt_lower_bound", "square"}:
            raise ValueError(f"Unsupported DA2 tensor preprocess: {preprocess}")
        self.model = model.to(device).eval()
        self.input_size = int(input_size)
        self.preprocess = preprocess
        self.device = device
        self.prepare = PrepareForNet()
        self.resize = Resize(
            width=self.input_size,
            height=self.input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )

    def _preprocess_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        image = image_rgb.astype(np.float32) / 255.0
        if self.preprocess == "square":
            image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            tensor = to_chw_image(image)
        else:
            sample = self.resize({"image": image})
            sample = self.prepare(sample)
            tensor = torch.from_numpy(sample["image"]).float()
        tensor = normalize_imagenet(tensor)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray, gt_depth: np.ndarray, sample_name: str) -> DepthPrediction:
        image_t = self._preprocess_image(image_rgb)
        pred_disp = self.model(image_t)[0].detach().cpu()
        return DepthPrediction(
            values=pred_disp,
            kind="disparity",
            label="DA2 tensor disparity",
            metadata={"input_size": self.input_size, "preprocess": self.preprocess},
        )

    def metadata(self) -> dict[str, Any]:
        out = super().metadata()
        out.update({"input_size": self.input_size, "preprocess": self.preprocess})
        return out


class UNetBaselineAdapter(BaseEvalAdapter):
    name = "unet_baseline"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        input_size: int = 128,
        max_depth: float = 80.0,
        prediction_kind: str = "legacy_normalized_depth",
        device: torch.device,
    ) -> None:
        if prediction_kind not in {"legacy_normalized_depth", "disparity", "depth"}:
            raise ValueError(f"Unsupported U-Net eval prediction_kind={prediction_kind!r}")
        self.input_size = int(input_size)
        self.max_depth = float(max_depth)
        self.prediction_kind = prediction_kind
        self.device = device
        self.model = UNetBaseline(prediction_kind=prediction_kind).to(device).eval()
        payload = torch.load(checkpoint_path, map_location="cpu")
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        self.model.load_state_dict(state, strict=True)
        self.checkpoint_path = str(checkpoint_path)
        self.prediction_kind_for_eval = "depth" if prediction_kind in {"legacy_normalized_depth", "depth"} else "disparity"

    def _preprocess_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        image = image_rgb.astype(np.float32) / 255.0
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray, gt_depth: np.ndarray, sample_name: str) -> DepthPrediction:
        image_t = self._preprocess_image(image_rgb)
        pred = self.model(image_t)[0, 0].detach().cpu()
        if self.prediction_kind == "legacy_normalized_depth":
            pred = pred * self.max_depth
            label = "U-Net legacy normalized depth"
        elif self.prediction_kind == "depth":
            label = "U-Net metric depth"
        else:
            label = "U-Net disparity"
        return DepthPrediction(
            values=pred,
            kind=self.prediction_kind_for_eval,
            label=label,
            metadata={"input_size": self.input_size, "prediction_kind": self.prediction_kind},
        )

    def metadata(self) -> dict[str, Any]:
        out = super().metadata()
        out.update({
            "input_size": self.input_size,
            "max_depth": self.max_depth,
            "checkpoint": self.checkpoint_path,
            "prediction_kind": self.prediction_kind,
        })
        return out


def da2_preprocess_from_config(config: dict[str, Any]) -> str:
    data = config.get("data", {})
    if data.get("eval_preprocess"):
        return str(data["eval_preprocess"])
    protocol = data.get("eval_protocol", "native_resolution")
    eval_view = (data.get("views") or {}).get("eval", {}) or {}
    if protocol == "legacy_square" or eval_view.get("resize_policy") == "square":
        return "square"
    return "dpt_lower_bound"


def build_eval_adapter(
    *,
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> BaseEvalAdapter:
    family = config.get("model", {}).get("family")
    data = config.get("data", {})
    if family == "da2_relative":
        if checkpoint_path is None:
            raise ValueError("DA2 evaluation requires a checkpoint path")
        model = build_da2_model_for_eval(config=config, checkpoint_path=checkpoint_path, device=device)
        if data.get("eval_protocol") == "raw_infer_native":
            return DA2RawInferAdapter(model=model, input_size=int(data.get("image_size", data.get("img_size", 518))), device=device)
        return DA2TensorAdapter(
            model=model,
            input_size=int(data.get("image_size", data.get("img_size", 518))),
            preprocess=da2_preprocess_from_config(config),
            device=device,
        )
    if family in {"unet", "unet_baseline"}:
        if checkpoint_path is None:
            raise ValueError("U-Net evaluation requires --checkpoint pointing to a checkpoint")
        prediction_kind = config.get("model", {}).get("prediction_kind")
        if prediction_kind is None:
            prediction_kind = "legacy_normalized_depth" if family == "unet_baseline" else "disparity"
        return UNetBaselineAdapter(
            checkpoint_path=checkpoint_path,
            input_size=int(data.get("image_size", data.get("img_size", 128))),
            max_depth=float(data.get("max_depth", 80.0)),
            prediction_kind=str(prediction_kind),
            device=device,
        )
    raise ValueError(f"Unsupported model.family for unified eval: {family!r}")

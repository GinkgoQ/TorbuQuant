"""Activation calibration for vLLM TurboQuant metadata."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from turboquant.integration.vllm.metadata import (
    CacheMetadata,
    CalibrationMetadata,
    LayerMetadata,
    TensorMetadata,
)
from turboquant.integration.vllm.recipe import canonical_recipe, outlier_count


PROJECTION_PATTERN = re.compile(
    r"(^|.*\.)layers\.(?P<layer>\d+)\.self_attn\.(?P<proj>[kv]_proj)$"
)


@dataclass(frozen=True)
class ModelShape:
    head_dim: int
    num_kv_heads: int
    num_hidden_layers: int
    layer_types: tuple[str, ...] | None = None
    quantization_config: dict[str, Any] | None = None


def load_prompts(path: str | Path, *, limit: int | None = None) -> list[str]:
    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError("prompt file is empty")
    return prompts


def prompts_sha256(prompts: list[str]) -> str:
    return hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()


def derive_model_shape(config: Any) -> ModelShape:
    text_config = getattr(config, "text_config", None)
    source = text_config if text_config is not None else config

    head_dim = getattr(source, "head_dim", None)
    hidden_size = getattr(source, "hidden_size", None)
    num_attention_heads = getattr(source, "num_attention_heads", None)
    num_kv_heads = getattr(source, "num_key_value_heads", num_attention_heads)
    num_hidden_layers = getattr(source, "num_hidden_layers", None)
    if head_dim is None:
        if hidden_size is None or num_attention_heads is None:
            raise ValueError("cannot derive head_dim from config")
        head_dim = int(hidden_size) // int(num_attention_heads)
    if num_kv_heads is None:
        raise ValueError("cannot derive num_kv_heads from config")
    if num_hidden_layers is None:
        raise ValueError("cannot derive num_hidden_layers from config")
    layer_types = getattr(source, "layer_types", None)
    if layer_types is not None:
        layer_types = tuple(str(item) for item in layer_types)
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None and text_config is not None:
        quantization_config = getattr(text_config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        quantization_config = None
    return ModelShape(
        head_dim=int(head_dim),
        num_kv_heads=int(num_kv_heads),
        num_hidden_layers=int(num_hidden_layers),
        layer_types=layer_types,
        quantization_config=quantization_config,
    )


def is_quantized_model(quantization_config: dict[str, Any] | None) -> bool:
    return isinstance(quantization_config, dict) and bool(quantization_config)


def validate_calibration_model_choice(
    *,
    target_model: str,
    calibration_model: str,
    quantization_config: dict[str, Any] | None,
) -> None:
    if calibration_model != target_model:
        return
    if not is_quantized_model(quantization_config):
        return
    quant_method = quantization_config.get("quant_method", "unknown")
    raise ValueError(
        "calibration must use a non-quantized checkpoint when the target "
        f"checkpoint is quantized ({quant_method})"
    )


def resolve_layer_indices(num_hidden_layers: int, layer_types: tuple[str, ...] | None) -> list[int]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if layer_types is None:
        return list(range(num_hidden_layers))
    if len(layer_types) != num_hidden_layers:
        raise ValueError("layer_types length does not match num_hidden_layers")
    return [
        idx
        for idx, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    ]


def discover_projection_modules(
    model: torch.nn.Module,
    required_layer_indices: list[int],
) -> dict[tuple[int, str], torch.nn.Module]:
    required = set(required_layer_indices)
    modules: dict[tuple[int, str], torch.nn.Module] = {}
    for module_name, module in model.named_modules():
        match = PROJECTION_PATTERN.match(module_name)
        if match is None:
            continue
        layer_idx = int(match.group("layer"))
        if layer_idx not in required:
            continue
        tensor_name = "key" if match.group("proj") == "k_proj" else "value"
        modules[(layer_idx, tensor_name)] = module
    missing = [
        (layer_idx, tensor_name)
        for layer_idx in required_layer_indices
        for tensor_name in ("key", "value")
        if (layer_idx, tensor_name) not in modules
    ]
    if missing:
        names = ", ".join(f"layer {idx} {tensor_name}" for idx, tensor_name in missing)
        raise ValueError(f"missing projection modules: {names}")
    return modules


def select_high_precision_indices(
    channel_scores: torch.Tensor,
    count: int,
) -> tuple[tuple[int, ...], ...]:
    if channel_scores.ndim != 2:
        raise ValueError("channel_scores must have shape [kv_heads, head_dim]")
    if count <= 0 or count >= channel_scores.shape[-1]:
        raise ValueError("count must be in the channel range")
    tie_break = torch.linspace(
        0.0,
        1e-9,
        channel_scores.shape[-1],
        dtype=channel_scores.dtype,
        device=channel_scores.device,
    )
    selected = torch.topk(channel_scores + tie_break.unsqueeze(0), k=count, dim=-1).indices
    selected = torch.sort(selected, dim=-1).values.cpu()
    return tuple(tuple(int(item) for item in row.tolist()) for row in selected)


def tensor_metadata_from_scores(channel_scores: torch.Tensor, count: int) -> TensorMetadata:
    return TensorMetadata(select_high_precision_indices(channel_scores, count))


class ActivationAccumulator:
    def __init__(self, *, num_kv_heads: int, head_dim: int):
        if num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("num_kv_heads and head_dim must be positive")
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.current_attention_mask: torch.Tensor | None = None
        self.channel_scores: dict[tuple[int, str], torch.Tensor] = {}

    def set_attention_mask(self, attention_mask: torch.Tensor) -> None:
        self.current_attention_mask = attention_mask

    def clear_attention_mask(self) -> None:
        self.current_attention_mask = None

    def hook(self, layer_idx: int, tensor_name: str):
        if tensor_name not in {"key", "value"}:
            raise ValueError("tensor_name must be key or value")

        def _hook(_module, _inputs, output):
            projected = output[0] if isinstance(output, tuple) else output
            if projected.ndim != 3:
                raise ValueError("projection output must have shape [batch, seq, hidden]")
            expected_hidden = self.num_kv_heads * self.head_dim
            if projected.shape[-1] != expected_hidden:
                raise ValueError(
                    f"projection hidden size must be {expected_hidden}, got {projected.shape[-1]}"
                )
            if self.current_attention_mask is None:
                raise RuntimeError("attention mask was not set before calibration forward")
            flat_mask = self.current_attention_mask.reshape(-1).to(torch.bool)
            flat_projected = projected.detach().to(torch.float32).reshape(
                -1,
                self.num_kv_heads,
                self.head_dim,
            )
            if flat_projected.shape[0] != flat_mask.numel():
                raise ValueError("projection output and attention mask token counts differ")
            valid = flat_projected[flat_mask]
            scores = valid.square().sum(dim=0).cpu()
            key = (int(layer_idx), tensor_name)
            prior = self.channel_scores.get(key)
            self.channel_scores[key] = scores if prior is None else prior + scores

        return _hook


def build_calibrated_metadata(
    *,
    recipe: str,
    head_dim: int,
    model_name: str | None,
    num_hidden_layers: int,
    layer_types: tuple[str, ...] | None,
    layer_pattern: str,
    num_kv_heads: int,
    calibration_scores: dict[tuple[int, str], torch.Tensor],
    calibration: CalibrationMetadata,
) -> CacheMetadata:
    recipe = canonical_recipe(recipe)
    high_count = outlier_count(head_dim, recipe)
    layers: dict[str, LayerMetadata] = {}
    for layer_idx in resolve_layer_indices(num_hidden_layers, layer_types):
        key_scores = calibration_scores.get((layer_idx, "key"))
        value_scores = calibration_scores.get((layer_idx, "value"))
        if key_scores is None or value_scores is None:
            raise ValueError(f"missing calibration scores for layer {layer_idx}")
        expected_shape = (num_kv_heads, head_dim)
        if tuple(key_scores.shape) != expected_shape:
            raise ValueError(f"key score shape for layer {layer_idx} must be {expected_shape}")
        if tuple(value_scores.shape) != expected_shape:
            raise ValueError(f"value score shape for layer {layer_idx} must be {expected_shape}")
        layer_name = layer_pattern.format(i=layer_idx)
        layers[layer_name] = LayerMetadata(
            key=tensor_metadata_from_scores(key_scores, high_count),
            value=tensor_metadata_from_scores(value_scores, high_count),
        )
    return CacheMetadata(
        recipe=recipe,
        head_dim=head_dim,
        model_name=model_name,
        layers=layers,
        calibration=calibration,
    )

"""Metadata JSON for vLLM TurboQuant channel groups."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from torbuquant.integration.vllm.recipe import canonical_recipe, outlier_count


METADATA_VERSION = 1
TRANSFORM_VERSION = "structured_hadamard_v1"
CODEBOOK_VERSION = "lloyd_beta_v1"


@dataclass(frozen=True)
class TensorMetadata:
    high_precision_indices: tuple[tuple[int, ...], ...]

    def get_group_indices(
        self,
        *,
        device: torch.device,
        head_dim: int,
        recipe: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        high_cpu, low_cpu = _cached_group_indices(
            self.high_precision_indices,
            head_dim,
            recipe,
        )
        if device.type == "cpu":
            return high_cpu, low_cpu
        return high_cpu.to(device=device), low_cpu.to(device=device)

    def to_json(self) -> list[list[int]]:
        return [list(indices) for indices in self.high_precision_indices]


@lru_cache(maxsize=None)
def _cached_group_indices(
    high_precision_indices: tuple[tuple[int, ...], ...],
    head_dim: int,
    recipe: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    high_count = outlier_count(head_dim, recipe)
    if len(high_precision_indices) == 0:
        raise ValueError("TurboQuant metadata must include at least one KV head")

    all_idx = torch.arange(head_dim, dtype=torch.int64)
    high_groups: list[torch.Tensor] = []
    low_groups: list[torch.Tensor] = []
    for head_idx, raw_indices in enumerate(high_precision_indices):
        if len(raw_indices) != high_count:
            raise ValueError(
                "high precision group size mismatch for "
                f"head {head_idx}: expected {high_count}, got {len(raw_indices)}"
            )
        high = torch.tensor(raw_indices, dtype=torch.int64)
        if torch.any(high[:-1] >= high[1:]):
            raise ValueError("high precision indices must be sorted and unique")
        if int(high.min()) < 0 or int(high.max()) >= head_dim:
            raise ValueError("high precision indices are out of range")
        low_mask = torch.ones(head_dim, dtype=torch.bool)
        low_mask.scatter_(0, high, False)
        high_groups.append(high)
        low_groups.append(all_idx[low_mask])

    return (
        torch.stack(high_groups, dim=0).contiguous(),
        torch.stack(low_groups, dim=0).contiguous(),
    )


@dataclass(frozen=True)
class LayerMetadata:
    key: TensorMetadata
    value: TensorMetadata

    def to_json(self) -> dict[str, list[list[int]]]:
        return {
            "key_high_precision_indices": self.key.to_json(),
            "value_high_precision_indices": self.value.to_json(),
        }


@dataclass(frozen=True)
class CalibrationMetadata:
    method: str
    objective: str
    num_prompts: int
    max_seq_len: int
    batch_size: int
    num_observed_tokens: int
    dtype: str
    device: str
    prompts_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "objective": self.objective,
            "num_prompts": self.num_prompts,
            "max_seq_len": self.max_seq_len,
            "batch_size": self.batch_size,
            "num_observed_tokens": self.num_observed_tokens,
            "dtype": self.dtype,
            "device": self.device,
            "prompts_sha256": self.prompts_sha256,
        }


@dataclass(frozen=True)
class CacheMetadata:
    recipe: str
    head_dim: int
    model_name: str | None
    layers: dict[str, LayerMetadata]
    calibration: CalibrationMetadata | None = None
    version: int = METADATA_VERSION
    transform_version: str = TRANSFORM_VERSION
    codebook_version: str = CODEBOOK_VERSION

    def get_layer(self, layer_name: str) -> LayerMetadata:
        for candidate in _layer_name_candidates(layer_name):
            layer = self.layers.get(candidate)
            if layer is not None:
                return layer
        raise KeyError(f"TurboQuant metadata does not include layer {layer_name!r}")

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "recipe": self.recipe,
            "head_size": self.head_dim,
            "model_name": self.model_name,
            "transform_version": self.transform_version,
            "codebook_version": self.codebook_version,
            "layers": {
                layer_name: layer_metadata.to_json()
                for layer_name, layer_metadata in self.layers.items()
            },
        }
        if self.calibration is not None:
            payload["calibration"] = self.calibration.to_json()
        return payload


def slice_layer_metadata_for_tp(
    layer_metadata: LayerMetadata,
    *,
    num_kv_heads: int,
    tp_rank: int,
    tp_size: int,
) -> LayerMetadata:
    key_head_count = len(layer_metadata.key.high_precision_indices)
    value_head_count = len(layer_metadata.value.high_precision_indices)
    if key_head_count != value_head_count:
        raise ValueError(
            "key and value metadata KV head counts differ: "
            f"{key_head_count} vs {value_head_count}"
        )
    if key_head_count == num_kv_heads:
        return layer_metadata
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")

    total_heads = key_head_count
    if total_heads >= tp_size:
        if total_heads % tp_size != 0:
            raise ValueError("metadata KV head count does not divide tp_size")
        expected_heads = total_heads // tp_size
        if expected_heads != num_kv_heads:
            raise ValueError(
                "metadata KV head count does not match per-rank KV heads: "
                f"{expected_heads} vs {num_kv_heads}"
            )
        start = tp_rank * num_kv_heads
        end = start + num_kv_heads
    else:
        if tp_size % total_heads != 0:
            raise ValueError("tp_size does not divide metadata KV head count")
        if num_kv_heads != 1:
            raise ValueError(f"replicated metadata expects one KV head, got {num_kv_heads}")
        replicas = tp_size // total_heads
        start = tp_rank // replicas
        end = start + 1

    return LayerMetadata(
        key=TensorMetadata(layer_metadata.key.high_precision_indices[start:end]),
        value=TensorMetadata(layer_metadata.value.high_precision_indices[start:end]),
    )


def metadata_from_json(payload: dict[str, Any]) -> CacheMetadata:
    version = int(payload.get("version", METADATA_VERSION))
    if version != METADATA_VERSION:
        raise ValueError(f"unknown TurboQuant metadata version {version}")

    recipe = payload.get("recipe")
    if not isinstance(recipe, str):
        raise ValueError("metadata recipe must be a string")
    recipe = canonical_recipe(recipe)
    head_size = payload.get("head_size", payload.get("head_dim"))
    if not isinstance(head_size, int):
        raise ValueError("metadata head_size must be an integer")
    model_name = payload.get("model_name")
    if model_name is not None and not isinstance(model_name, str):
        raise ValueError("metadata model_name must be a string or null")
    layers_payload = payload.get("layers")
    if not isinstance(layers_payload, dict):
        raise ValueError("metadata layers must be an object")

    layers: dict[str, LayerMetadata] = {}
    for layer_name, layer_payload in layers_payload.items():
        if not isinstance(layer_name, str) or not isinstance(layer_payload, dict):
            raise ValueError("metadata layer entries must be named objects")
        layers[layer_name] = LayerMetadata(
            key=_parse_tensor_metadata(
                layer_payload.get("key_high_precision_indices"),
                "key_high_precision_indices",
            ),
            value=_parse_tensor_metadata(
                layer_payload.get("value_high_precision_indices"),
                "value_high_precision_indices",
            ),
        )

    calibration_payload = payload.get("calibration")
    calibration = None
    if calibration_payload is not None:
        if not isinstance(calibration_payload, dict):
            raise ValueError("metadata calibration must be an object or null")
        calibration = CalibrationMetadata(
            method=str(calibration_payload.get("method", "")),
            objective=str(calibration_payload.get("objective", "")),
            num_prompts=int(calibration_payload.get("num_prompts", 0)),
            max_seq_len=int(calibration_payload.get("max_seq_len", 0)),
            batch_size=int(calibration_payload.get("batch_size", 0)),
            num_observed_tokens=int(calibration_payload.get("num_observed_tokens", 0)),
            dtype=str(calibration_payload.get("dtype", "")),
            device=str(calibration_payload.get("device", "")),
            prompts_sha256=str(calibration_payload.get("prompts_sha256", "")),
        )

    return CacheMetadata(
        recipe=recipe,
        head_dim=head_size,
        model_name=model_name,
        layers=layers,
        calibration=calibration,
        transform_version=str(payload.get("transform_version", TRANSFORM_VERSION)),
        codebook_version=str(payload.get("codebook_version", CODEBOOK_VERSION)),
        version=version,
    )


@lru_cache(maxsize=None)
def load_metadata(path: str) -> CacheMetadata:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("metadata root must be an object")
    return metadata_from_json(payload)


def save_metadata(metadata: CacheMetadata, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_json(), f, indent=2, sort_keys=True)
        f.write("\n")


def discover_metadata_path(model_name_or_path: str | None, explicit_path: str | None) -> str | None:
    if explicit_path is not None:
        return str(Path(explicit_path).expanduser().resolve())
    if model_name_or_path is None:
        return None
    model_path = Path(model_name_or_path).expanduser().resolve()
    if model_path.is_file():
        model_path = model_path.parent
    elif not model_path.is_dir():
        return None
    metadata_path = model_path / "turboquant_kv.json"
    if metadata_path.is_file():
        return str(metadata_path.resolve())
    return None


def build_metadata(
    *,
    recipe: str,
    head_dim: int,
    num_kv_heads: int,
    layer_names: list[str],
    model_name: str | None = None,
) -> CacheMetadata:
    high_count = outlier_count(head_dim, recipe)
    high = tuple(tuple(range(high_count)) for _ in range(num_kv_heads))
    layer_metadata = LayerMetadata(
        key=TensorMetadata(high),
        value=TensorMetadata(high),
    )
    return CacheMetadata(
        recipe=recipe,
        head_dim=head_dim,
        model_name=model_name,
        layers={layer_name: layer_metadata for layer_name in layer_names},
    )


def _parse_tensor_metadata(payload: object, field_name: str) -> TensorMetadata:
    if not isinstance(payload, list):
        raise ValueError(f"metadata field {field_name!r} must be a list")
    rows: list[tuple[int, ...]] = []
    for item in payload:
        if not isinstance(item, list) or not all(isinstance(index, int) for index in item):
            raise ValueError(f"metadata field {field_name!r} must contain integer lists")
        rows.append(tuple(item))
    return TensorMetadata(tuple(rows))


def _layer_name_candidates(layer_name: str) -> tuple[str, ...]:
    candidates: list[str] = []

    def add(name: str) -> None:
        if name not in candidates:
            candidates.append(name)

    add(layer_name)
    if layer_name.endswith(".attn"):
        add(layer_name.removesuffix(".attn"))
    if layer_name.startswith("language_model."):
        stripped = layer_name.removeprefix("language_model.")
        add(stripped)
        if stripped.endswith(".attn"):
            add(stripped.removesuffix(".attn"))
    return tuple(candidates)

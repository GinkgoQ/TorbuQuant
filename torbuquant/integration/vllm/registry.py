"""vLLM cache dtype registry for TurboQuant modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torbuquant.integration.vllm.metadata import discover_metadata_path, load_metadata
from torbuquant.integration.vllm.recipe import (
    RecipeName,
    canonical_recipe,
    is_turboquant_dtype,
    recipe_cuda_allowed,
)


TQ4_DTYPES = frozenset({"tq4", "tq4_kv", "k4v4"})
RECIPE_DTYPES = frozenset({"turboquant25", "turboquant35"})


@dataclass(frozen=True)
class CacheDTypeInfo:
    name: str
    storage_dtype: torch.dtype
    mode: str
    requires_metadata: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "storage_dtype": str(self.storage_dtype).replace("torch.", ""),
            "mode": self.mode,
            "requires_metadata": self.requires_metadata,
        }


def is_tq4_dtype(cache_dtype: str | None) -> bool:
    return cache_dtype in TQ4_DTYPES


def is_recipe_dtype(cache_dtype: str | None) -> bool:
    return is_turboquant_dtype(cache_dtype)


def is_turboquant_cache_dtype(cache_dtype: str | None) -> bool:
    return is_tq4_dtype(cache_dtype) or is_recipe_dtype(cache_dtype)


def cache_dtype_info(cache_dtype: str) -> CacheDTypeInfo:
    if is_tq4_dtype(cache_dtype):
        return CacheDTypeInfo(
            name=cache_dtype,
            storage_dtype=torch.uint8,
            mode="tq4",
            requires_metadata=False,
        )
    if is_recipe_dtype(cache_dtype):
        return CacheDTypeInfo(
            name=canonical_recipe(cache_dtype),
            storage_dtype=torch.uint8,
            mode="recipe",
            requires_metadata=True,
        )
    raise ValueError(f"unknown TurboQuant cache dtype: {cache_dtype}")


def validate_runtime_gate(
    *,
    cache_dtype: str,
    enabled: bool,
    capability: Any | None = None,
) -> CacheDTypeInfo:
    info = cache_dtype_info(cache_dtype)
    if not enabled:
        raise ValueError("TurboQuant cache dtype requires enable_turboquant=True")
    if capability is not None and not recipe_cuda_allowed(capability):
        raise ValueError("CUDA capability is not supported for this TurboQuant path")
    return info


def resolve_metadata(
    *,
    cache_dtype: RecipeName | str,
    model_name_or_path: str | None,
    metadata_path: str | None,
    head_dim: int,
):
    recipe = canonical_recipe(cache_dtype)
    path = discover_metadata_path(model_name_or_path, metadata_path)
    if path is None:
        raise FileNotFoundError("TurboQuant recipe mode requires turboquant_kv.json")
    metadata = load_metadata(path)
    if metadata.recipe != recipe:
        raise ValueError(f"metadata recipe {metadata.recipe} does not match {recipe}")
    if metadata.head_dim != head_dim:
        raise ValueError(f"metadata head_dim {metadata.head_dim} does not match {head_dim}")
    return metadata

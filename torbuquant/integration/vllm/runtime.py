"""Runtime dispatch helpers for vLLM TurboQuant integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torbuquant.integration.vllm.metadata import CacheMetadata
from torbuquant.integration.vllm.page_cache import PageGeometry
from torbuquant.integration.vllm.registry import cache_dtype_info, resolve_metadata
from torbuquant.integration.vllm.recipe import env_flag, parse_tq_bits_env, recipe_packed_dim


DecodePath = Literal["delegate", "tq4_paged", "recipe_paged"]
PrefillPath = Literal["dense_live", "packed_cache", "int8_query"]


@dataclass(frozen=True)
class VLLMRuntimeConfig:
    cache_dtype: str
    enabled: bool
    model_name_or_path: str | None
    metadata_path: str | None
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    k_bits: int = 4
    v_bits: int = 4


@dataclass(frozen=True)
class VLLMLayerContext:
    config: VLLMRuntimeConfig
    layer_name: str
    metadata: CacheMetadata | None


def parse_tq_gate_env(environ: dict[str, str] | None = None) -> dict[str, bool]:
    return {
        "fused_paged": env_flag("TQ4_USE_FUSED_PAGED", environ),
        "int8_prefill": env_flag("TQ4_USE_INT8_PREFILL", environ),
    }


def select_decode_path(cache_dtype: str) -> DecodePath:
    info = cache_dtype_info(cache_dtype)
    if info.mode == "tq4":
        return "tq4_paged"
    if info.mode == "recipe":
        return "recipe_paged"
    return "delegate"


def select_prefill_path(
    *,
    has_cache: bool,
    int8_enabled: bool,
    mixed_with_cache: bool,
) -> PrefillPath:
    if int8_enabled and has_cache and not mixed_with_cache:
        return "int8_query"
    if has_cache and mixed_with_cache:
        return "packed_cache"
    return "dense_live"


def requires_cuda_graph_buffers(*, decode_path: DecodePath, single_token_decode: bool) -> bool:
    return single_token_decode and decode_path in {"tq4_paged", "recipe_paged"}


def build_layer_context(
    *,
    config: VLLMRuntimeConfig,
    layer_name: str,
) -> VLLMLayerContext:
    info = cache_dtype_info(config.cache_dtype)
    metadata = None
    if info.requires_metadata:
        metadata = resolve_metadata(
            cache_dtype=config.cache_dtype,
            model_name_or_path=config.model_name_or_path,
            metadata_path=config.metadata_path,
            head_dim=config.head_dim,
        )
        metadata.get_layer(layer_name)
    return VLLMLayerContext(config=config, layer_name=layer_name, metadata=metadata)


def build_page_geometry(config: VLLMRuntimeConfig, *, num_blocks: int, padded: bool = False) -> PageGeometry:
    if cache_dtype_info(config.cache_dtype).mode == "recipe":
        row_bytes = recipe_packed_dim(config.head_dim, config.cache_dtype)
        return PageGeometry(
            num_blocks=num_blocks,
            block_size=config.block_size,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            k_bits=4,
            v_bits=4,
            padded=False,
            k_row_override=row_bytes,
            v_row_override=row_bytes,
        )
    k_bits, v_bits = parse_tq_bits_env(
        {"TQ4_K_BITS": str(config.k_bits), "TQ4_V_BITS": str(config.v_bits)}
    )
    return PageGeometry(
        num_blocks=num_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        k_bits=k_bits,
        v_bits=v_bits,
        padded=padded,
    )


def kv_head_for_query_head(num_q_heads: int, num_kv_heads: int, device: torch.device) -> torch.Tensor:
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    return torch.arange(num_q_heads, device=device, dtype=torch.int64) // (num_q_heads // num_kv_heads)

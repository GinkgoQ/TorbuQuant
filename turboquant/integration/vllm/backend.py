"""vLLM backend-facing entry points."""

from __future__ import annotations

import torch

from torbuquant.integration.common import KVQuantConfig, OptionalDependencyError
from torbuquant.integration.vllm.cache_spec import PagedCacheSpec
from torbuquant.integration.vllm.config import vllm_settings
from torbuquant.integration.vllm.ops import PagedCompressedKVCache


def require_vllm() -> object:
    try:
        import vllm
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise OptionalDependencyError("vllm is required for vLLM runtime registration") from exc
    return vllm


def build_paged_cache(
    *,
    config: KVQuantConfig,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> PagedCompressedKVCache:
    settings = vllm_settings(config, block_size=block_size)
    spec = PagedCacheSpec(
        block_size=settings.block_size,
        num_kv_heads=settings.num_kv_heads,
        head_dim=settings.head_dim,
        k_format=settings.k_format,
        v_format=settings.v_format,
    )
    return PagedCompressedKVCache(config=config, spec=spec, device=device, dtype=dtype)


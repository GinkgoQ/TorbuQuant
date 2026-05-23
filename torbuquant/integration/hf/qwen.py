"""Qwen cache integration helpers for HuggingFace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torbuquant.integration.common import KVQuantConfig, ProductionModeError, detect_backend, integration_report, policy_from_config
from torbuquant.kv import CompressedKVCache


@dataclass
class HFDiagnosticCacheAdapter:
    """Diagnostic adapter that stores compressed cache state and returns dense tensors to HF."""

    config: KVQuantConfig
    device: torch.device
    dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.config.mode == "production":
            raise ProductionModeError("HF DynamicCache adapters are diagnostic because they return dense tensors")
        self.policy = policy_from_config(self.config)
        self.layers: dict[int, CompressedKVCache] = {}
        self.fallback_count = int(self.policy.fallback_count)
        self.diagnostic_label = "diagnostic_dequant"

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("HF cache update expects (batch, heads, tokens, head_dim)")
        if key_states.shape != value_states.shape:
            raise ValueError("key/value shape mismatch")
        if key_states.shape[0] != 1:
            raise ValueError("diagnostic HF adapter currently expects batch size 1")

        cache = self.layers.get(layer_idx)
        if cache is None:
            cache = CompressedKVCache(
                head_dim=int(key_states.shape[-1]),
                num_kv_heads=int(key_states.shape[1]),
                layer_idx=layer_idx,
                policy=self.policy,
                device=self.device,
                dtype=self.dtype,
                num_layers=self.config.num_layers,
                seed=self.config.seed,
            )
            self.layers[layer_idx] = cache
        keys = key_states[0].transpose(0, 1).contiguous()
        values = value_states[0].transpose(0, 1).contiguous()
        if keys.shape[0] == 1:
            cache.append_decode(keys, values)
        else:
            cache.prefill(keys, values)
        dense = cache.diagnostic_dense()
        self.diagnostic_label = dense.label
        out_k = dense.keys.to(device=key_states.device, dtype=key_states.dtype).unsqueeze(0).transpose(1, 2)
        out_v = dense.values.to(device=value_states.device, dtype=value_states.dtype).unsqueeze(0).transpose(1, 2)
        return out_k.contiguous(), out_v.contiguous()

    def report(self) -> dict[str, Any]:
        capability = detect_backend("hf")
        memory = None
        cache_report = None
        if self.layers:
            reports = [cache.to_report() for cache in self.layers.values()]
            memory = {
                str(layer_idx): cache.memory_report().as_dict()
                for layer_idx, cache in self.layers.items()
            }
            cache_report = {"layers": reports}
        return integration_report(
            config=self.config,
            capability=capability,
            k_format=self.policy.k_format,
            v_format=self.policy.v_format,
            memory=memory,
            cache=cache_report,
            fallback_count=self.fallback_count,
            fallback_mode=self.policy.fallback_mode,
            diagnostic_label=self.diagnostic_label,
        ).to_dict()


class DynamicCachePatch:
    """Context manager that patches a Transformers cache update method for diagnostics."""

    def __init__(self, cache: Any, adapter: HFDiagnosticCacheAdapter):
        self.cache = cache
        self.adapter = adapter
        self._original_update = None

    def __enter__(self) -> "DynamicCachePatch":
        if not hasattr(self.cache, "update"):
            raise TypeError("cache object must expose update")
        self._original_update = self.cache.update
        self.cache.update = self._update
        return self

    def __exit__(self, *exc: object) -> bool:
        if self._original_update is not None:
            self.cache.update = self._original_update
        return False

    def _update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        return self.adapter.update(key_states, value_states, layer_idx)


def build_layer_cache_from_capture(
    *,
    config: KVQuantConfig,
    keys: torch.Tensor,
    values: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> CompressedKVCache:
    if keys.ndim == 4:
        if keys.shape[0] != 1:
            raise ValueError("captured cache builder expects one batch item")
        keys = keys[0]
        values = values[0]
    if keys.ndim != 3:
        raise ValueError("keys must have shape (tokens, kv_heads, head_dim)")
    policy = policy_from_config(config)
    cache = CompressedKVCache(
        head_dim=int(keys.shape[-1]),
        num_kv_heads=int(keys.shape[1]),
        layer_idx=config.layer_idx,
        policy=policy,
        device=device,
        dtype=dtype,
        num_layers=config.num_layers,
        seed=config.seed,
    )
    cache.prefill(keys.to(device=device, dtype=dtype), values.to(device=device, dtype=dtype))
    return cache

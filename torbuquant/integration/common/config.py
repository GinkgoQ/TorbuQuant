"""Shared integration configuration translation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from torbuquant.integration.common.backend import BackendName, detect_backend
from torbuquant.kv.policy import KVQuantPolicy, PresetName, qwen25_3b_policy

ModeName = Literal["diagnostic", "production"]


@dataclass(frozen=True)
class KVQuantConfig:
    preset: PresetName = "k16_v4"
    backend: BackendName = "diagnostic"
    mode: ModeName = "diagnostic"
    model_id: str = "Qwen/Qwen2.5-3B"
    context_length: int = 4096
    batch_size: int = 1
    num_q_heads: int = 16
    num_kv_heads: int = 2
    head_dim: int = 128
    layer_idx: int = 0
    num_layers: int | None = None
    recent_window: int | None = None
    boundary_tokens: int | None = None
    sparse_v_threshold: float | None = None
    vram_budget_bytes: int | None = None
    quality_priority: Literal["quality", "memory"] = "quality"
    seed: int = 42
    local_files_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def policy_from_config(config: KVQuantConfig) -> KVQuantPolicy:
    capability = detect_backend(config.backend).policy_capability
    policy = qwen25_3b_policy(
        preset=config.preset,
        backend=capability,
        context_length=config.context_length,
        batch_size=config.batch_size,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        vram_budget_bytes=config.vram_budget_bytes,
        quality_priority=config.quality_priority,
    )
    if config.recent_window is None and config.boundary_tokens is None and config.sparse_v_threshold is None:
        return policy
    return KVQuantPolicy(
        preset=policy.preset,
        k_format=policy.k_format,
        v_format=policy.v_format,
        recent_window=policy.recent_window if config.recent_window is None else config.recent_window,
        boundary_tokens=policy.boundary_tokens if config.boundary_tokens is None else config.boundary_tokens,
        preserve_first_layers=policy.preserve_first_layers,
        preserve_last_layers=policy.preserve_last_layers,
        sparse_v=policy.sparse_v,
        sparse_v_threshold=policy.sparse_v_threshold if config.sparse_v_threshold is None else config.sparse_v_threshold,
        allow_v3=policy.allow_v3,
        allow_k_low_bits=policy.allow_k_low_bits,
        fallback_mode=policy.fallback_mode,
        fallback_count=policy.fallback_count,
        reason=policy.reason,
    )

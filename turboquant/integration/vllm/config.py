"""vLLM integration configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from turboquant.integration.common import KVQuantConfig, ProductionModeError, detect_backend, policy_from_config


@dataclass(frozen=True)
class VLLMSettings:
    model_id: str
    block_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    preset: str
    k_format: str
    v_format: str
    mode: str
    paged_cache: bool
    fused_decode: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def vllm_settings(config: KVQuantConfig, *, block_size: int = 16) -> VLLMSettings:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    capability = detect_backend("vllm")
    policy = policy_from_config(config)
    if config.mode == "production" and not capability.policy_capability.paged_cache:
        raise ProductionModeError("vLLM production mode requires paged cache support")
    return VLLMSettings(
        model_id=config.model_id,
        block_size=block_size,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        preset=config.preset,
        k_format=policy.k_format,
        v_format=policy.v_format,
        mode=config.mode,
        paged_cache=capability.policy_capability.paged_cache,
        fused_decode=capability.policy_capability.fused_decode,
        seed=config.seed,
    )


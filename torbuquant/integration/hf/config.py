"""HuggingFace integration configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from torbuquant.integration.common import KVQuantConfig, ProductionModeError, detect_backend, policy_from_config


@dataclass(frozen=True)
class HFQwenSettings:
    model_id: str
    layer_idx: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    preset: str
    k_format: str
    v_format: str
    mode: str
    local_files_only: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hf_qwen_settings(config: KVQuantConfig) -> HFQwenSettings:
    capability = detect_backend("hf")
    policy = policy_from_config(config)
    if config.mode == "production" and not capability.policy_capability.fused_decode:
        raise ProductionModeError("HF production mode requires compressed decode kernels")
    return HFQwenSettings(
        model_id=config.model_id,
        layer_idx=config.layer_idx,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        preset=config.preset,
        k_format=policy.k_format,
        v_format=policy.v_format,
        mode=config.mode,
        local_files_only=config.local_files_only,
        seed=config.seed,
    )


"""Paged compressed cache operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from turboquant.core.types import MSEData, ValueData
from turboquant.integration.common import KVQuantConfig, detect_backend, integration_report, policy_from_config
from turboquant.integration.vllm.cache_spec import PagedCacheSpec
from turboquant.kv import CompressedKVBlock, CompressedKVCache, DenseKeyData, QuantizedKeyData, TurboKeyData
from turboquant.triton import KernelCounters, KernelResult, decode_block


@dataclass(frozen=True)
class PagedSlot:
    page_id: int
    offset: int

    @classmethod
    def from_flat(cls, slot: int, block_size: int) -> "PagedSlot":
        if slot < 0:
            raise ValueError("slot id must be non-negative")
        return cls(page_id=slot // block_size, offset=slot % block_size)

    def to_dict(self) -> dict[str, int]:
        return {"page_id": self.page_id, "offset": self.offset}


class PagedCompressedKVCache:
    """Owns compressed KV blocks addressed by vLLM-style flat slots."""

    def __init__(
        self,
        *,
        config: KVQuantConfig,
        spec: PagedCacheSpec,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.spec = spec
        self.device = device
        self.dtype = dtype
        base_policy = policy_from_config(config)
        self.policy = type(base_policy)(
            preset=base_policy.preset,
            k_format=base_policy.k_format,
            v_format=base_policy.v_format,
            recent_window=0,
            boundary_tokens=0,
            preserve_first_layers=base_policy.preserve_first_layers,
            preserve_last_layers=base_policy.preserve_last_layers,
            sparse_v=base_policy.sparse_v,
            allow_v3=base_policy.allow_v3,
            allow_k_low_bits=base_policy.allow_k_low_bits,
            fallback_mode=base_policy.fallback_mode,
            fallback_count=base_policy.fallback_count,
            reason=base_policy.reason,
        )
        self._writer = CompressedKVCache(
            head_dim=spec.head_dim,
            num_kv_heads=spec.num_kv_heads,
            layer_idx=config.layer_idx,
            policy=self.policy,
            device=device,
            dtype=dtype,
            chunk_size=1,
            num_layers=config.num_layers,
            seed=config.seed,
        )
        self._slots: dict[int, CompressedKVBlock] = {}
        self._writes = 0
        self._workspace_bytes = 0

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("paged write expects (tokens, kv_heads, head_dim)")
        if keys.shape != values.shape:
            raise ValueError("key/value shape mismatch")
        if keys.shape[1] != self.spec.num_kv_heads or keys.shape[2] != self.spec.head_dim:
            raise ValueError("key/value tensor shape does not match page spec")
        slots = slot_mapping.detach().cpu().reshape(-1).tolist()
        if len(slots) != int(keys.shape[0]):
            raise ValueError("slot_mapping length must match token count")

        for token_idx, slot in enumerate(slots):
            before = len(self._writer.blocks)
            self._writer.prefill(
                keys[token_idx : token_idx + 1].to(device=self.device, dtype=self.dtype),
                values[token_idx : token_idx + 1].to(device=self.device, dtype=self.dtype),
            )
            if len(self._writer.blocks) != before + 1:
                raise RuntimeError("paged writer did not emit one compressed block")
            self._slots[int(slot)] = self._writer.blocks[-1]
            self._writes += 1

    def slot(self, slot: int) -> CompressedKVBlock:
        try:
            return self._slots[int(slot)]
        except KeyError as exc:
            raise KeyError(f"slot {slot} has no compressed KV block") from exc

    def decode(
        self,
        query: torch.Tensor,
        slots: torch.Tensor,
        *,
        num_q_heads: int,
        scale: float,
        counters: KernelCounters | None = None,
    ) -> KernelResult:
        slot_list = [int(x) for x in slots.detach().cpu().reshape(-1).tolist()]
        if not slot_list:
            raise ValueError("decode requires at least one slot")
        blocks = [self.slot(slot) for slot in slot_list]
        block = merge_blocks(blocks)
        self._workspace_bytes += compressed_block_bytes(block)
        return decode_block(
            query,
            block,
            num_kv_heads=self.spec.num_kv_heads,
            scale=scale,
            quantizer=self._writer._k4_quantizer,
            counters=counters,
        )

    def to_report(self) -> dict[str, Any]:
        memory = self._writer.memory_report().as_dict()
        memory["workspace_bytes"] = self._workspace_bytes
        capability = detect_backend("vllm")
        return integration_report(
            config=self.config,
            capability=capability,
            k_format=self.policy.k_format,
            v_format=self.policy.v_format,
            memory=memory,
            cache={
                "spec": self.spec.to_dict(),
                "slots": len(self._slots),
                "writes": self._writes,
                "slot_ids": sorted(self._slots),
                "path": "paged_compressed_slots",
            },
            fallback_count=int(self.policy.fallback_count),
            fallback_mode=self.policy.fallback_mode,
        ).to_dict()


def compressed_block_bytes(block: CompressedKVBlock) -> int:
    total = 0
    for value in block.to_report().values():
        if isinstance(value, int):
            total += value
    return total


def merge_blocks(blocks: list[CompressedKVBlock]) -> CompressedKVBlock:
    if not blocks:
        raise ValueError("merge_blocks requires at least one block")
    first = blocks[0]
    if any(block.k_format != first.k_format or block.v_format != first.v_format for block in blocks):
        raise ValueError("cannot merge blocks with different formats")

    if isinstance(first.key, DenseKeyData):
        key = DenseKeyData(torch.cat([block.key.data for block in blocks], dim=0).contiguous(), first.k_format)
    elif isinstance(first.key, QuantizedKeyData):
        key = QuantizedKeyData(_cat_value_data([block.key.data for block in blocks]), first.k_format)
    elif isinstance(first.key, TurboKeyData):
        key = TurboKeyData(_cat_mse_data([block.key.data for block in blocks]), first.k_format)
    else:  # pragma: no cover - closed union
        raise TypeError(f"unknown key payload type: {type(first.key)!r}")
    value = _cat_value_data([block.value for block in blocks])
    return CompressedKVBlock(
        key=key,
        value=value,
        start=0,
        length=sum(block.length for block in blocks),
        k_format=first.k_format,
        v_format=first.v_format,
    )


def _cat_value_data(values: list[ValueData]) -> ValueData:
    first = values[0]
    return ValueData(
        data=torch.cat([value.data for value in values], dim=0).contiguous(),
        scales=torch.cat([value.scales for value in values], dim=0).contiguous(),
        zeros=torch.cat([value.zeros for value in values], dim=0).contiguous(),
        bits=first.bits,
        dim=first.dim,
        group_size=first.group_size,
    )


def _cat_mse_data(values: list[MSEData]) -> MSEData:
    first = values[0]
    return MSEData(
        indices=torch.cat([value.indices for value in values], dim=0).contiguous(),
        norms=torch.cat([value.norms for value in values], dim=0).contiguous(),
        bits=first.bits,
        dim=first.dim,
    )

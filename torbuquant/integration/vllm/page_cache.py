"""Uint8 paged cache storage for vLLM-style slot mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torbuquant.integration.vllm.recipe import (
    TQ_BITS,
    TQ_NORM_BYTES,
    padded_slot_bytes,
    tq_bytes_per_token,
    tq_bytes_per_token_kv,
)


@dataclass(frozen=True)
class PageGeometry:
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    k_bits: int = TQ_BITS
    v_bits: int = TQ_BITS
    padded: bool = False
    k_row_override: int | None = None
    v_row_override: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("num_blocks", self.num_blocks),
            ("block_size", self.block_size),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def k_row_bytes(self) -> int:
        if self.k_row_override is not None:
            return self.k_row_override
        return tq_bytes_per_token(self.head_dim, self.k_bits)

    @property
    def v_row_bytes(self) -> int:
        if self.v_row_override is not None:
            return self.v_row_override
        return tq_bytes_per_token(self.head_dim, self.v_bits)

    @property
    def row_bytes(self) -> int:
        if self.k_row_override is not None or self.v_row_override is not None:
            raw = self.k_row_bytes + self.v_row_bytes
            if self.padded:
                return 1 << (raw - 1).bit_length()
            return raw
        if self.padded:
            return padded_slot_bytes(self.head_dim, k_bits=self.k_bits, v_bits=self.v_bits)
        return tq_bytes_per_token_kv(self.head_dim, k_bits=self.k_bits, v_bits=self.v_bits)

    @property
    def dense_kv_bytes(self) -> int:
        return self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim * 2 * 2

    @property
    def packed_kv_bytes(self) -> int:
        return self.num_blocks * self.block_size * self.num_kv_heads * self.row_bytes

    def cache_shape(self) -> tuple[int, int, int, int]:
        return (self.num_blocks, self.block_size, self.num_kv_heads, self.row_bytes)

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "padded": self.padded,
            "k_row_override": self.k_row_override or 0,
            "v_row_override": self.v_row_override or 0,
            "k_row_bytes": self.k_row_bytes,
            "v_row_bytes": self.v_row_bytes,
            "row_bytes": self.row_bytes,
            "dense_kv_bytes": self.dense_kv_bytes,
            "packed_kv_bytes": self.packed_kv_bytes,
        }


class PackedPageCache:
    def __init__(
        self,
        *,
        geometry: PageGeometry,
        device: torch.device,
    ):
        self.geometry = geometry
        self.device = device
        self.storage = torch.zeros(
            geometry.cache_shape(),
            dtype=torch.uint8,
            device=device,
        )
        self.writes = 0

    @property
    def key_region(self) -> torch.Tensor:
        return self.storage[..., : self.geometry.k_row_bytes]

    @property
    def value_region(self) -> torch.Tensor:
        start = self.geometry.k_row_bytes
        return self.storage[..., start : start + self.geometry.v_row_bytes]

    def write_rows(
        self,
        key_rows: torch.Tensor,
        value_rows: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._check_rows(key_rows, value_rows, slot_mapping)
        slots = slot_mapping.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
        for token_idx, slot in enumerate(slots):
            if int(slot) < 0:
                continue
            page_id = int(slot) // self.geometry.block_size
            offset = int(slot) % self.geometry.block_size
            if page_id >= self.geometry.num_blocks:
                raise IndexError(f"slot {slot} exceeds allocated pages")
            self.key_region[page_id, offset].copy_(key_rows[token_idx].to(self.device))
            self.value_region[page_id, offset].copy_(value_rows[token_idx].to(self.device))
            self.writes += 1

    def read_slot(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        if slot < 0:
            raise ValueError("slot must be non-negative")
        page_id = int(slot) // self.geometry.block_size
        offset = int(slot) % self.geometry.block_size
        if page_id >= self.geometry.num_blocks:
            raise IndexError(f"slot {slot} exceeds allocated pages")
        return (
            self.key_region[page_id, offset].clone(),
            self.value_region[page_id, offset].clone(),
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry.to_dict(),
            "writes": self.writes,
            "storage_bytes": int(self.storage.numel() * self.storage.element_size()),
        }

    def _check_rows(
        self,
        key_rows: torch.Tensor,
        value_rows: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if key_rows.ndim != 3 or value_rows.ndim != 3:
            raise ValueError("rows must have shape [tokens, kv_heads, row_bytes]")
        if key_rows.shape[:2] != value_rows.shape[:2]:
            raise ValueError("key/value row prefixes differ")
        if key_rows.shape[1] != self.geometry.num_kv_heads:
            raise ValueError("row KV head count differs from geometry")
        if key_rows.shape[-1] != self.geometry.k_row_bytes:
            raise ValueError(f"key row width must be {self.geometry.k_row_bytes}")
        if value_rows.shape[-1] != self.geometry.v_row_bytes:
            raise ValueError(f"value row width must be {self.geometry.v_row_bytes}")
        if slot_mapping.reshape(-1).numel() != key_rows.shape[0]:
            raise ValueError("slot_mapping length must match token count")


def split_tq_row(row: torch.Tensor, *, head_dim: int, k_bits: int, v_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    k_bytes = tq_bytes_per_token(head_dim, k_bits)
    v_bytes = tq_bytes_per_token(head_dim, v_bits)
    if row.shape[-1] < k_bytes + v_bytes:
        raise ValueError("row is shorter than K/V byte regions")
    return row[..., :k_bytes], row[..., k_bytes : k_bytes + v_bytes]


def make_empty_rows(
    *,
    tokens: int,
    num_kv_heads: int,
    row_bytes: int,
    device: torch.device,
) -> torch.Tensor:
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    if num_kv_heads <= 0 or row_bytes <= 0:
        raise ValueError("num_kv_heads and row_bytes must be positive")
    return torch.zeros(tokens, num_kv_heads, row_bytes, dtype=torch.uint8, device=device)

"""Paged compressed cache layout for vLLM-style slot mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from turboquant.kv.formats import get_k_format, get_v_format
from turboquant.packing.bits import packed_length
from turboquant.integration.vllm.recipe import is_turboquant_dtype, packed_dim, recipe_layout


@dataclass(frozen=True)
class SlotComponentBytes:
    k_data: int
    k_scales: int
    k_zeros: int
    k_norms: int
    v_data: int
    v_scales: int
    v_zeros: int

    @property
    def total(self) -> int:
        return self.k_data + self.k_scales + self.k_zeros + self.k_norms + self.v_data + self.v_scales + self.v_zeros

    def to_dict(self) -> dict[str, int]:
        return asdict(self) | {"total": self.total}


@dataclass(frozen=True)
class PagedCacheSpec:
    block_size: int
    num_kv_heads: int
    head_dim: int
    k_format: str
    v_format: str
    group_size: int = 32
    scale_dtype_bytes: int = 2
    zero_dtype_bytes: int = 2
    norm_dtype_bytes: int = 2
    cache_dtype: str | None = None

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.cache_dtype is not None:
            if not is_turboquant_dtype(self.cache_dtype):
                raise ValueError(f"unknown cache_dtype: {self.cache_dtype}")
            packed_dim(self.head_dim, self.cache_dtype)
        else:
            get_k_format(self.k_format)
            get_v_format(self.v_format)

    @property
    def groups(self) -> int:
        return (self.head_dim + self.group_size - 1) // self.group_size

    @property
    def slot_components(self) -> SlotComponentBytes:
        if self.cache_dtype is not None:
            data = self.num_kv_heads * packed_dim(self.head_dim, self.cache_dtype)
            return SlotComponentBytes(
                k_data=data,
                k_scales=0,
                k_zeros=0,
                k_norms=0,
                v_data=data,
                v_scales=0,
                v_zeros=0,
            )
        if self.k_format == "K16":
            k_data = self.num_kv_heads * self.head_dim * 2
            k_scales = k_zeros = k_norms = 0
        elif self.k_format == "K8":
            k_data = self.num_kv_heads * self.head_dim
            k_scales = self.num_kv_heads * self.groups * self.scale_dtype_bytes
            k_zeros = self.num_kv_heads * self.groups * self.zero_dtype_bytes
            k_norms = 0
        elif self.k_format == "K4":
            k_data = self.num_kv_heads * packed_length(self.head_dim, 4)
            k_scales = k_zeros = 0
            k_norms = self.num_kv_heads * self.norm_dtype_bytes
        else:
            raise ValueError(f"paged integration does not support {self.k_format}")

        v_bits = int(self.v_format[1:])
        v_data = self.num_kv_heads * packed_length(self.head_dim, v_bits)
        v_scales = self.num_kv_heads * self.groups * self.scale_dtype_bytes
        v_zeros = self.num_kv_heads * self.groups * self.zero_dtype_bytes
        return SlotComponentBytes(
            k_data=k_data,
            k_scales=k_scales,
            k_zeros=k_zeros,
            k_norms=k_norms,
            v_data=v_data,
            v_scales=v_scales,
            v_zeros=v_zeros,
        )

    @property
    def slot_bytes(self) -> int:
        return self.slot_components.total

    @property
    def page_bytes(self) -> int:
        return self.block_size * self.slot_bytes

    def shape(self, num_pages: int) -> tuple[int, int, int]:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        return (num_pages, self.block_size, self.slot_bytes)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["slot_components"] = self.slot_components.to_dict()
        data["slot_bytes"] = self.slot_bytes
        data["page_bytes"] = self.page_bytes
        if self.cache_dtype is not None:
            data["recipe_layout"] = recipe_layout(self.cache_dtype, self.head_dim).to_dict()
        return data

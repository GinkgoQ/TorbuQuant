"""Byte layouts for compressed KV blocks and pages."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from turboquant.kv.formats import get_k_format, get_v_format
from turboquant.packing.bits import packed_length


@dataclass(frozen=True)
class CacheGeometry:
    batch: int
    num_kv_heads: int
    tokens: int
    head_dim: int
    dtype_bytes: int = 2

    @property
    def vectors(self) -> int:
        return self.batch * self.num_kv_heads * self.tokens

    @property
    def dense_kv_bytes(self) -> int:
        return 2 * self.vectors * self.head_dim * self.dtype_bytes


@dataclass(frozen=True)
class PackedKVLayout:
    k_format: str
    v_format: str
    head_dim: int
    value_group_size: int = 32
    norm_bytes: int = 2
    scale_bytes: int = 2
    zero_bytes: int = 2
    include_qjl: bool = False
    page_padding_multiple: int | None = None
    bit_order: str = "little"

    def k_payload_bytes_per_vector(self) -> int:
        spec = get_k_format(self.k_format)
        if spec.bits == 16:
            return self.head_dim * 2
        if spec.bits == 8 and not spec.uses_codebook:
            return self.head_dim
        return packed_length(self.head_dim, spec.bits)

    def k_norm_bytes_per_vector(self) -> int:
        spec = get_k_format(self.k_format)
        if not spec.uses_norms:
            return 0
        return self.norm_bytes * (2 if self.include_qjl else 1)

    def k_metadata_bytes_per_vector(self) -> int:
        spec = get_k_format(self.k_format)
        if not (spec.uses_scales or spec.uses_zeros):
            return 0
        total = 0
        if spec.uses_scales:
            total += self.scale_bytes
        if spec.uses_zeros:
            total += self.zero_bytes
        return total

    def v_payload_bytes_per_vector(self) -> int:
        spec = get_v_format(self.v_format)
        return packed_length(self.padded_value_dim, spec.bits)

    @property
    def padded_value_dim(self) -> int:
        return ((self.head_dim + self.value_group_size - 1) // self.value_group_size) * self.value_group_size

    @property
    def value_groups(self) -> int:
        return self.padded_value_dim // self.value_group_size

    def v_metadata_bytes_per_vector(self) -> int:
        spec = get_v_format(self.v_format)
        total = 0
        if spec.uses_scales:
            total += self.value_groups * self.scale_bytes
        if spec.uses_zeros:
            total += self.value_groups * self.zero_bytes
        return total

    def bytes_per_vector_pair(self) -> int:
        raw = (
            self.k_payload_bytes_per_vector()
            + self.k_norm_bytes_per_vector()
            + self.k_metadata_bytes_per_vector()
            + self.v_payload_bytes_per_vector()
            + self.v_metadata_bytes_per_vector()
        )
        if self.page_padding_multiple is None:
            return raw
        return int(math.ceil(raw / self.page_padding_multiple) * self.page_padding_multiple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "k_payload_bytes_per_vector": self.k_payload_bytes_per_vector(),
                "k_norm_bytes_per_vector": self.k_norm_bytes_per_vector(),
                "k_metadata_bytes_per_vector": self.k_metadata_bytes_per_vector(),
                "v_payload_bytes_per_vector": self.v_payload_bytes_per_vector(),
                "v_metadata_bytes_per_vector": self.v_metadata_bytes_per_vector(),
                "bytes_per_vector_pair": self.bytes_per_vector_pair(),
                "padded_value_dim": self.padded_value_dim,
                "value_groups": self.value_groups,
            }
        )
        return data


def estimate_persistent_bytes(geometry: CacheGeometry, layout: PackedKVLayout) -> dict[str, int]:
    """Estimate persistent cache bytes for a geometry/layout pair."""
    vectors = geometry.vectors
    return {
        "dense_kv_bytes": geometry.dense_kv_bytes,
        "compressed_k_bytes": vectors * layout.k_payload_bytes_per_vector(),
        "compressed_v_bytes": vectors * layout.v_payload_bytes_per_vector(),
        "norms_bytes": vectors * layout.k_norm_bytes_per_vector(),
        "scales_bytes": vectors * (
            layout.value_groups * layout.scale_bytes
            + (layout.scale_bytes if get_k_format(layout.k_format).uses_scales else 0)
        ),
        "zeros_bytes": vectors * (
            layout.value_groups * layout.zero_bytes
            + (layout.zero_bytes if get_k_format(layout.k_format).uses_zeros else 0)
        ),
    }

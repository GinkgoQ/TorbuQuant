"""Named KV cache storage formats."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from turboquant.packing.bits import packed_length

KFormatName = Literal["K16", "K8", "K4", "K3", "K2"]
VFormatName = Literal["V8", "V4", "V3", "V2"]


@dataclass(frozen=True)
class KVFormatSpec:
    name: str
    kind: Literal["key", "value"]
    bits: int
    packed: bool
    uses_scales: bool = False
    uses_zeros: bool = False
    uses_norms: bool = False
    uses_codebook: bool = False
    requires_gate: bool = False

    def packed_values_bytes(self, dim: int) -> int:
        if self.bits == 16:
            return dim * 2
        return packed_length(dim, self.bits) if self.packed else dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


K_FORMATS: dict[str, KVFormatSpec] = {
    "K16": KVFormatSpec("K16", "key", 16, packed=False),
    "K8": KVFormatSpec("K8", "key", 8, packed=False, uses_scales=True, uses_zeros=True),
    "K4": KVFormatSpec("K4", "key", 4, packed=True, uses_norms=True, uses_codebook=True),
    "K3": KVFormatSpec("K3", "key", 3, packed=True, uses_norms=True, uses_codebook=True, requires_gate=True),
    "K2": KVFormatSpec("K2", "key", 2, packed=True, uses_norms=True, uses_codebook=True, requires_gate=True),
}

V_FORMATS: dict[str, KVFormatSpec] = {
    "V8": KVFormatSpec("V8", "value", 8, packed=False, uses_scales=True, uses_zeros=True),
    "V4": KVFormatSpec("V4", "value", 4, packed=True, uses_scales=True, uses_zeros=True),
    "V3": KVFormatSpec("V3", "value", 3, packed=True, uses_scales=True, uses_zeros=True, requires_gate=True),
    "V2": KVFormatSpec("V2", "value", 2, packed=True, uses_scales=True, uses_zeros=True),
}


def get_k_format(name: KFormatName | str) -> KVFormatSpec:
    try:
        return K_FORMATS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown key format: {name}") from exc


def get_v_format(name: VFormatName | str) -> KVFormatSpec:
    try:
        return V_FORMATS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown value format: {name}") from exc


def validate_v_format(name: VFormatName | str, *, allow_v3: bool = False) -> KVFormatSpec:
    spec = get_v_format(name)
    if spec.name == "V3" and not allow_v3:
        raise ValueError("V3 requires an explicit format gate")
    return spec


def validate_k_format(
    name: KFormatName | str,
    *,
    allow_low_bit_keys: bool = False,
) -> KVFormatSpec:
    spec = get_k_format(name)
    if spec.requires_gate and not allow_low_bit_keys:
        raise ValueError(f"{spec.name} requires an explicit format gate")
    return spec

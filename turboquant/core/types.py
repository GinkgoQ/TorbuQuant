"""Shared data structures for quantized tensors and codec metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, NamedTuple

import torch


@dataclass(frozen=True)
class TransformSpec:
    """Serializable rotation or sketch transform metadata."""

    kind: Literal["qr_rotation", "rht", "qjl"]
    dim: int
    seed: int
    dtype: str = "float32"
    device_type: str = "cpu"
    pad_dim: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformSpec":
        return cls(**data)


@dataclass(frozen=True)
class CodebookSpec:
    """Serializable Lloyd-Max codebook metadata."""

    dim: int
    bits: int
    distribution: Literal["beta_sphere", "gaussian_approx"] = "beta_sphere"
    cache_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodebookSpec":
        return cls(**data)


class QuantizedTensor(NamedTuple):
    """Common tensor payload plus byte-level format metadata."""

    data: torch.Tensor
    bits: int
    dim: int
    format: str


class QuantizedKeys(NamedTuple):
    """Key payload with transform and codebook specs."""

    payload: torch.Tensor
    norms: torch.Tensor
    bits: int
    dim: int
    transform: TransformSpec
    codebook: CodebookSpec


class QuantizedValues(NamedTuple):
    """Value payload with scale/zero metadata."""

    payload: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int
    dim: int
    group_size: int


class MSEData(NamedTuple):
    """
    Output of TurboQuantMSE.quantize().

    indices : (BH, N, Pk) uint8
    norms   : (BH, N) fp16
    bits    : int
    dim     : int
    """
    indices: torch.Tensor
    norms: torch.Tensor
    bits: int
    dim: int


class ProdData(NamedTuple):
    """
    Output of TurboQuantProd.quantize().

    mse_indices    : (BH, N, Pk) uint8
    qjl_signs      : (BH, N, Ps) uint8
    residual_norms : (BH, N) fp32
    norms          : (BH, N) fp16
    mse_bits       : int
    dim            : int
    """
    mse_indices: torch.Tensor
    qjl_signs: torch.Tensor
    residual_norms: torch.Tensor
    norms: torch.Tensor
    mse_bits: int
    dim: int


class ChannelSplitData(NamedTuple):
    """
    Output of channel-split MSE quantization.

    high        : MSEData for selected high-bit channels, or None
    low         : MSEData for remaining channels, or None
    high_index  : selected high-bit channel indices
    low_index   : remaining channel indices
    dim         : original vector dimension
    target_bits : average bit target before norm metadata
    """
    high: MSEData | None
    low: MSEData | None
    high_index: torch.Tensor
    low_index: torch.Tensor
    dim: int
    target_bits: float


class ValueData(NamedTuple):
    """
    Output of quantize_values().

    data   : (BH, N, Pv) uint8
    scales : (BH, N, G) fp16
    zeros  : (BH, N, G) fp16
    bits   : int
    dim    : int
    group_size : int
    """
    data: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int
    dim: int
    group_size: int

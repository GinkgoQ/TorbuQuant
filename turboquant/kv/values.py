"""Grouped value quantization and packed storage."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from turboquant.core.types import ValueData
from turboquant.kv.formats import validate_v_format
from turboquant.packing.bits import pack_unsigned, unpack_unsigned

ScaleLayout = Literal["group", "token"]


def _group_count(dim: int, group_size: int) -> int:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    return (dim + group_size - 1) // group_size


def padded_dim(dim: int, group_size: int) -> int:
    return _group_count(dim, group_size) * group_size


def quantize_values(
    values: torch.Tensor,
    *,
    bits: int,
    group_size: int = 32,
    scale_layout: ScaleLayout = "group",
    allow_v3: bool = False,
    scale_dtype: torch.dtype = torch.float16,
) -> ValueData:
    """Quantize values with per-group or per-token affine metadata."""
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"value bits must be one of 2, 3, 4, 8, got {bits}")
    validate_v_format(f"V{bits}", allow_v3=allow_v3)

    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    dim = int(values.shape[-1])
    effective_group = dim if scale_layout == "token" else group_size
    if scale_layout not in ("group", "token"):
        raise ValueError(f"unknown scale layout: {scale_layout}")

    dim_pad = padded_dim(dim, effective_group)
    if dim_pad != dim:
        values_pad = F.pad(values, (0, dim_pad - dim))
    else:
        values_pad = values

    groups = dim_pad // effective_group
    grouped = values_pad.float().reshape(*values.shape[:-1], groups, effective_group)
    valid = torch.arange(dim_pad, device=values.device) < dim
    valid = valid.reshape(groups, effective_group)
    view_shape = (1,) * (grouped.ndim - 2) + valid.shape
    valid = valid.reshape(view_shape)
    grouped_for_min = grouped.masked_fill(~valid, float("inf"))
    grouped_for_max = grouped.masked_fill(~valid, float("-inf"))
    v_min = grouped_for_min.min(dim=-1, keepdim=True).values
    v_max = grouped_for_max.max(dim=-1, keepdim=True).values
    levels = (1 << bits) - 1
    scales = ((v_max - v_min) / levels).clamp(min=1e-12)
    zeros = v_min
    q = ((grouped - zeros) / scales).round().clamp(0, levels).to(torch.int64)
    q = q.masked_fill(~valid, 0)
    q_flat = q.reshape(*values.shape[:-1], dim_pad)
    data = pack_unsigned(q_flat, bits)

    return ValueData(
        data=data,
        scales=scales.squeeze(-1).to(scale_dtype),
        zeros=zeros.squeeze(-1).to(scale_dtype),
        bits=bits,
        dim=dim,
        group_size=effective_group,
    )


def dequantize_values(q: ValueData, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct values from packed affine metadata."""
    dim_pad = int(q.scales.shape[-1]) * q.group_size
    indices = unpack_unsigned(q.data, q.bits, dim_pad).float()
    grouped = indices.reshape(*indices.shape[:-1], q.scales.shape[-1], q.group_size)
    values = grouped * q.scales.float().unsqueeze(-1) + q.zeros.float().unsqueeze(-1)
    return values.reshape(*indices.shape[:-1], dim_pad)[..., : q.dim].to(dtype)


def value_data_nbytes(q: ValueData) -> dict[str, int]:
    """Return byte counts for a ValueData payload."""
    return {
        "compressed_v_bytes": q.data.numel() * q.data.element_size(),
        "scales_bytes": q.scales.numel() * q.scales.element_size(),
        "zeros_bytes": q.zeros.numel() * q.zeros.element_size(),
    }


def value_formula_nbytes(
    shape: tuple[int, ...],
    *,
    bits: int,
    group_size: int,
    scale_dtype_bytes: int = 2,
    zero_dtype_bytes: int = 2,
    scale_layout: ScaleLayout = "group",
) -> dict[str, int]:
    """Compute expected value bytes for a shape without allocating tensors."""
    if not shape:
        raise ValueError("shape must have at least one dimension")
    dim = int(shape[-1])
    effective_group = dim if scale_layout == "token" else group_size
    dim_pad = padded_dim(dim, effective_group)
    outer = math.prod(shape[:-1]) if len(shape) > 1 else 1
    groups = dim_pad // effective_group
    payload = outer * ((dim_pad * bits + 7) // 8)
    meta = outer * groups
    return {
        "compressed_v_bytes": payload,
        "scales_bytes": meta * scale_dtype_bytes,
        "zeros_bytes": meta * zero_dtype_bytes,
        "padded_dim": dim_pad,
        "groups": groups,
    }

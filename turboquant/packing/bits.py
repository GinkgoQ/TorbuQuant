"""Bit packing utilities for K/V cache tensors."""

from __future__ import annotations

import math

import torch


def packed_length(num_values: int, bits: int) -> int:
    """Return the number of bytes needed for packed unsigned values."""
    _check_bits(bits)
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")
    return (num_values * bits + 7) // 8


def _check_bits(bits: int) -> None:
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")


def pack_unsigned(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned integer values along the last dimension.

    The last dimension is encoded as a bitstream with little-endian bit order
    inside each byte. Values must be in the range [0, 2**bits).
    """
    _check_bits(bits)
    if values.numel() == 0:
        return torch.empty(*values.shape[:-1], 0, dtype=torch.uint8, device=values.device)

    values_i = values.to(torch.int64)
    if torch.any(values_i < 0) or torch.any(values_i >= (1 << bits)):
        raise ValueError(f"values must be in [0, {1 << bits}) for bits={bits}")

    *batch, width = values_i.shape
    out_len = packed_length(width, bits)
    rows = values_i.reshape(-1, width)
    out = torch.zeros(rows.shape[0], out_len, dtype=torch.int64, device=values.device)

    positions = torch.arange(width, device=values.device, dtype=torch.int64)
    for bit in range(bits):
        bit_values = (rows >> bit) & 1
        absolute = positions * bits + bit
        byte_idx = absolute // 8
        bit_idx = absolute % 8
        out.scatter_add_(1, byte_idx.expand(rows.shape[0], -1), bit_values << bit_idx)

    return out.to(torch.uint8).reshape(*batch, out_len)


def unpack_unsigned(packed: torch.Tensor, bits: int, num_values: int) -> torch.Tensor:
    """Unpack unsigned integer values from the last packed dimension."""
    _check_bits(bits)
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")
    expected = packed_length(num_values, bits)
    if packed.shape[-1] != expected:
        raise ValueError(f"packed last dimension must be {expected}, got {packed.shape[-1]}")

    if num_values == 0:
        return torch.empty(*packed.shape[:-1], 0, dtype=torch.int64, device=packed.device)

    *batch, _ = packed.shape
    rows = packed.reshape(-1, expected).to(torch.int64)
    positions = torch.arange(num_values, device=packed.device, dtype=torch.int64)
    out = torch.zeros(rows.shape[0], num_values, dtype=torch.int64, device=packed.device)

    for bit in range(bits):
        absolute = positions * bits + bit
        byte_idx = absolute // 8
        bit_idx = absolute % 8
        bit_values = (rows[:, byte_idx] >> bit_idx) & 1
        out |= bit_values << bit

    return out.reshape(*batch, num_values)


def pack_k4(values: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit key indices, two values per byte."""
    return pack_unsigned(values, 4)


def unpack_k4(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack 4-bit key indices."""
    return unpack_unsigned(packed, 4, num_values)


def pack_k3(values: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit key indices as a bitstream."""
    return pack_unsigned(values, 3)


def unpack_k3(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack 3-bit key indices."""
    return unpack_unsigned(packed, 3, num_values)


def pack_k2(values: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit key indices, four values per byte."""
    return pack_unsigned(values, 2)


def unpack_k2(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack 2-bit key indices."""
    return unpack_unsigned(packed, 2, num_values)


def pack_v4(values: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit value indices, two values per byte."""
    return pack_unsigned(values, 4)


def unpack_v4(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack 4-bit value indices."""
    return unpack_unsigned(packed, 4, num_values)


def pack_v2(values: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit value indices, four values per byte."""
    return pack_unsigned(values, 2)


def unpack_v2(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack 2-bit value indices."""
    return unpack_unsigned(packed, 2, num_values)


def pack_v3(values: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit value indices as a bitstream."""
    return pack_unsigned(values, 3)


def unpack_v3(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack 3-bit value indices."""
    return unpack_unsigned(packed, 3, num_values)


def pack_sign_bits(signs: torch.Tensor) -> torch.Tensor:
    """Pack sign values into one bit each.

    Accepted inputs are bool, {0, 1}, or {-1, 1}. The unpacked form is bool.
    """
    if signs.dtype == torch.bool:
        bits = signs.to(torch.int64)
    else:
        signs_i = signs.to(torch.int64)
        if torch.any((signs_i != 0) & (signs_i != 1) & (signs_i != -1)):
            raise ValueError("signs must contain bool, {0, 1}, or {-1, 1} values")
        bits = (signs_i > 0).to(torch.int64)
    return pack_unsigned(bits, 1)


def unpack_sign_bits(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack sign bits to bool values."""
    return unpack_unsigned(packed, 1, num_values).bool()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return tensor storage bytes for dense tensors."""
    return tensor.numel() * tensor.element_size()


def packed_nbytes(shape: tuple[int, ...], bits: int) -> int:
    """Return storage bytes for packing the last dimension of a shape."""
    if not shape:
        raise ValueError("shape must have at least one dimension")
    outer = math.prod(shape[:-1]) if len(shape) > 1 else 1
    return outer * packed_length(shape[-1], bits)

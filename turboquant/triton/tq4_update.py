"""TQ packed page update operations."""

from __future__ import annotations

import torch

from turboquant.core.rotation import RotationState, rotate_forward
from turboquant.packing.bits import pack_unsigned, packed_length


def pack_tq_rows(
    raw: torch.Tensor,
    *,
    rotation: RotationState | torch.Tensor,
    centroids: torch.Tensor,
    bit_width: int,
) -> torch.Tensor:
    if raw.ndim != 3:
        raise ValueError("raw must have shape [tokens, kv_heads, head_dim]")
    if bit_width < 1 or bit_width > 8:
        raise ValueError("bit_width must be in [1, 8]")
    tokens, heads, dim = raw.shape
    norms = raw.float().norm(dim=-1).clamp(min=1e-12)
    unit = raw.float() / norms.unsqueeze(-1)
    rotated = _rotate_forward(unit, rotation)
    distances = (rotated.unsqueeze(-1) - centroids.to(raw.device, torch.float32)).abs()
    indices = torch.argmin(distances, dim=-1).to(torch.int64)
    packed = pack_unsigned(indices, bit_width)
    norm_bytes = norms.float().contiguous().view(torch.uint8).reshape(tokens, heads, 4)
    return torch.cat([packed, norm_bytes], dim=-1).contiguous()


def write_tq4_kv(
    raw: torch.Tensor,
    page_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    rotation: RotationState | torch.Tensor,
    centroids: torch.Tensor,
    bit_width: int = 4,
) -> torch.Tensor:
    rows = pack_tq_rows(
        raw,
        rotation=rotation,
        centroids=centroids,
        bit_width=bit_width,
    )
    if page_tensor.ndim != 4:
        raise ValueError("page_tensor must have shape [blocks, block_size, kv_heads, row_bytes]")
    if page_tensor.dtype != torch.uint8:
        raise ValueError("page_tensor must use uint8 storage")
    if rows.shape[1] != page_tensor.shape[2]:
        raise ValueError("KV head count differs between rows and page tensor")
    if rows.shape[-1] > page_tensor.shape[-1]:
        raise ValueError("page row is smaller than packed row")
    slots = slot_mapping.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    if len(slots) != raw.shape[0]:
        raise ValueError("slot_mapping length must match token count")
    block_size = int(page_tensor.shape[1])
    for token_idx, slot in enumerate(slots):
        if int(slot) < 0:
            continue
        block = int(slot) // block_size
        offset = int(slot) % block_size
        if block >= page_tensor.shape[0]:
            raise IndexError(f"slot {slot} exceeds page tensor")
        page_tensor[block, offset, :, : rows.shape[-1]].copy_(rows[token_idx].to(page_tensor.device))
    return rows


def tq_row_bytes(head_dim: int, bit_width: int) -> int:
    return packed_length(head_dim, bit_width) + 4


def _rotate_forward(x: torch.Tensor, rotation: RotationState | torch.Tensor) -> torch.Tensor:
    if isinstance(rotation, RotationState):
        return rotate_forward(x, rotation)
    return torch.matmul(x.float(), rotation.to(device=x.device, dtype=torch.float32).T)

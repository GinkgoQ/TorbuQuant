"""TQ packed page decode operations."""

from __future__ import annotations

import math

import torch

from turboquant.core.rotation import RotationState, rotate_backward
from turboquant.packing.bits import packed_length, unpack_unsigned


def unpack_tq_rows(
    rows: torch.Tensor,
    *,
    rotation: RotationState | torch.Tensor,
    centroids: torch.Tensor,
    bit_width: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if rows.ndim < 2:
        raise ValueError("rows must have a packed byte dimension")
    index_bytes = packed_length(head_dim, bit_width)
    if rows.shape[-1] < index_bytes + 4:
        raise ValueError("row is smaller than expected packed payload")
    indices = unpack_unsigned(rows[..., :index_bytes], bit_width, head_dim)
    norms = rows[..., index_bytes : index_bytes + 4].contiguous().view(torch.float32).squeeze(-1)
    values = centroids.to(device=rows.device, dtype=torch.float32)[indices]
    restored = _rotate_backward(values, rotation)
    return (restored * norms.unsqueeze(-1)).to(dtype)


def decode_tq4_paged(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    rotation: RotationState | torch.Tensor,
    centroids: torch.Tensor,
    num_kv_heads: int,
    bit_width_k: int = 4,
    bit_width_v: int = 4,
    scale: float | None = None,
    sliding_window: int | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if query.ndim != 3:
        raise ValueError("query must have shape [seqs, q_heads, head_dim]")
    if key_pages.shape[:3] != value_pages.shape[:3]:
        raise ValueError("key/value page geometry differs")
    seqs, q_heads, head_dim = query.shape
    if q_heads % num_kv_heads != 0:
        raise ValueError("query heads must divide KV heads")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    outputs = torch.empty_like(query, dtype=output_dtype or query.dtype)
    group = q_heads // num_kv_heads
    for seq_idx in range(seqs):
        seq_len = int(seq_lens[seq_idx].item())
        start = 0 if sliding_window is None else max(0, seq_len - int(sliding_window))
        key_rows = _read_sequence_rows(key_pages, block_table[seq_idx], start, seq_len)
        value_rows = _read_sequence_rows(value_pages, block_table[seq_idx], start, seq_len)
        keys = unpack_tq_rows(
            key_rows,
            rotation=rotation,
            centroids=centroids,
            bit_width=bit_width_k,
            head_dim=head_dim,
            dtype=torch.float32,
        )
        values = unpack_tq_rows(
            value_rows,
            rotation=rotation,
            centroids=centroids,
            bit_width=bit_width_v,
            head_dim=head_dim,
            dtype=torch.float32,
        )
        for q_head in range(q_heads):
            kv_head = q_head // group
            scores = torch.matmul(keys[:, kv_head], query[seq_idx, q_head].float()) * scale
            weights = torch.softmax(scores, dim=-1)
            outputs[seq_idx, q_head] = torch.matmul(weights, values[:, kv_head]).to(outputs.dtype)
    return outputs


def _read_sequence_rows(
    pages: torch.Tensor,
    block_table_row: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    block_size = int(pages.shape[1])
    rows = []
    for token in range(start, stop):
        logical = token // block_size
        offset = token % block_size
        physical = int(block_table_row[logical].item())
        rows.append(pages[physical, offset])
    if not rows:
        return torch.empty(0, pages.shape[2], pages.shape[3], dtype=pages.dtype, device=pages.device)
    return torch.stack(rows, dim=0)


def _rotate_backward(x: torch.Tensor, rotation: RotationState | torch.Tensor) -> torch.Tensor:
    if isinstance(rotation, RotationState):
        return rotate_backward(x, rotation)
    return torch.matmul(x.float(), rotation.to(device=x.device, dtype=torch.float32))

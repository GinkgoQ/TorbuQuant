"""Hybrid attention over compressed history and exact recent buffer.

Provides efficient attention computation combining:
- Compressed historical KV (TurboQuant quantized)
- Exact recent buffer (fp16/bf16)

Design: compressed path is only invoked when history is large enough
to justify quantization overhead (>= MIN_HISTORY_FOR_TQ tokens).
"""

from __future__ import annotations

import math
from typing import Optional, NamedTuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torbuquant.core.types import ProdData, ValueData
    from torbuquant.core.polar import TorbuquantProd

MIN_HISTORY_FOR_TQ = 16


class FlatCache(NamedTuple):
    """Flattened view of compressed KV for fast read access."""
    prod_q: "ProdData"
    value_q: "ValueData"
    num_tokens: int


class HybridAttentionResult(NamedTuple):
    """Result of hybrid attention computation."""
    output: torch.Tensor
    scores: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None


def compute_hybrid_attention(
    query: torch.Tensor,
    compressed_keys: Optional["ProdData"],
    compressed_values: Optional["ValueData"],
    recent_k: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    quantizer: Optional["TorbuquantProd"],
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: Optional[float] = None,
    return_scores: bool = False,
) -> HybridAttentionResult:
    """Compute attention output combining compressed history and exact recent buffer.

    Args:
        query: (num_tokens, num_query_heads, head_dim) — typically num_tokens=1 for decode
        compressed_keys: ProdData from TurboQuantProd quantization, or None
        compressed_values: ValueData from value quantization, or None
        recent_k: (recent_len, num_kv_heads, head_dim) or None
        recent_v: (recent_len, num_kv_heads, head_dim) or None
        quantizer: TurboQuantProd instance for dequantization
        num_query_heads: total query heads (for GQA expansion)
        num_kv_heads: number of KV heads
        head_dim: dimension per head
        scale: attention scale factor (default: 1/sqrt(head_dim))
        return_scores: whether to return attention scores

    Returns:
        HybridAttentionResult with output: (num_tokens, num_query_heads, head_dim)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    has_history = compressed_keys is not None and compressed_keys.mse_indices.shape[-2] >= MIN_HISTORY_FOR_TQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return HybridAttentionResult(
            output=torch.zeros(
                query.shape[0], num_query_heads, head_dim,
                device=query.device, dtype=query.dtype,
            )
        )

    gqa_ratio = num_query_heads // num_kv_heads

    if has_history and not has_recent:
        return _attend_compressed_only(
            query, compressed_keys, compressed_values, quantizer,
            gqa_ratio, num_kv_heads, head_dim, scale, return_scores
        )

    if not has_history and has_recent:
        return _attend_exact_only(
            query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale, return_scores
        )

    # Both segments present — merge via concatenation
    return _attend_hybrid(
        query, compressed_keys, compressed_values, quantizer,
        recent_k, recent_v, gqa_ratio, num_kv_heads, head_dim, scale, return_scores
    )


def _attend_compressed_only(
    query: torch.Tensor,
    compressed_keys: "ProdData",
    compressed_values: "ValueData",
    quantizer: "TorbuquantProd",
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    return_scores: bool,
) -> HybridAttentionResult:
    """Attention over compressed history only."""
    from torbuquant.kv.values import dequantize_values as dequant_v

    k_dequant = quantizer.dequantize(compressed_keys)
    v_dequant = dequant_v(compressed_values)

    return _matmul_attend(
        query, k_dequant, v_dequant,
        gqa_ratio, num_kv_heads, scale, return_scores
    )


def _attend_exact_only(
    query: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
    return_scores: bool,
) -> HybridAttentionResult:
    """Attention over exact recent buffer only."""
    # recent_k: (T, H_kv, D) -> (H_kv, T, D)
    k = recent_k.transpose(0, 1)
    v = recent_v.transpose(0, 1)
    return _matmul_attend(
        query, k, v, gqa_ratio, num_kv_heads, scale, return_scores
    )


def _attend_hybrid(
    query: torch.Tensor,
    compressed_keys: "ProdData",
    compressed_values: "ValueData",
    quantizer: "TorbuquantProd",
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    return_scores: bool,
) -> HybridAttentionResult:
    """Merge compressed history + exact recent via concatenated attention."""
    from torbuquant.kv.values import dequantize_values as dequant_v

    k_hist = quantizer.dequantize(compressed_keys)  # (H_kv, N_hist, D)
    v_hist = dequant_v(compressed_values)

    k_recent = recent_k.transpose(0, 1)   # (H_kv, N_recent, D)
    v_recent = recent_v.transpose(0, 1)

    k_all = torch.cat([k_hist.float(), k_recent.float()], dim=1)
    v_all = torch.cat([v_hist.float(), v_recent.float()], dim=1)

    return _matmul_attend(
        query, k_all, v_all, gqa_ratio, num_kv_heads, scale, return_scores
    )


def _matmul_attend(
    query: torch.Tensor,
    kv_keys: torch.Tensor,
    kv_values: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
    return_scores: bool,
) -> HybridAttentionResult:
    """Standard matmul attention with GQA support.

    query: (T, Q_heads, D)
    kv_keys: (H_kv, N, D)
    kv_values: (H_kv, N, D)

    Returns: output (T, Q_heads, D)
    """
    T, Q, D = query.shape
    H_kv = num_kv_heads
    if Q != H_kv * gqa_ratio:
        raise ValueError(
            f"Incompatible GQA shapes: Q={Q}, H_kv={H_kv}, gqa_ratio={gqa_ratio}"
        )

    # Avoid repeat_interleave(Q/H) on KV tensors to keep memory bounded at long context.
    # q: (T, Q, D) -> (H_kv, G, T, D)
    q = query.float().view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)
    k = kv_keys.float().unsqueeze(1)   # (H_kv, 1, N, D) broadcast over G
    v = kv_values.float().unsqueeze(1) # (H_kv, 1, N, D) broadcast over G

    # scores: (H_kv, G, T, N)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)

    # Back to (T, Q, D)
    output = out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)

    if return_scores:
        return HybridAttentionResult(
            output=output,
            scores=scores.permute(2, 0, 1, 3).reshape(T, Q, -1),
            weights=weights.permute(2, 0, 1, 3).reshape(T, Q, -1),
        )
    return HybridAttentionResult(output=output)


def compute_hybrid_attention_store(
    query: torch.Tensor,
    flat_cache,  # FlatCache from kv/store.py
    key_compressor,  # TurboQuantCompressorV2
    value_compressor,  # TurboQuantCompressorMSE
    recent_k: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    num_query_heads: int,
    num_kv_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute hybrid attention using store-based compressed cache.

    This variant works with the CompressedKVStore from kv/store.py.

    Args:
        query: (num_tokens, num_query_heads, head_dim) query vectors
        flat_cache: FlatCache from CompressedKVStore.get_flat_cache()
        key_compressor: TurboQuantCompressorV2 instance
        value_compressor: TurboQuantCompressorMSE instance
        recent_k: (recent_len, num_kv_heads, head_dim) or None
        recent_v: (recent_len, num_kv_heads, head_dim) or None
        num_query_heads: total query heads
        num_kv_heads: number of KV heads
        scale: attention scale (default: 1/sqrt(head_dim))

    Returns:
        output: (num_tokens, num_query_heads, head_dim)
    """
    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    has_history = flat_cache is not None and flat_cache.num_tokens >= MIN_HISTORY_FOR_TQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros(
            query.shape[0], num_query_heads, head_dim,
            device=query.device, dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // num_kv_heads

    # Decompress historical KV
    if has_history:
        # Decompress using the compressors
        k_hist = key_compressor.decompress(flat_cache.key_compressed).float()  # (H_kv, N, D)
        v_hist = value_compressor.decompress(flat_cache.value_compressed).float()
        # Reshape from (1, H_kv, N, D) to (H_kv, N, D) if needed
        if k_hist.dim() == 4 and k_hist.shape[0] == 1:
            k_hist = k_hist.squeeze(0)
            v_hist = v_hist.squeeze(0)
    else:
        k_hist = None
        v_hist = None

    # Process recent buffer
    if has_recent:
        k_recent = recent_k.transpose(0, 1).float()  # (H_kv, N_recent, D)
        v_recent = recent_v.transpose(0, 1).float()
    else:
        k_recent = None
        v_recent = None

    # Concatenate if both present
    if k_hist is not None and k_recent is not None:
        k_all = torch.cat([k_hist, k_recent], dim=1)
        v_all = torch.cat([v_hist, v_recent], dim=1)
    elif k_hist is not None:
        k_all = k_hist
        v_all = v_hist
    else:
        k_all = k_recent
        v_all = v_recent

    # Compute attention
    T, Q, D = query.shape
    H_kv = num_kv_heads

    q = query.float().view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)
    k = k_all.unsqueeze(1)   # (H_kv, 1, N, D)
    v = v_all.unsqueeze(1)   # (H_kv, 1, N, D)

    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)

    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)


def online_softmax_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    scale: float,
    block_size: int = 64,
) -> torch.Tensor:
    """Online softmax attention over KV blocks.

    Flash-attention style memory-efficient implementation.

    Args:
        query: (Q, D) single query vector
        keys: (N, D) key vectors
        values: (N, D) value vectors
        scale: attention scale
        block_size: block size for online processing

    Returns:
        output: (Q, D) attention output
    """
    Q, D = query.shape
    N = keys.shape[0]

    # Initialize online softmax state
    m_i = torch.full((Q,), float("-inf"), device=query.device, dtype=torch.float32)
    l_i = torch.zeros((Q,), device=query.device, dtype=torch.float32)
    acc = torch.zeros((Q, D), device=query.device, dtype=torch.float32)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        k_block = keys[start:end].float()  # (B, D)
        v_block = values[start:end].float()  # (B, D)

        # Compute scores for this block
        scores = torch.matmul(query.float(), k_block.T) * scale  # (Q, B)

        # Online softmax update
        m_new = torch.maximum(m_i, scores.max(dim=-1).values)
        alpha = torch.exp(m_i - m_new)
        p = torch.exp(scores - m_new.unsqueeze(-1))

        # Update accumulator
        l_i = l_i * alpha + p.sum(dim=-1)
        acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, v_block)
        m_i = m_new

    # Final normalization
    return (acc / l_i.unsqueeze(-1)).to(query.dtype)

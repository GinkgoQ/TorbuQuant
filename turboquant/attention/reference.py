"""Reference attention paths for compressed KV caches."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F

from turboquant.core.rotation import rotate_forward
from turboquant.core.types import MSEData
from turboquant.kv.cache import CompressedKVBlock, CompressedKVCache
from turboquant.kv.keys import DenseKeyData, QuantizedKeyData, TurboKeyData, build_k4_quantizer, dequantize_k8
from turboquant.kv.values import dequantize_values

AttentionLabel = Literal[
    "dense_baseline",
    "diagnostic_dequant",
    "direct_qk",
    "weighted_packed_v",
    "sparse_weighted_packed_v",
    "fused_decode",
    "fused_paged_decode",
    "fallback_dense",
]


@dataclass(frozen=True)
class AttentionReport:
    label: AttentionLabel
    score_label: AttentionLabel | None = None
    value_label: AttentionLabel | None = None
    dense_materialized: bool = False
    q_len: int = 0
    kv_len: int = 0
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    segments: list[dict[str, Any]] = field(default_factory=list)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionResult:
    output: torch.Tensor
    scores: torch.Tensor | None
    weights: torch.Tensor | None
    report: AttentionReport


@dataclass(frozen=True)
class SparseVReport:
    threshold: float
    total_weights: int
    skipped_weights: int
    skipped_weight_sum: float

    @property
    def skip_ratio(self) -> float:
        if self.total_weights == 0:
            return 0.0
        return self.skipped_weights / self.total_weights

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["skip_ratio"] = self.skip_ratio
        return data


def default_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def gqa_kv_heads(num_q_heads: int, num_kv_heads: int, device: torch.device | None = None) -> torch.Tensor:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    ratio = num_q_heads // num_kv_heads
    return torch.arange(num_q_heads, device=device, dtype=torch.long) // ratio


def _validate_query(query: torch.Tensor) -> tuple[int, int, int]:
    if query.ndim != 3:
        raise ValueError("query must have shape (tokens, query_heads, head_dim)")
    return int(query.shape[0]), int(query.shape[1]), int(query.shape[2])


def _validate_kv(keys: torch.Tensor, values: torch.Tensor) -> tuple[int, int, int]:
    if keys.shape != values.shape:
        raise ValueError(f"key/value shape mismatch: {keys.shape} vs {values.shape}")
    if keys.ndim != 3:
        raise ValueError("keys and values must have shape (tokens, kv_heads, head_dim)")
    return int(keys.shape[0]), int(keys.shape[1]), int(keys.shape[2])


def apply_causal_mask(
    scores: torch.Tensor,
    *,
    causal: bool,
    q_start_pos: int | None = None,
) -> torch.Tensor:
    if not causal:
        return scores
    q_len, _, kv_len = scores.shape
    if q_start_pos is None:
        q_start_pos = kv_len - q_len
    q_pos = torch.arange(q_len, device=scores.device).reshape(q_len, 1) + q_start_pos
    kv_pos = torch.arange(kv_len, device=scores.device).reshape(1, kv_len)
    mask = kv_pos <= q_pos
    return scores.masked_fill(~mask[:, None, :], float("-inf"))


def dense_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    scale: float | None = None,
    causal: bool = False,
    q_start_pos: int | None = None,
) -> AttentionResult:
    q_len, num_q_heads, head_dim = _validate_query(query)
    kv_len, num_kv_heads, kv_dim = _validate_kv(keys, values)
    if head_dim != kv_dim:
        raise ValueError(f"query/key dim mismatch: {head_dim} vs {kv_dim}")
    scale = default_scale(head_dim) if scale is None else scale
    scores = dense_scores(
        query,
        keys,
        scale=scale,
        causal=causal,
        q_start_pos=q_start_pos,
    )
    weights = torch.softmax(scores.float(), dim=-1)
    output = dense_value_accumulation(weights, values, num_q_heads=num_q_heads).to(query.dtype)
    report = AttentionReport(
        label="dense_baseline",
        score_label="dense_baseline",
        value_label="dense_baseline",
        dense_materialized=True,
        q_len=q_len,
        kv_len=kv_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        segments=[{"kind": "dense", "tokens": kv_len}],
    )
    return AttentionResult(output=output, scores=scores, weights=weights, report=report)


def dense_sdpa_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> AttentionResult:
    q_len, num_q_heads, head_dim = _validate_query(query)
    kv_len, num_kv_heads, kv_dim = _validate_kv(keys, values)
    if head_dim != kv_dim:
        raise ValueError(f"query/key dim mismatch: {head_dim} vs {kv_dim}")
    keys = keys.to(query.device)
    values = values.to(query.device)
    kv_map = gqa_kv_heads(num_q_heads, num_kv_heads, query.device)
    k_q = keys[:, kv_map, :].permute(1, 0, 2).unsqueeze(0).contiguous()
    v_q = values[:, kv_map, :].permute(1, 0, 2).unsqueeze(0).contiguous()
    q = query.permute(1, 0, 2).unsqueeze(0).contiguous()
    output = F.scaled_dot_product_attention(
        q,
        k_q,
        v_q,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )
    output = output.squeeze(0).permute(1, 0, 2).contiguous()
    scores = dense_scores(
        query,
        keys,
        scale=default_scale(head_dim) if scale is None else scale,
        causal=causal,
    )
    weights = torch.softmax(scores.float(), dim=-1)
    report = AttentionReport(
        label="dense_baseline",
        score_label="dense_baseline",
        value_label="dense_baseline",
        dense_materialized=True,
        q_len=q_len,
        kv_len=kv_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        segments=[{"kind": "sdpa", "tokens": kv_len}],
    )
    return AttentionResult(output=output, scores=scores, weights=weights, report=report)


def dense_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    scale: float | None = None,
    causal: bool = False,
    q_start_pos: int | None = None,
) -> torch.Tensor:
    q_len, num_q_heads, head_dim = _validate_query(query)
    kv_len, num_kv_heads, kv_dim = _validate_kv(keys, keys)
    if head_dim != kv_dim:
        raise ValueError(f"query/key dim mismatch: {head_dim} vs {kv_dim}")
    scale = default_scale(head_dim) if scale is None else scale
    keys = keys.to(query.device)
    kv_map = gqa_kv_heads(num_q_heads, num_kv_heads, query.device)
    k_by_q = keys.float()[:, kv_map, :]
    scores = torch.einsum("tqd,sqd->tqs", query.float(), k_by_q) * scale
    return apply_causal_mask(scores, causal=causal, q_start_pos=q_start_pos)


def dense_value_accumulation(weights: torch.Tensor, values: torch.Tensor, *, num_q_heads: int) -> torch.Tensor:
    kv_len, num_kv_heads, _ = _validate_kv(values, values)
    if weights.shape[1] != num_q_heads or weights.shape[2] != kv_len:
        raise ValueError("weight shape does not match values")
    values = values.to(weights.device)
    kv_map = gqa_kv_heads(num_q_heads, num_kv_heads, weights.device)
    v_by_q = values.float()[:, kv_map, :]
    return torch.einsum("tqs,sqd->tqd", weights.float(), v_by_q)


def online_softmax(scores: torch.Tensor, *, block_size: int = 64) -> torch.Tensor:
    if scores.ndim != 3:
        raise ValueError("scores must have shape (tokens, query_heads, kv_tokens)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    q_len, num_q_heads, kv_len = scores.shape
    flat = scores.float().reshape(q_len * num_q_heads, kv_len)
    running_max = torch.full((flat.shape[0],), -float("inf"), device=scores.device)
    running_sum = torch.zeros_like(running_max)
    chunks: list[torch.Tensor] = []
    for start in range(0, kv_len, block_size):
        chunk = flat[:, start : start + block_size]
        chunk_max = chunk.max(dim=-1).values
        next_max = torch.maximum(running_max, chunk_max)
        alpha = torch.exp(running_max - next_max)
        exp_chunk = torch.exp(chunk - next_max.unsqueeze(-1))
        running_sum = running_sum * alpha + exp_chunk.sum(dim=-1)
        running_max = next_max
        chunks.append(chunk)
    probs: list[torch.Tensor] = []
    for chunk in chunks:
        probs.append(torch.exp(chunk - running_max.unsqueeze(-1)) / running_sum.unsqueeze(-1))
    return torch.cat(probs, dim=-1).reshape(q_len, num_q_heads, kv_len).to(scores.dtype)


def diagnostic_dequant_attention(
    query: torch.Tensor,
    cache: CompressedKVCache,
    *,
    scale: float | None = None,
    causal: bool = False,
    q_start_pos: int | None = None,
) -> AttentionResult:
    dense = cache.diagnostic_dense()
    result = dense_attention(
        query,
        dense.keys.to(query.device),
        dense.values.to(query.device),
        scale=scale,
        causal=causal,
        q_start_pos=q_start_pos,
    )
    report = AttentionReport(
        label="diagnostic_dequant",
        score_label="diagnostic_dequant",
        value_label="diagnostic_dequant",
        dense_materialized=True,
        q_len=result.report.q_len,
        kv_len=result.report.kv_len,
        num_q_heads=result.report.num_q_heads,
        num_kv_heads=result.report.num_kv_heads,
        head_dim=result.report.head_dim,
        segments=[{"kind": dense.label, "tokens": int(dense.keys.shape[0])}],
    )
    return AttentionResult(output=result.output, scores=result.scores, weights=result.weights, report=report)


def direct_qk_scores(
    query: torch.Tensor,
    cache: CompressedKVCache,
    *,
    scale: float | None = None,
    causal: bool = False,
    q_start_pos: int | None = None,
) -> tuple[torch.Tensor, AttentionReport]:
    q_len, num_q_heads, head_dim = _validate_query(query)
    if head_dim != cache.head_dim:
        raise ValueError(f"query/cache dim mismatch: {head_dim} vs {cache.head_dim}")
    scale = default_scale(head_dim) if scale is None else scale
    segments: list[torch.Tensor] = []
    segment_reports: list[dict[str, Any]] = []

    if cache.layer_keys is not None:
        scores = dense_scores(query, cache.layer_keys.to(query.device), scale=scale)
        segments.append(scores)
        segment_reports.append({"kind": "layer_dense", "tokens": int(cache.layer_keys.shape[0])})
    else:
        if cache.boundary_keys is not None:
            scores = dense_scores(query, cache.boundary_keys.to(query.device), scale=scale)
            segments.append(scores)
            segment_reports.append({"kind": "boundary", "tokens": int(cache.boundary_keys.shape[0])})
        for block in cache.blocks:
            scores = _block_qk_scores(query, block, cache, scale=scale)
            segments.append(scores)
            segment_reports.append({"kind": block.k_format, "tokens": block.length, "start": block.start})
        recent_k, _recent_v = cache.recent_window.peek()
        if recent_k is not None:
            scores = dense_scores(query, recent_k.to(query.device), scale=scale)
            segments.append(scores)
            segment_reports.append({"kind": "recent", "tokens": int(recent_k.shape[0])})

    if segments:
        out = torch.cat(segments, dim=-1)
    else:
        out = torch.empty(q_len, num_q_heads, 0, device=query.device, dtype=torch.float32)
    out = apply_causal_mask(out, causal=causal, q_start_pos=q_start_pos)
    report = AttentionReport(
        label="direct_qk",
        score_label="direct_qk",
        dense_materialized=False,
        q_len=q_len,
        kv_len=int(out.shape[-1]),
        num_q_heads=num_q_heads,
        num_kv_heads=cache.num_kv_heads,
        head_dim=head_dim,
        segments=segment_reports,
    )
    return out, report


def weighted_packed_v_accumulation(
    weights: torch.Tensor,
    cache: CompressedKVCache,
    *,
    num_q_heads: int,
    sparse_v_threshold: float | None = None,
) -> tuple[torch.Tensor, AttentionReport]:
    if weights.ndim != 3:
        raise ValueError("weights must have shape (tokens, query_heads, kv_tokens)")
    if sparse_v_threshold is not None and sparse_v_threshold < 0:
        raise ValueError("sparse_v_threshold must be non-negative")
    q_len, weight_heads, kv_len = weights.shape
    if weight_heads != num_q_heads:
        raise ValueError("weight heads do not match num_q_heads")
    parts: list[torch.Tensor] = []
    segments: list[dict[str, Any]] = []
    cursor = 0
    skipped_weights = 0
    skipped_weight_sum = 0.0

    def add_dense(values: torch.Tensor, kind: str) -> None:
        nonlocal cursor, skipped_weights, skipped_weight_sum
        length = int(values.shape[0])
        if length == 0:
            return
        segment_weights = weights[:, :, cursor : cursor + length]
        segment: dict[str, Any] = {"kind": kind, "tokens": length}
        if sparse_v_threshold is not None:
            skip_mask = segment_weights < sparse_v_threshold
            segment_skipped = int(skip_mask.sum().item())
            if segment_skipped:
                skipped_weights += segment_skipped
                skipped_weight_sum += float(segment_weights.masked_select(skip_mask).float().sum().item())
                segment_weights = segment_weights.masked_fill(skip_mask, 0)
            segment.update(
                {
                    "sparse_v_threshold": sparse_v_threshold,
                    "sparse_v_skipped_weights": segment_skipped,
                    "sparse_v_total_weights": int(segment_weights.numel()),
                }
            )
        parts.append(dense_value_accumulation(segment_weights, values, num_q_heads=num_q_heads))
        segments.append(segment)
        cursor += length

    if cache.layer_values is not None:
        add_dense(cache.layer_values.to(weights.device), "layer_dense")
    else:
        if cache.boundary_values is not None:
            add_dense(cache.boundary_values.to(weights.device), "boundary")
        for block in cache.blocks:
            values = dequantize_values(block.value).to(weights.device)
            add_dense(values, block.v_format)
        _recent_k, recent_v = cache.recent_window.peek()
        if recent_v is not None:
            add_dense(recent_v.to(weights.device), "recent")

    if cursor != kv_len:
        raise ValueError(f"weights cover {kv_len} tokens, but cache values cover {cursor}")
    if not parts:
        out = torch.zeros(q_len, num_q_heads, cache.head_dim, device=weights.device, dtype=weights.dtype)
    else:
        out = sum(parts)
    label: AttentionLabel = "sparse_weighted_packed_v" if sparse_v_threshold is not None else "weighted_packed_v"
    if sparse_v_threshold is not None:
        sparse_report = SparseVReport(
            threshold=sparse_v_threshold,
            total_weights=int(weights.numel()),
            skipped_weights=skipped_weights,
            skipped_weight_sum=skipped_weight_sum,
        ).to_dict()
        segments.append({"kind": "sparse_v", **sparse_report})
    report = AttentionReport(
        label=label,
        value_label=label,
        dense_materialized=False,
        q_len=q_len,
        kv_len=kv_len,
        num_q_heads=num_q_heads,
        num_kv_heads=cache.num_kv_heads,
        head_dim=cache.head_dim,
        segments=segments,
    )
    return out, report


def direct_qk_attention(
    query: torch.Tensor,
    cache: CompressedKVCache,
    *,
    scale: float | None = None,
    causal: bool = False,
    q_start_pos: int | None = None,
    softmax_block_size: int = 64,
    sparse_v_threshold: float | None = None,
) -> AttentionResult:
    if sparse_v_threshold is None and cache.policy.sparse_v:
        sparse_v_threshold = cache.policy.sparse_v_threshold
    scores, score_report = direct_qk_scores(
        query,
        cache,
        scale=scale,
        causal=causal,
        q_start_pos=q_start_pos,
    )
    weights = online_softmax(scores, block_size=softmax_block_size)
    output, value_report = weighted_packed_v_accumulation(
        weights,
        cache,
        num_q_heads=int(query.shape[1]),
        sparse_v_threshold=sparse_v_threshold,
    )
    report = AttentionReport(
        label="direct_qk",
        score_label="direct_qk",
        value_label=value_report.label,
        dense_materialized=False,
        q_len=score_report.q_len,
        kv_len=score_report.kv_len,
        num_q_heads=score_report.num_q_heads,
        num_kv_heads=score_report.num_kv_heads,
        head_dim=score_report.head_dim,
        segments=[
            {"scores": score_report.segments},
            {"values": value_report.segments},
        ],
    )
    return AttentionResult(output=output.to(query.dtype), scores=scores, weights=weights, report=report)


def fallback_dense_attention(
    query: torch.Tensor,
    cache: CompressedKVCache,
    *,
    reason: str,
    scale: float | None = None,
    causal: bool = False,
    q_start_pos: int | None = None,
) -> AttentionResult:
    dense = cache.diagnostic_dense()
    result = dense_attention(
        query,
        dense.keys.to(query.device),
        dense.values.to(query.device),
        scale=scale,
        causal=causal,
        q_start_pos=q_start_pos,
    )
    report = AttentionReport(
        label="fallback_dense",
        score_label="fallback_dense",
        value_label="fallback_dense",
        dense_materialized=True,
        q_len=result.report.q_len,
        kv_len=result.report.kv_len,
        num_q_heads=result.report.num_q_heads,
        num_kv_heads=result.report.num_kv_heads,
        head_dim=result.report.head_dim,
        segments=[{"kind": dense.label, "tokens": int(dense.keys.shape[0])}],
        fallback_events=[{"reason": reason}],
    )
    return AttentionResult(output=result.output, scores=result.scores, weights=result.weights, report=report)


def _block_qk_scores(
    query: torch.Tensor,
    block: CompressedKVBlock,
    cache: CompressedKVCache,
    *,
    scale: float,
) -> torch.Tensor:
    if isinstance(block.key, DenseKeyData):
        return dense_scores(query, block.key.data.to(query.device), scale=scale)
    if isinstance(block.key, QuantizedKeyData):
        keys = dequantize_k8(block.key).to(query.device)
        return dense_scores(query, keys, scale=scale)
    if isinstance(block.key, TurboKeyData):
        return _k4_block_scores(query, block, cache, scale=scale)
    raise TypeError(f"unsupported key payload: {type(block.key)!r}")


def _k4_block_scores(
    query: torch.Tensor,
    block: CompressedKVBlock,
    cache: CompressedKVCache,
    *,
    scale: float,
) -> torch.Tensor:
    q_len, num_q_heads, head_dim = _validate_query(query)
    if cache._k4_quantizer is None:
        cache._k4_quantizer = build_k4_quantizer(
            head_dim=cache.head_dim,
            layer_idx=cache.layer_idx,
            device=cache.device,
            seed=cache.seed,
        )
    quantizer = cache._k4_quantizer
    if quantizer is None:
        raise RuntimeError("K4 quantizer is not available")
    data = block.key.data
    rotated = rotate_forward(query.reshape(-1, head_dim), quantizer.rotation).reshape(q_len, num_q_heads, head_dim)
    kv_map = gqa_kv_heads(num_q_heads, cache.num_kv_heads, query.device)
    mapped = MSEData(
        indices=data.indices.to(query.device)[:, kv_map, :].permute(1, 0, 2).contiguous(),
        norms=data.norms.to(query.device)[:, kv_map].permute(1, 0).contiguous(),
        bits=data.bits,
        dim=data.dim,
    )
    scores = quantizer.score_rotated(rotated.permute(1, 0, 2).contiguous(), mapped, scale=scale)
    return scores.permute(1, 0, 2).contiguous()

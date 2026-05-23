from dataclasses import replace

import torch

from torbuquant.attention import (
    dense_attention,
    dense_sdpa_attention,
    dense_value_accumulation,
    diagnostic_dequant_attention,
    direct_qk_attention,
    direct_qk_scores,
    fallback_dense_attention,
    gqa_kv_heads,
    online_softmax,
    weighted_packed_v_accumulation,
)
from torbuquant.kv import BackendCapability, CompressedKVCache, qwen25_3b_policy
from torbuquant.kv.values import dequantize_values


def _kv(tokens: int, heads: int = 2, dim: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(tokens * heads * dim, dtype=torch.float32).reshape(tokens, heads, dim)
    keys = torch.sin(base / 71.0)
    values = torch.cos(base / 83.0)
    return keys, values


def _query(tokens: int, heads: int = 4, dim: int = 128) -> torch.Tensor:
    base = torch.arange(tokens * heads * dim, dtype=torch.float32).reshape(tokens, heads, dim)
    return torch.sin(base / 53.0)


def _cache(preset: str, *, recent_window: int = 2, chunk_size: int = 3) -> CompressedKVCache:
    backend = BackendCapability("vllm", fused_decode=True) if preset.startswith("k4") else BackendCapability("diagnostic")
    policy = replace(qwen25_3b_policy(preset=preset, backend=backend), recent_window=recent_window)
    return CompressedKVCache(
        head_dim=128,
        num_kv_heads=2,
        layer_idx=0,
        policy=policy,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_size=chunk_size,
    )


def test_gqa_kv_head_mapping():
    mapping = gqa_kv_heads(8, 2)
    torch.testing.assert_close(mapping, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))


def test_dense_sdpa_matches_dense_reference_without_mask():
    keys, values = _kv(5)
    query = _query(3)

    ref = dense_attention(query, keys, values)
    sdpa = dense_sdpa_attention(query, keys, values)

    torch.testing.assert_close(sdpa.output, ref.output, rtol=1e-5, atol=1e-5)
    assert sdpa.report.label == "dense_baseline"


def test_online_softmax_matches_torch_softmax():
    scores = torch.tensor(
        [
            [[1000.0, 999.0, -1000.0, 3.0, 4.0], [0.0, -1.0, -2.0, -3.0, -4.0]],
            [[5.0, 4.0, 3.0, 2.0, 1.0], [-30.0, 0.0, 30.0, -10.0, 10.0]],
        ]
    )

    out = online_softmax(scores, block_size=2)

    torch.testing.assert_close(out, torch.softmax(scores, dim=-1), rtol=1e-6, atol=1e-6)


def test_direct_qk_matches_diagnostic_scores_for_k16_v4():
    keys, values = _kv(8)
    query = _query(2)
    cache = _cache("k16_v4", recent_window=2, chunk_size=3)
    cache.prefill(keys, values)

    direct_scores, report = direct_qk_scores(query, cache)
    diagnostic = diagnostic_dequant_attention(query, cache)

    torch.testing.assert_close(direct_scores, diagnostic.scores, rtol=1e-6, atol=1e-6)
    assert report.label == "direct_qk"
    assert not report.dense_materialized


def test_direct_qk_matches_diagnostic_scores_for_k4_v4():
    keys, values = _kv(7)
    query = _query(1)
    cache = _cache("k4_v4", recent_window=1, chunk_size=3)
    cache.prefill(keys, values)

    direct_scores, report = direct_qk_scores(query, cache)
    diagnostic = diagnostic_dequant_attention(query, cache)

    torch.testing.assert_close(direct_scores, diagnostic.scores, rtol=2e-5, atol=2e-5)
    assert any(segment["kind"] == "K4" for segment in report.segments)


def test_weighted_packed_v_matches_diagnostic_value_accumulation():
    keys, values = _kv(9)
    query = _query(2)
    cache = _cache("k16_v4", recent_window=2, chunk_size=4)
    cache.prefill(keys, values)
    diagnostic = diagnostic_dequant_attention(query, cache)

    out, report = weighted_packed_v_accumulation(diagnostic.weights, cache, num_q_heads=query.shape[1])

    torch.testing.assert_close(out, diagnostic.output, rtol=1e-6, atol=1e-6)
    assert report.label == "weighted_packed_v"


def test_sparse_weighted_packed_v_threshold_zero_matches_base_path():
    keys, values = _kv(9)
    query = _query(2)
    cache = _cache("k16_v4", recent_window=2, chunk_size=4)
    cache.prefill(keys, values)
    diagnostic = diagnostic_dequant_attention(query, cache)

    base, _base_report = weighted_packed_v_accumulation(diagnostic.weights, cache, num_q_heads=query.shape[1])
    sparse, report = weighted_packed_v_accumulation(
        diagnostic.weights,
        cache,
        num_q_heads=query.shape[1],
        sparse_v_threshold=0.0,
    )

    torch.testing.assert_close(sparse, base, rtol=1e-6, atol=1e-6)
    assert report.label == "sparse_weighted_packed_v"
    assert report.segments[-1]["kind"] == "sparse_v"
    assert report.segments[-1]["skipped_weights"] == 0


def test_sparse_weighted_packed_v_applies_attention_threshold():
    keys, values = _kv(6)
    cache = _cache("k16_v4", recent_window=0, chunk_size=6)
    cache.prefill(keys, values)
    weights = torch.zeros(1, 4, 6)
    weights[:, :, 0] = 0.7
    weights[:, :, 1] = 0.2
    weights[:, :, 2] = 0.09
    weights[:, :, 3] = 0.009
    weights[:, :, 4] = 0.0009
    weights[:, :, 5] = 0.00009

    out, report = weighted_packed_v_accumulation(
        weights,
        cache,
        num_q_heads=4,
        sparse_v_threshold=0.001,
    )
    masked = weights.masked_fill(weights < 0.001, 0)
    expected = dense_value_accumulation(masked, dequantize_values(cache.blocks[0].value), num_q_heads=4)

    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
    assert report.segments[-1]["kind"] == "sparse_v"
    assert report.segments[-1]["skipped_weights"] == 8
    assert report.segments[-1]["total_weights"] == 24


def test_direct_qk_attention_matches_diagnostic_reference_for_k16_v4():
    keys, values = _kv(9)
    query = _query(2)
    cache = _cache("k16_v4", recent_window=2, chunk_size=4)
    cache.prefill(keys, values)

    direct = direct_qk_attention(query, cache, softmax_block_size=3)
    diagnostic = diagnostic_dequant_attention(query, cache)

    torch.testing.assert_close(direct.output, diagnostic.output, rtol=1e-6, atol=1e-6)
    assert direct.report.score_label == "direct_qk"
    assert direct.report.value_label == "weighted_packed_v"
    assert not direct.report.dense_materialized


def test_direct_qk_attention_reports_sparse_value_path():
    keys, values = _kv(10)
    query = _query(1)
    cache = _cache("k16_v4", recent_window=0, chunk_size=10)
    cache.prefill(keys, values)

    result = direct_qk_attention(query, cache, sparse_v_threshold=1e-6)

    assert result.report.value_label == "sparse_weighted_packed_v"
    value_segments = result.report.segments[1]["values"]
    assert value_segments[-1]["kind"] == "sparse_v"
    assert value_segments[-1]["threshold"] == 1e-6


def test_direct_qk_attention_uses_sparse_v_policy_threshold():
    keys, values = _kv(10)
    query = _query(1)
    policy = replace(
        qwen25_3b_policy(
            preset="sparse_v",
            backend=BackendCapability("diagnostic", supports_sparse_v=True),
        ),
        recent_window=0,
    )
    cache = CompressedKVCache(
        head_dim=128,
        num_kv_heads=2,
        layer_idx=0,
        policy=policy,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_size=10,
    )
    cache.prefill(keys, values)

    result = direct_qk_attention(query, cache)

    assert result.report.value_label == "sparse_weighted_packed_v"
    assert result.report.segments[1]["values"][-1]["threshold"] == 1e-6


def test_fallback_dense_label_is_visible():
    keys, values = _kv(4)
    query = _query(1)
    cache = _cache("k16_v4", recent_window=1, chunk_size=2)
    cache.prefill(keys, values)

    result = fallback_dense_attention(query, cache, reason="test_reason")

    assert result.report.label == "fallback_dense"
    assert result.report.dense_materialized
    assert result.report.fallback_events == [{"reason": "test_reason"}]

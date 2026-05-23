from dataclasses import replace

import pytest
import torch

from torbuquant.attention import direct_qk_attention, direct_qk_scores, weighted_packed_v_accumulation
from torbuquant.kv import BackendCapability, CompressedKVCache, qwen25_3b_policy
from torbuquant.triton import (
    KernelCounters,
    UnsupportedFormatError,
    decode_block,
    score_k16,
    score_k4,
    score_k8,
    triton_available,
    weighted_v4,
)


pytestmark = pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels require CUDA")


def _kv(tokens: int, heads: int = 2, dim: int = 128, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda") if device is None else device
    base = torch.arange(tokens * heads * dim, dtype=torch.float32, device=device).reshape(tokens, heads, dim)
    keys = torch.sin(base / 113.0)
    values = torch.cos(base / 127.0)
    return keys, values


def _query(tokens: int = 1, heads: int = 4, dim: int = 128, device: torch.device | None = None) -> torch.Tensor:
    device = torch.device("cuda") if device is None else device
    base = torch.arange(tokens * heads * dim, dtype=torch.float32, device=device).reshape(tokens, heads, dim)
    return torch.sin(base / 97.0)


def _cache(preset: str, tokens: int = 17) -> CompressedKVCache:
    device = torch.device("cuda")
    backend = BackendCapability("vllm", fused_decode=True) if preset.startswith("k4") else BackendCapability("diagnostic")
    policy = replace(qwen25_3b_policy(preset=preset, backend=backend), recent_window=0)
    cache = CompressedKVCache(
        head_dim=128,
        num_kv_heads=2,
        layer_idx=0,
        policy=policy,
        device=device,
        dtype=torch.float32,
        chunk_size=tokens,
    )
    keys, values = _kv(tokens, device=device)
    cache.prefill(keys, values)
    return cache


def test_score_k16_matches_reference_scores():
    cache = _cache("k16_v4")
    query = _query(tokens=2)
    block = cache.blocks[0]

    scores = score_k16(query, block.key.data, num_kv_heads=cache.num_kv_heads, scale=1.0 / (128**0.5))
    ref, _report = direct_qk_scores(query, cache)

    torch.testing.assert_close(scores, ref, rtol=1e-5, atol=1e-5)


def test_score_k8_matches_reference_scores():
    cache = _cache("k8_v4")
    query = _query(tokens=2)
    block = cache.blocks[0]

    scores = score_k8(query, block.key.data, num_kv_heads=cache.num_kv_heads, scale=1.0 / (128**0.5))
    ref, _report = direct_qk_scores(query, cache)

    torch.testing.assert_close(scores, ref, rtol=1e-5, atol=1e-5)


def test_score_k4_matches_reference_scores():
    cache = _cache("k4_v4")
    query = _query(tokens=2)
    block = cache.blocks[0]

    scores = score_k4(
        query,
        block,
        num_kv_heads=cache.num_kv_heads,
        scale=1.0 / (128**0.5),
        quantizer=cache._k4_quantizer,
    )
    ref, _report = direct_qk_scores(query, cache)

    torch.testing.assert_close(scores, ref, rtol=2e-4, atol=2e-4)


def test_weighted_v4_reads_packed_values_inside_kernel():
    cache = _cache("k16_v4", tokens=19)
    query = _query(tokens=2)
    block = cache.blocks[0]
    ref_attention = direct_qk_attention(query, cache)

    out = weighted_v4(ref_attention.weights, block.value, num_kv_heads=cache.num_kv_heads)
    ref, report = weighted_packed_v_accumulation(ref_attention.weights, cache, num_q_heads=query.shape[1])

    torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)
    assert report.label == "weighted_packed_v"


def test_weighted_v4_sparse_threshold_matches_reference_masking():
    cache = _cache("k16_v4", tokens=19)
    weights = torch.zeros(2, 4, 19, device=torch.device("cuda"))
    weights[:, :, 0] = 0.5
    weights[:, :, 1] = 0.25
    weights[:, :, 2] = 0.001
    weights[:, :, 3:] = 0.00001
    block = cache.blocks[0]

    out = weighted_v4(weights, block.value, num_kv_heads=cache.num_kv_heads, sparse_v_threshold=0.001)
    ref, report = weighted_packed_v_accumulation(
        weights,
        cache,
        num_q_heads=weights.shape[1],
        sparse_v_threshold=0.001,
    )

    torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)
    assert report.label == "sparse_weighted_packed_v"


@pytest.mark.parametrize("preset", ["k16_v4", "k8_v4"])
def test_decode_block_for_k16_and_k8_matches_reference(preset):
    cache = _cache(preset, tokens=23)
    query = _query(tokens=2)
    block = cache.blocks[0]
    counters = KernelCounters()

    result = decode_block(
        query,
        block,
        num_kv_heads=cache.num_kv_heads,
        scale=1.0 / (128**0.5),
        counters=counters,
    )
    ref = direct_qk_attention(query, cache)

    torch.testing.assert_close(result.output, ref.output, rtol=2e-5, atol=2e-5)
    assert result.report.counters["score_calls"] == 1
    assert result.report.counters["v_accumulation_calls"] == 1


def test_decode_block_sparse_v_for_k16_matches_reference():
    cache = _cache("k16_v4", tokens=23)
    query = _query(tokens=2)
    block = cache.blocks[0]
    counters = KernelCounters()

    result = decode_block(
        query,
        block,
        num_kv_heads=cache.num_kv_heads,
        scale=1.0 / (128**0.5),
        counters=counters,
        sparse_v_threshold=1e-6,
    )
    ref = direct_qk_attention(query, cache, sparse_v_threshold=1e-6)

    torch.testing.assert_close(result.output, ref.output, rtol=2e-5, atol=2e-5)
    assert result.report.counters["v_accumulation_calls"] == 1


def test_fused_decode_k4v4_matches_reference_for_one_token_decode():
    cache = _cache("k4_v4", tokens=29)
    query = _query(tokens=1)
    block = cache.blocks[0]
    counters = KernelCounters()

    result = decode_block(
        query,
        block,
        num_kv_heads=cache.num_kv_heads,
        scale=1.0 / (128**0.5),
        quantizer=cache._k4_quantizer,
        counters=counters,
    )
    ref = direct_qk_attention(query, cache)

    torch.testing.assert_close(result.output, ref.output, rtol=2e-4, atol=2e-4)
    assert result.scores is None
    assert result.report.name == "fused_decode_k4v4"
    assert result.report.counters["fused_calls"] == 1


def test_fused_decode_k4v4_rejects_sparse_v_until_two_pass_kernel_exists():
    cache = _cache("k4_v4", tokens=29)
    query = _query(tokens=1)

    with pytest.raises(UnsupportedFormatError, match="two-pass"):
        decode_block(
            query,
            cache.blocks[0],
            num_kv_heads=cache.num_kv_heads,
            scale=1.0 / (128**0.5),
            quantizer=cache._k4_quantizer,
            sparse_v_threshold=1e-6,
        )


def test_decode_block_rejects_missing_kernel_format():
    cache = _cache("boundary_v", tokens=11)
    query = _query(tokens=1)

    with pytest.raises(UnsupportedFormatError, match="value format"):
        decode_block(
            query,
            cache.blocks[0],
            num_kv_heads=cache.num_kv_heads,
            scale=1.0 / (128**0.5),
        )

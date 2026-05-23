from dataclasses import replace

import pytest
import torch

from torbuquant.kv import (
    AutoPolicyInput,
    BackendCapability,
    CompressedKVCache,
    KVQuantPolicy,
    RecentWindow,
    choose_auto_kv_policy,
    estimate_dense_kv_bytes,
    estimate_policy_kv_bytes,
    qwen25_3b_policy,
)
from torbuquant.integration.common.config import KVQuantConfig, policy_from_config


def _kv(tokens: int, heads: int = 2, dim: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(tokens * heads * dim, dtype=torch.float32).reshape(tokens, heads, dim)
    keys = torch.sin(base / 97.0)
    values = torch.cos(base / 89.0)
    return keys, values


def test_recent_window_returns_oldest_overflow_and_keeps_tail():
    keys, values = _kv(5, heads=1, dim=8)
    window = RecentWindow(
        capacity=3,
        num_kv_heads=1,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    overflow_k, overflow_v = window.append(keys[:2], values[:2])
    assert overflow_k is None
    overflow_k, overflow_v = window.append(keys[2:], values[2:])

    assert window.size == 3
    torch.testing.assert_close(overflow_k, keys[:2])
    torch.testing.assert_close(overflow_v, values[:2])
    recent_k, recent_v = window.peek()
    torch.testing.assert_close(recent_k, keys[2:])
    torch.testing.assert_close(recent_v, values[2:])


def test_prefill_compresses_prefix_and_keeps_recent_tail():
    keys, values = _kv(10)
    policy = replace(qwen25_3b_policy(preset="k16_v4"), recent_window=3)
    cache = CompressedKVCache(
        head_dim=128,
        num_kv_heads=2,
        layer_idx=1,
        policy=policy,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_size=4,
    )

    cache.prefill(keys, values)
    handle = cache.production_handle()

    assert cache.seq_len == 10
    assert cache.compressed_tokens == 7
    assert cache.recent_tokens == 3
    assert [block.length for block in handle.blocks] == [4, 3]
    with pytest.raises(RuntimeError, match="production cache handle"):
        handle.dense()

    dense = cache.diagnostic_dense()
    assert dense.label == "diagnostic_dequant"
    torch.testing.assert_close(dense.keys, keys, rtol=0, atol=0)
    assert dense.values.shape == values.shape


def test_decode_append_flushes_recent_overflow_to_blocks():
    keys, values = _kv(7)
    policy = replace(qwen25_3b_policy(preset="k16_v4"), recent_window=2)
    cache = CompressedKVCache(
        head_dim=128,
        num_kv_heads=2,
        layer_idx=0,
        policy=policy,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_size=3,
    )

    cache.prefill(keys[:2], values[:2])
    for idx in range(2, 7):
        cache.append_decode(keys[idx : idx + 1], values[idx : idx + 1])

    assert cache.seq_len == 7
    assert cache.compressed_tokens == 5
    assert cache.recent_tokens == 2
    assert [block.length for block in cache.blocks] == [1, 1, 1, 1, 1]


def test_boundary_tokens_are_stored_dense_before_compressed_region():
    keys, values = _kv(9)
    policy = replace(qwen25_3b_policy(preset="boundary_v"), recent_window=2)
    cache = CompressedKVCache(
        head_dim=128,
        num_kv_heads=2,
        layer_idx=2,
        policy=policy,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_size=3,
    )

    cache.prefill(keys, values)
    report = cache.to_report()
    dense = cache.diagnostic_dense()

    assert report["boundary"]["stored_tokens"] == 4
    assert cache.compressed_tokens == 3
    assert cache.recent_tokens == 2
    torch.testing.assert_close(dense.keys[:4], keys[:4], rtol=0, atol=0)
    torch.testing.assert_close(dense.values[:4], values[:4], rtol=0, atol=0)


def test_layer_preservation_keeps_layer_dense_and_counts_boundary_bytes():
    keys, values = _kv(6)
    policy = KVQuantPolicy(
        preset="layer_boundary",
        k_format="K16",
        v_format="V4",
        recent_window=2,
        preserve_first_layers=1,
    )
    cache = CompressedKVCache(
        head_dim=128,
        num_kv_heads=2,
        layer_idx=0,
        policy=policy,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_size=3,
        num_layers=4,
    )

    cache.prefill(keys[:4], values[:4])
    cache.append_decode(keys[4:5], values[4:5])
    cache.append_decode(keys[5:6], values[5:6])
    dense = cache.diagnostic_dense()
    memory = cache.memory_report()

    assert cache.layer_preserved
    assert cache.compressed_tokens == 0
    assert cache.recent_tokens == 0
    assert dense.label == "diagnostic_dense_layer"
    torch.testing.assert_close(dense.keys, keys, rtol=0, atol=0)
    torch.testing.assert_close(dense.values, values, rtol=0, atol=0)
    assert memory.boundary_bytes == keys.numel() * keys.element_size() + values.numel() * values.element_size()


def test_k4_cache_counts_codebook_and_rotation_metadata():
    keys, values = _kv(4)
    policy = replace(
        qwen25_3b_policy(
            preset="k4_v4",
            backend=BackendCapability("vllm", fused_decode=True),
        ),
        recent_window=0,
    )
    cache = CompressedKVCache(
        head_dim=128,
        num_kv_heads=2,
        layer_idx=3,
        policy=policy,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_size=4,
    )

    cache.prefill(keys, values)
    memory = cache.memory_report()
    report = cache.to_report()

    assert cache.compressed_tokens == 4
    assert memory.centroids_bytes > 0
    assert memory.rotation_bytes > 0
    assert report["k4_metadata"]["rotation"]["kind"] == "rht"


def test_qwen_policy_gates_backend_and_memory_selection():
    policy = qwen25_3b_policy(
        preset="k4_v4",
        backend=BackendCapability("diagnostic", fused_decode=False),
    )
    assert policy.k_format == "K16"
    assert policy.fallback_count == 1

    memory_policy = qwen25_3b_policy(
        preset="k16_v4",
        vram_budget_bytes=1_000_000,
        quality_priority="memory",
    )
    assert memory_policy.k_format == "K8"

    with pytest.raises(ValueError, match="sparse_v"):
        qwen25_3b_policy(preset="sparse_v", backend=BackendCapability("vllm"))

    with pytest.raises(ValueError, match="HF K4"):
        qwen25_3b_policy(
            preset="k4_v4",
            backend=BackendCapability("hf", fused_decode=False),
            quality_priority="memory",
        )


def test_sparse_v_policy_is_available_for_diagnostic_reference_path():
    policy = qwen25_3b_policy(
        preset="sparse_v",
        backend=BackendCapability("diagnostic", supports_sparse_v=True),
    )

    assert policy.sparse_v
    assert policy.sparse_v_threshold == 1e-6
    assert policy.k_format == "K16"
    assert policy.v_format == "V4"

    configured = policy_from_config(
        KVQuantConfig(
            preset="sparse_v",
            backend="diagnostic",
            sparse_v_threshold=1e-7,
        )
    )
    assert configured.sparse_v_threshold == 1e-7


def test_auto_policy_keeps_dense_when_budget_allows():
    dense = estimate_dense_kv_bytes(
        batch_size=1,
        num_layers=36,
        num_kv_heads=2,
        head_dim=128,
        context_tokens=1024,
    )

    decision = choose_auto_kv_policy(
        AutoPolicyInput(
            batch_size=1,
            num_layers=36,
            num_kv_heads=2,
            num_q_heads=16,
            head_dim=128,
            context_tokens=1024,
            available_kv_memory_bytes=dense * 2,
            model_id="Qwen/Qwen2.5-3B",
            prefer_low_latency=True,
        )
    )

    assert decision.backend == "dense"
    assert decision.policy is None
    assert decision.dense_bytes == dense
    assert decision.cache_bytes == dense
    assert decision.compression_ratio == 1.0


def test_auto_policy_selects_qwen_key_preserving_mode_under_pressure():
    dense = estimate_dense_kv_bytes(
        batch_size=4,
        num_layers=36,
        num_kv_heads=2,
        head_dim=128,
        context_tokens=8192,
    )

    decision = choose_auto_kv_policy(
        AutoPolicyInput(
            batch_size=4,
            num_layers=36,
            num_kv_heads=2,
            num_q_heads=16,
            head_dim=128,
            context_tokens=8192,
            available_kv_memory_bytes=int(dense / 1.1),
            model_id="Qwen/Qwen2.5-3B",
        )
    )

    assert decision.backend == "torbuquant"
    assert decision.policy is not None
    assert decision.policy.k_format == "K16"
    assert decision.policy.v_format == "V4"
    assert decision.cache_bytes < decision.dense_bytes
    assert decision.compression_ratio > 1.0


def test_auto_policy_moves_to_k4v4_when_other_modes_exceed_budget():
    dense = estimate_dense_kv_bytes(
        batch_size=16,
        num_layers=36,
        num_kv_heads=2,
        head_dim=128,
        context_tokens=8192,
    )
    k4v4 = qwen25_3b_policy(
        preset="k4_v4",
        backend=BackendCapability("vllm", fused_decode=True),
        context_length=8192,
        batch_size=16,
    )
    k4v4_bytes = estimate_policy_kv_bytes(
        k4v4,
        batch_size=16,
        num_layers=36,
        num_kv_heads=2,
        head_dim=128,
        context_tokens=8192,
    )

    decision = choose_auto_kv_policy(
        AutoPolicyInput(
            batch_size=16,
            num_layers=36,
            num_kv_heads=2,
            num_q_heads=16,
            head_dim=128,
            context_tokens=8192,
            available_kv_memory_bytes=int(k4v4_bytes * 1.01),
            model_id="Qwen/Qwen2.5-3B",
        )
    )

    assert decision.backend == "torbuquant"
    assert decision.policy is not None
    assert decision.policy.k_format == "K4"
    assert decision.policy.v_format == "V4"
    assert decision.cache_bytes <= int(k4v4_bytes * 1.01)
    assert decision.dense_bytes == dense

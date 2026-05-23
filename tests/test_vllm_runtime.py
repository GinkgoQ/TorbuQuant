from __future__ import annotations

import pytest
import torch

from torbuquant.integration.vllm import (
    VLLMRuntimeConfig,
    build_layer_context,
    build_metadata,
    build_page_geometry,
    cache_dtype_info,
    kv_head_for_query_head,
    parse_tq_gate_env,
    requires_cuda_graph_buffers,
    save_metadata,
    select_decode_path,
    select_prefill_path,
    validate_runtime_gate,
)


def _runtime_config(**kwargs) -> VLLMRuntimeConfig:
    values = {
        "cache_dtype": "tq4",
        "enabled": True,
        "model_name_or_path": None,
        "metadata_path": None,
        "num_q_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
        "block_size": 16,
        "k_bits": 4,
        "v_bits": 4,
    }
    values.update(kwargs)
    return VLLMRuntimeConfig(**values)


def test_registry_reports_tq4_and_recipe_modes():
    assert cache_dtype_info("tq4").mode == "tq4"
    assert cache_dtype_info("turboquant35").requires_metadata
    assert select_decode_path("tq4") == "tq4_paged"
    assert select_decode_path("turboquant25") == "recipe_paged"


def test_runtime_gate_rejects_when_disabled():
    with pytest.raises(ValueError, match="enable_turboquant"):
        validate_runtime_gate(cache_dtype="tq4", enabled=False)


def test_prefill_and_cuda_graph_selection():
    assert parse_tq_gate_env({"TQ4_USE_FUSED_PAGED": "1", "TQ4_USE_INT8_PREFILL": "yes"}) == {
        "fused_paged": True,
        "int8_prefill": True,
    }
    assert select_prefill_path(has_cache=False, int8_enabled=True, mixed_with_cache=False) == "dense_live"
    assert select_prefill_path(has_cache=True, int8_enabled=True, mixed_with_cache=False) == "int8_query"
    assert select_prefill_path(has_cache=True, int8_enabled=True, mixed_with_cache=True) == "packed_cache"
    assert requires_cuda_graph_buffers(decode_path="tq4_paged", single_token_decode=True)


def test_page_geometry_uses_recipe_rows(tmp_path):
    metadata = build_metadata(
        recipe="turboquant35",
        head_dim=128,
        num_kv_heads=2,
        layer_names=["model.layers.0.self_attn"],
    )
    path = tmp_path / "turboquant_kv.json"
    save_metadata(metadata, path)
    config = _runtime_config(
        cache_dtype="turboquant35",
        metadata_path=str(path),
    )

    context = build_layer_context(config=config, layer_name="model.layers.0.self_attn")
    geometry = build_page_geometry(config, num_blocks=2)

    assert context.metadata is not None
    assert geometry.k_row_bytes == 64
    assert geometry.v_row_bytes == 64
    assert geometry.row_bytes == 128


def test_recipe_context_rejects_head_dim_mismatch(tmp_path):
    metadata = build_metadata(
        recipe="turboquant25",
        head_dim=64,
        num_kv_heads=2,
        layer_names=["layer0"],
    )
    path = tmp_path / "turboquant_kv.json"
    save_metadata(metadata, path)

    with pytest.raises(ValueError, match="head_dim"):
        build_layer_context(
            config=_runtime_config(cache_dtype="turboquant25", metadata_path=str(path)),
            layer_name="layer0",
        )


def test_query_to_kv_head_mapping():
    mapping = kv_head_for_query_head(8, 2, torch.device("cpu"))

    assert mapping.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="divisible"):
        kv_head_for_query_head(7, 2, torch.device("cpu"))

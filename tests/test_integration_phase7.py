import math
import json

import pytest
import torch

from torbuquant.attention.reference import direct_qk_attention
from torbuquant.integration import KVQuantConfig, ProductionModeError, detect_backend, policy_from_config
from torbuquant.integration.hf import HFDiagnosticCacheAdapter, build_layer_cache_from_capture, hf_qwen_settings
from torbuquant.integration.vllm import (
    CalibrationMetadata,
    LayerMetadata,
    PagedCacheSpec,
    PagedCompressedKVCache,
    TensorMetadata,
    build_metadata,
    build_recipe_tables,
    discover_metadata_path,
    env_flag,
    get_recipe_bits,
    group_dims,
    is_turboquant_cuda,
    is_turboquant_dtype,
    load_metadata,
    packed_dim,
    pack_recipe_vectors,
    padded_slot_bytes,
    parse_tq_bits_env,
    recipe_layout,
    save_metadata,
    slice_layer_metadata_for_tp,
    tq_bytes_per_token,
    tq_bytes_per_token_kv,
    tq_index_bytes,
    unpack_recipe_vectors,
)
from torbuquant.kv import CompressedKVCache
from torbuquant.triton import KernelCounters, triton_available


def _config(**kwargs):
    data = {
        "preset": "k16_v4",
        "backend": "hf",
        "mode": "diagnostic",
        "context_length": 128,
        "num_q_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
        "recent_window": 0,
    }
    data.update(kwargs)
    return KVQuantConfig(**data)


def test_common_config_translates_to_policy_and_capability_report():
    config = _config()
    capability = detect_backend("hf")
    policy = policy_from_config(config)
    assert capability.backend == "hf"
    assert policy.k_format == "K16"
    assert policy.v_format == "V4"
    assert policy.recent_window == 0
    assert capability.to_dict()["versions"]["torch"]


def test_hf_settings_rejects_dynamic_cache_production_without_kernel_contract(monkeypatch):
    config = _config(mode="production")
    monkeypatch.setattr("torbuquant.integration.hf.config.detect_backend", lambda _name: detect_backend("diagnostic"))
    with pytest.raises(ProductionModeError):
        hf_qwen_settings(config)


def test_hf_diagnostic_adapter_compresses_and_reports_real_cache_state():
    config = _config()
    adapter = HFDiagnosticCacheAdapter(config=config, device=torch.device("cpu"))
    key = torch.randn(1, 2, 8, 128, dtype=torch.float16)
    value = torch.randn(1, 2, 8, 128, dtype=torch.float16)

    out_k, out_v = adapter.update(key, value, layer_idx=0)
    report = adapter.report()

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert report["diagnostic_label"] == "diagnostic_dequant"
    assert report["cache"]["layers"][0]["compressed_tokens"] == 8
    assert report["memory"]["0"]["compressed_v_bytes"] > 0


def test_hf_capture_cache_builder_uses_project_cache_model():
    config = _config()
    keys = torch.randn(12, 2, 128, dtype=torch.float16)
    values = torch.randn(12, 2, 128, dtype=torch.float16)
    cache = build_layer_cache_from_capture(
        config=config,
        keys=keys,
        values=values,
        device=torch.device("cpu"),
    )
    report = cache.to_report()
    assert report["k_format"] == "K16"
    assert report["v_format"] == "V4"
    assert report["compressed_tokens"] == 12
    assert cache.memory_report().compressed_total_bytes < cache.memory_report().dense_kv_bytes


def test_vllm_page_spec_counts_slot_bytes_for_k16v4():
    spec = PagedCacheSpec(block_size=16, num_kv_heads=2, head_dim=128, k_format="K16", v_format="V4")
    components = spec.slot_components
    assert components.k_data == 2 * 128 * 2
    assert components.v_data == 2 * 64
    assert components.v_scales == 2 * 4 * 2
    assert spec.slot_bytes == components.total
    assert spec.page_bytes == 16 * spec.slot_bytes
    assert spec.shape(3) == (3, 16, spec.slot_bytes)


def test_vllm_turboquant_recipe_registry_and_layout():
    assert is_turboquant_dtype("turboquant25")
    assert is_turboquant_dtype("turboquant35")
    assert not is_turboquant_dtype("turboquant4")
    assert get_recipe_bits("turboquant25") == 2.5
    assert get_recipe_bits("turboquant35") == 3.5
    assert group_dims(128, "turboquant25") == (32, 96)
    assert group_dims(128, "turboquant35") == (64, 64)
    assert packed_dim(128, "turboquant25") == 44
    assert packed_dim(128, "turboquant35") == 64

    layout = recipe_layout("turboquant25", 128)
    assert layout.groups[0].mse_bits == 2
    assert layout.groups[1].mse_bits == 1
    assert layout.packed_dim == sum(group.packed_bytes for group in layout.groups)
    assert is_turboquant_cuda((8, 6))
    assert not is_turboquant_cuda((9, 0))


def test_vllm_page_spec_counts_recipe_slot_bytes():
    spec = PagedCacheSpec(
        block_size=16,
        num_kv_heads=2,
        head_dim=128,
        k_format="K4",
        v_format="V4",
        cache_dtype="turboquant35",
    )

    assert spec.slot_components.k_data == 2 * 64
    assert spec.slot_components.v_data == 2 * 64
    assert spec.slot_bytes == 256
    assert spec.page_bytes == 16 * 256
    assert spec.to_dict()["recipe_layout"]["packed_dim"] == 64


def test_vllm_tq4_env_and_byte_helpers():
    assert env_flag("TQ4_USE_FUSED_PAGED", {"TQ4_USE_FUSED_PAGED": "1"})
    assert env_flag("TQ4_USE_INT8_PREFILL", {"TQ4_USE_INT8_PREFILL": "yes"})
    assert not env_flag("TQ4_USE_FUSED_PAGED", {"TQ4_USE_FUSED_PAGED": "0"})
    assert parse_tq_bits_env({"TQ4_K_BITS": "4", "TQ4_V_BITS": "3"}) == (4, 3)

    assert tq_index_bytes(128, 4) == 64
    assert tq_index_bytes(128, 3) == 48
    assert tq_index_bytes(128, 2) == 32
    assert tq_bytes_per_token(128, 4) == 68
    assert tq_bytes_per_token_kv(128, k_bits=4, v_bits=3) == 120
    assert padded_slot_bytes(128, k_bits=4, v_bits=3) == 128

    with pytest.raises(ValueError, match="one of"):
        parse_tq_bits_env({"TQ4_K_BITS": "6", "TQ4_V_BITS": "4"})


def test_vllm_turboquant_metadata_roundtrip_and_discovery(tmp_path):
    metadata = build_metadata(
        recipe="turboquant25",
        head_dim=128,
        num_kv_heads=2,
        layer_names=["model.layers.0.self_attn"],
        model_name="tests/qwen",
    )
    metadata = type(metadata)(
        recipe=metadata.recipe,
        head_dim=metadata.head_dim,
        model_name=metadata.model_name,
        layers=metadata.layers,
        calibration=CalibrationMetadata(
            method="activation_energy_v1",
            objective="sum_squared_activation",
            num_prompts=4,
            max_seq_len=512,
            batch_size=1,
            num_observed_tokens=2048,
            dtype="float16",
            device="cuda:0",
            prompts_sha256="abc123",
        ),
    )
    model_dir = tmp_path / "model"
    path = model_dir / "turboquant_kv.json"

    save_metadata(metadata, path)
    loaded = load_metadata(str(path))

    assert discover_metadata_path(str(model_dir), None) == str(path)
    assert loaded.recipe == "turboquant25"
    assert loaded.head_dim == 128
    assert loaded.calibration is not None
    assert loaded.calibration.num_observed_tokens == 2048
    layer = loaded.get_layer("model.layers.0.self_attn.attn")
    high, low = layer.key.get_group_indices(
        device=torch.device("cpu"),
        head_dim=128,
        recipe="turboquant25",
    )
    assert high.shape == (2, 32)
    assert low.shape == (2, 96)


def test_vllm_turboquant_metadata_accepts_head_dim_alias(tmp_path):
    payload = {
        "version": 1,
        "recipe": "turboquant35",
        "head_dim": 128,
        "model_name": None,
        "layers": {
            "layer0": {
                "key_high_precision_indices": [list(range(64))],
                "value_high_precision_indices": [list(range(64))],
            }
        },
    }
    path = tmp_path / "turboquant_kv.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_metadata(str(path))

    assert loaded.head_dim == 128
    assert loaded.get_layer("layer0").key.high_precision_indices == (tuple(range(64)),)


def test_vllm_turboquant_metadata_slices_partitioned_and_replicated_heads():
    partitioned = LayerMetadata(
        key=TensorMetadata(tuple(tuple(range(i, i + 32)) for i in (0, 32, 64, 96))),
        value=TensorMetadata(tuple(tuple(range(i, i + 32)) for i in (128, 160, 192, 224))),
    )
    part = slice_layer_metadata_for_tp(
        partitioned,
        num_kv_heads=2,
        tp_rank=1,
        tp_size=2,
    )
    assert part.key.high_precision_indices == (tuple(range(64, 96)), tuple(range(96, 128)))

    replicated = LayerMetadata(
        key=TensorMetadata((tuple(range(32)), tuple(range(32, 64)))),
        value=TensorMetadata((tuple(range(64, 96)), tuple(range(96, 128)))),
    )
    repl = slice_layer_metadata_for_tp(
        replicated,
        num_kv_heads=1,
        tp_rank=2,
        tp_size=4,
    )
    assert repl.key.high_precision_indices == (tuple(range(32, 64)),)


def test_vllm_turboquant_metadata_rejects_bad_group_indices():
    metadata = TensorMetadata((tuple(reversed(range(32))),))
    with pytest.raises(ValueError, match="sorted"):
        metadata.get_group_indices(
            device=torch.device("cpu"),
            head_dim=128,
            recipe="turboquant25",
        )


def test_vllm_turboquant_recipe_vector_pack_roundtrip():
    torch.manual_seed(31)
    x = torch.randn(6, 2, 128, dtype=torch.float32)
    metadata = build_metadata(
        recipe="turboquant25",
        head_dim=128,
        num_kv_heads=2,
        layer_names=["layer0"],
    )
    groups = metadata.get_layer("layer0").key.get_group_indices(
        device=torch.device("cpu"),
        head_dim=128,
        recipe="turboquant25",
    )
    tables = build_recipe_tables(
        recipe="turboquant25",
        head_dim=128,
        group_indices=groups,
        device=torch.device("cpu"),
    )

    packed = pack_recipe_vectors(x, "turboquant25", tables)
    restored = unpack_recipe_vectors(
        packed,
        "turboquant25",
        tables,
        dtype=torch.float32,
    )

    assert packed.shape == (6, 2, packed_dim(128, "turboquant25"))
    assert restored.shape == x.shape
    assert torch.isfinite(restored).all()
    assert ((x - restored) ** 2).mean().item() < 0.25


def test_vllm_turboquant_recipe35_has_lower_error_than_recipe25():
    torch.manual_seed(37)
    x = torch.randn(8, 2, 128, dtype=torch.float32)
    errors = {}
    for recipe in ("turboquant25", "turboquant35"):
        metadata = build_metadata(
            recipe=recipe,
            head_dim=128,
            num_kv_heads=2,
            layer_names=["layer0"],
        )
        groups = metadata.get_layer("layer0").key.get_group_indices(
            device=torch.device("cpu"),
            head_dim=128,
            recipe=recipe,
        )
        tables = build_recipe_tables(
            recipe=recipe,
            head_dim=128,
            group_indices=groups,
            device=torch.device("cpu"),
        )
        packed = pack_recipe_vectors(x, recipe, tables)
        restored = unpack_recipe_vectors(packed, recipe, tables, dtype=torch.float32)
        errors[recipe] = ((x - restored) ** 2).mean().item()

    assert errors["turboquant35"] < errors["turboquant25"]


def test_vllm_paged_cache_writes_compressed_slots_without_dense_cache():
    config = _config(backend="vllm")
    spec = PagedCacheSpec(block_size=4, num_kv_heads=2, head_dim=128, k_format="K16", v_format="V4")
    cache = PagedCompressedKVCache(config=config, spec=spec, device=torch.device("cpu"))
    keys = torch.randn(3, 2, 128, dtype=torch.float16)
    values = torch.randn(3, 2, 128, dtype=torch.float16)

    cache.write(keys, values, torch.tensor([0, 5, 7]))
    report = cache.to_report()

    assert report["cache"]["slots"] == 3
    assert report["cache"]["slot_ids"] == [0, 5, 7]
    with pytest.raises(KeyError):
        cache.slot(1)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels are unavailable")
def test_vllm_paged_cache_decode_matches_reference_on_compressed_slots():
    device = torch.device("cuda")
    config = _config(backend="vllm")
    spec = PagedCacheSpec(block_size=4, num_kv_heads=2, head_dim=128, k_format="K16", v_format="V4")
    cache = PagedCompressedKVCache(config=config, spec=spec, device=device)
    torch.manual_seed(7)
    keys = torch.randn(5, 2, 128, device=device, dtype=torch.float16)
    values = torch.randn(5, 2, 128, device=device, dtype=torch.float16)
    query = torch.randn(1, 16, 128, device=device, dtype=torch.float16)
    slots = torch.tensor([0, 2, 4, 6, 8], device=device)
    cache.write(keys, values, slots)

    counters = KernelCounters()
    result = cache.decode(query, slots, num_q_heads=16, scale=1.0 / math.sqrt(128), counters=counters)
    ref_cache = CompressedKVCache(
        head_dim=128,
        num_kv_heads=2,
        layer_idx=0,
        policy=policy_from_config(config),
        device=device,
        dtype=torch.float16,
    )
    ref_cache.prefill(keys, values)
    ref_cache.flush_recent()
    ref = direct_qk_attention(
        query.float(),
        ref_cache,
        scale=1.0 / math.sqrt(128),
    )

    assert result.report.name == "decode_k16v4"
    assert counters.score_calls == 1
    assert counters.v_accumulation_calls == 1
    torch.testing.assert_close(result.output.float(), ref.output.float(), atol=2e-4, rtol=2e-4)

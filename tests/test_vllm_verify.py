from __future__ import annotations

import pytest

from torbuquant.integration.vllm import (
    build_metadata,
    detect_model_cache_shape,
    format_verify_report,
    parse_verify_args,
    save_metadata,
    verify_metadata_only,
    version_report,
)


class _Config:
    model_type = "qwen2"
    hidden_size = 128
    num_attention_heads = 4
    num_key_value_heads = 2
    num_hidden_layers = 3
    num_kv_shared_layers = 1
    layer_types = ("full_attention", "sliding_attention", "full_attention")


class _Nested:
    model_type = "molmo2"
    text_config = _Config()


def test_detect_model_cache_shape_with_nested_text_config():
    shape = detect_model_cache_shape(_Nested())

    assert shape.model_type == "molmo2"
    assert shape.head_dim == 32
    assert shape.num_q_heads == 4
    assert shape.num_kv_heads == 2
    assert shape.unique_cache_layers == 2
    assert shape.layer_types == ("full_attention", "sliding_attention", "full_attention")


def test_parse_verify_args_rejects_mixed_bits():
    with pytest.raises(SystemExit):
        parse_verify_args(["--model", "m", "--bits", "4", "--k-bits", "4", "--v-bits", "2"])
    with pytest.raises(SystemExit):
        parse_verify_args(["--model", "m", "--k-bits", "4"])


def test_parse_verify_args_requires_metadata_for_recipe():
    with pytest.raises(SystemExit):
        parse_verify_args(["--model", "m", "--kv-cache-dtype", "turboquant35"])


def test_metadata_only_verification(tmp_path):
    metadata = build_metadata(
        recipe="turboquant25",
        head_dim=128,
        num_kv_heads=2,
        layer_names=["layer0"],
    )
    path = tmp_path / "turboquant_kv.json"
    save_metadata(metadata, path)

    result = verify_metadata_only(
        model="qwen",
        kv_cache_dtype="turboquant25",
        metadata_path=str(path),
        head_dim=128,
    )

    assert result["status"] == "PASS"
    assert result["metadata"]["layers"] == 1


def test_verify_report_and_versions_have_expected_fields():
    report = format_verify_report(
        {
            "model": "qwen",
            "status": "PASS",
            "family": "Qwen",
            "min_cosine": 0.99,
            "threshold": 0.98,
            "compression": {"compressed_bytes": 10, "baseline_bytes": 20},
        }
    )
    versions = version_report()

    assert "Model: qwen" in report
    assert "Compressed bytes: 10" in report
    assert "torch" in versions

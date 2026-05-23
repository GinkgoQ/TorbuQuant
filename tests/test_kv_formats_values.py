import pytest
import torch

from torbuquant.kv import (
    CacheGeometry,
    PackedKVLayout,
    dequantize_values,
    estimate_persistent_bytes,
    quantize_values,
    validate_k_format,
    validate_v_format,
    value_data_nbytes,
    value_formula_nbytes,
)


@pytest.mark.parametrize("bits", [8, 4, 2])
def test_value_quantization_round_trip_shape_and_bytes(bits):
    torch.manual_seed(bits)
    values = torch.randn(2, 3, 17)

    q = quantize_values(values, bits=bits, group_size=8)
    out = dequantize_values(q)
    byte_counts = value_data_nbytes(q)
    formula = value_formula_nbytes(values.shape, bits=bits, group_size=8)

    assert out.shape == values.shape
    assert q.data.dtype == torch.uint8
    assert byte_counts["compressed_v_bytes"] == formula["compressed_v_bytes"]
    assert byte_counts["scales_bytes"] == formula["scales_bytes"]
    assert byte_counts["zeros_bytes"] == formula["zeros_bytes"]
    assert formula["padded_dim"] == 24


def test_v3_requires_gate_and_round_trips_when_enabled():
    values = torch.randn(4, 13)

    with pytest.raises(ValueError, match="V3 requires"):
        quantize_values(values, bits=3, group_size=8)

    q = quantize_values(values, bits=3, group_size=8, allow_v3=True)
    out = dequantize_values(q)

    assert out.shape == values.shape
    assert q.data.shape[-1] == 6


def test_format_gates_for_low_bit_keys():
    with pytest.raises(ValueError, match="requires"):
        validate_k_format("K3")
    assert validate_k_format("K3", allow_low_bit_keys=True).bits == 3
    assert validate_v_format("V4").bits == 4


def test_packed_layout_counts_padding_and_metadata():
    layout = PackedKVLayout("K4", "V4", head_dim=17, value_group_size=8, include_qjl=True)

    assert layout.k_payload_bytes_per_vector() == 9
    assert layout.k_norm_bytes_per_vector() == 4
    assert layout.v_payload_bytes_per_vector() == 12
    assert layout.v_metadata_bytes_per_vector() == 12
    assert layout.bytes_per_vector_pair() == 37
    assert layout.to_dict()["bit_order"] == "little"


def test_value_padding_does_not_change_group_minmax():
    values = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0]])

    q = quantize_values(values, bits=4, group_size=4)
    out = dequantize_values(q)

    assert q.scales.shape[-1] == 2
    torch.testing.assert_close(out, values, rtol=1e-3, atol=1e-3)


def test_k8_layout_counts_key_scale_zero_metadata():
    geometry = CacheGeometry(batch=1, num_kv_heads=2, tokens=3, head_dim=17)
    layout = PackedKVLayout("K8", "V4", head_dim=17, value_group_size=8)
    estimate = estimate_persistent_bytes(geometry, layout)

    assert layout.k_metadata_bytes_per_vector() == 4
    assert estimate["scales_bytes"] == geometry.vectors * (layout.value_groups + 1) * 2
    assert estimate["zeros_bytes"] == geometry.vectors * (layout.value_groups + 1) * 2

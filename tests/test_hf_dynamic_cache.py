from __future__ import annotations

import pytest
import torch

from torbuquant.integration.hf import CompressedDynamicCache


class _Layer:
    def __init__(self):
        self.keys = None
        self.values = None
        self.is_initialized = False

    def lazy_initialization(self, *_args):
        self.is_initialized = True


class _Cache:
    def __init__(self, layers: int = 2):
        self.layers = [_Layer() for _ in range(layers)]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        del cache_kwargs
        layer = self.layers[layer_idx]
        if layer.keys is None:
            layer.keys = key_states
            layer.values = value_states
        else:
            layer.keys = torch.cat([layer.keys, key_states], dim=-2)
            layer.values = torch.cat([layer.values, value_states], dim=-2)
        return layer.keys, layer.values

    def get_seq_length(self, layer_idx=0):
        layer = self.layers[layer_idx]
        if layer.keys is None:
            return 0
        return int(layer.keys.shape[-2])


class _Config:
    layer_types = ("sliding_attention", "full_attention")


def test_dynamic_cache_stores_compressed_rows_and_reports_bytes():
    cache = _Cache()
    wrapper = CompressedDynamicCache(cache, head_dim=16, bits=4, device=torch.device("cpu"))
    key = torch.randn(1, 2, 3, 16, dtype=torch.float16)
    value = torch.randn(1, 2, 3, 16, dtype=torch.float16)

    out_key, out_value = cache.update(key, value, 0)
    cache.update(key[:, :, :1], value[:, :, :1], 0)
    packed = wrapper.get_compressed(0)
    stats = wrapper.compression_stats()

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert packed[0].dtype == torch.uint8
    assert cache.get_seq_length(0) == 4
    assert stats["compressed_bytes"] < stats["baseline_bytes"]
    assert stats["context_bytes"]["4096"] > 0


def test_dynamic_cache_disable_enable_and_restore():
    cache = _Cache()
    original = cache.update
    wrapper = CompressedDynamicCache(cache, head_dim=16, bits=4, device=torch.device("cpu"))
    key = torch.randn(1, 1, 2, 16, dtype=torch.float16)
    value = torch.randn(1, 1, 2, 16, dtype=torch.float16)

    wrapper.disable()
    cache.update(key, value, 0)
    assert wrapper.vram_bytes() == 0
    wrapper.enable()
    cache.update(key, value, 0)
    assert wrapper.vram_bytes() > 0
    wrapper.restore()

    assert cache.update == original


def test_dynamic_cache_context_restores_on_exception():
    cache = _Cache()
    original = cache.update

    with pytest.raises(RuntimeError):
        with CompressedDynamicCache(cache, head_dim=16, bits=4, device=torch.device("cpu")):
            raise RuntimeError("boom")

    assert cache.update == original


def test_dynamic_cache_asymmetric_bits_and_bypass_layer():
    cache = _Cache()
    wrapper = CompressedDynamicCache(
        cache,
        head_dim=16,
        k_bits=4,
        v_bits=2,
        bits=None,
        device=torch.device("cpu"),
        model_config=_Config(),
    )
    key = torch.randn(1, 1, 2, 16, dtype=torch.float16)
    value = torch.randn(1, 1, 2, 16, dtype=torch.float16)

    cache.update(key, value, 0)
    cache.update(key, value, 1)

    assert wrapper.k_bits == 4
    assert wrapper.v_bits == 2
    assert wrapper.get_compressed(0)[0].numel() > wrapper.get_compressed(0)[2].numel()
    assert cache.get_seq_length(1) == 2
    with pytest.raises(ValueError, match="no compressed"):
        wrapper.get_compressed(1)


def test_dynamic_cache_warns_on_double_wrap():
    cache = _Cache()
    CompressedDynamicCache(cache, head_dim=16, bits=4, device=torch.device("cpu"))

    with pytest.warns(UserWarning):
        CompressedDynamicCache(cache, head_dim=16, bits=4, device=torch.device("cpu"))

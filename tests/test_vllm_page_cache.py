from __future__ import annotations

import pytest
import torch

from torbuquant.integration.vllm.page_cache import (
    PackedPageCache,
    PageGeometry,
    make_empty_rows,
    split_tq_row,
)


def test_page_geometry_counts_tq4_bytes():
    geometry = PageGeometry(
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=128,
        k_bits=4,
        v_bits=2,
    )

    assert geometry.k_row_bytes == 68
    assert geometry.v_row_bytes == 36
    assert geometry.row_bytes == 104
    assert geometry.cache_shape() == (2, 4, 2, 104)
    assert geometry.packed_kv_bytes < geometry.dense_kv_bytes


def test_page_cache_writes_slots_and_ignores_negative_slots():
    geometry = PageGeometry(num_blocks=2, block_size=4, num_kv_heads=2, head_dim=64)
    cache = PackedPageCache(geometry=geometry, device=torch.device("cpu"))
    key_rows = torch.arange(3 * 2 * geometry.k_row_bytes, dtype=torch.uint8).reshape(
        3,
        2,
        geometry.k_row_bytes,
    )
    value_rows = torch.arange(3 * 2 * geometry.v_row_bytes, dtype=torch.uint8).reshape(
        3,
        2,
        geometry.v_row_bytes,
    )

    cache.write_rows(key_rows, value_rows, torch.tensor([0, -1, 5]))

    k0, v0 = cache.read_slot(0)
    k5, v5 = cache.read_slot(5)
    assert cache.writes == 2
    torch.testing.assert_close(k0, key_rows[0])
    torch.testing.assert_close(v0, value_rows[0])
    torch.testing.assert_close(k5, key_rows[2])
    torch.testing.assert_close(v5, value_rows[2])


def test_page_cache_slot_overwrite():
    geometry = PageGeometry(num_blocks=1, block_size=4, num_kv_heads=1, head_dim=64)
    cache = PackedPageCache(geometry=geometry, device=torch.device("cpu"))
    first_key = torch.ones(1, 1, geometry.k_row_bytes, dtype=torch.uint8)
    first_value = torch.ones(1, 1, geometry.v_row_bytes, dtype=torch.uint8) * 2
    second_key = torch.ones(1, 1, geometry.k_row_bytes, dtype=torch.uint8) * 3
    second_value = torch.ones(1, 1, geometry.v_row_bytes, dtype=torch.uint8) * 4

    cache.write_rows(first_key, first_value, torch.tensor([2]))
    cache.write_rows(second_key, second_value, torch.tensor([2]))

    key, value = cache.read_slot(2)
    assert cache.writes == 2
    torch.testing.assert_close(key, second_key[0])
    torch.testing.assert_close(value, second_value[0])


def test_page_cache_rejects_out_of_range_slot():
    geometry = PageGeometry(num_blocks=1, block_size=4, num_kv_heads=1, head_dim=64)
    cache = PackedPageCache(geometry=geometry, device=torch.device("cpu"))
    key = torch.zeros(1, 1, geometry.k_row_bytes, dtype=torch.uint8)
    value = torch.zeros(1, 1, geometry.v_row_bytes, dtype=torch.uint8)

    with pytest.raises(IndexError, match="exceeds"):
        cache.write_rows(key, value, torch.tensor([4]))


def test_empty_rows_and_split_tq_row():
    row = torch.arange(104, dtype=torch.uint8)
    key, value = split_tq_row(row, head_dim=128, k_bits=4, v_bits=2)
    empty = make_empty_rows(tokens=2, num_kv_heads=3, row_bytes=5, device=torch.device("cpu"))

    assert key.numel() == 68
    assert value.numel() == 36
    assert empty.shape == (2, 3, 5)

from __future__ import annotations

import math

import torch

from torbuquant.core.codebook import get_codebook_tensors
from torbuquant.core.rotation import RotationMode, build_rotation
from torbuquant.triton import decode_tq4_paged, tq_row_bytes, unpack_tq_rows, write_tq4_kv


def test_decode_tq4_paged_matches_unpacked_attention():
    torch.manual_seed(9)
    dim = 16
    kv_heads = 2
    q_heads = 4
    tokens = 5
    block_size = 4
    rotation = build_rotation(dim, RotationMode.RHT, torch.device("cpu"), seed=23)
    centroids, _ = get_codebook_tensors(dim, 4, torch.device("cpu"))
    keys = torch.randn(tokens, kv_heads, dim)
    values = torch.randn(tokens, kv_heads, dim)
    row_bytes = tq_row_bytes(dim, 4)
    key_pages = torch.zeros(2, block_size, kv_heads, row_bytes, dtype=torch.uint8)
    value_pages = torch.zeros_like(key_pages)
    slots = torch.arange(tokens)
    write_tq4_kv(keys, key_pages, slots, rotation=rotation, centroids=centroids)
    write_tq4_kv(values, value_pages, slots, rotation=rotation, centroids=centroids)
    query = torch.randn(1, q_heads, dim)
    block_table = torch.tensor([[0, 1]], dtype=torch.int64)
    seq_lens = torch.tensor([tokens], dtype=torch.int64)

    out = decode_tq4_paged(
        query,
        key_pages,
        value_pages,
        block_table,
        seq_lens,
        rotation=rotation,
        centroids=centroids,
        num_kv_heads=kv_heads,
    )
    key_rows = torch.stack([key_pages[int(i // block_size), int(i % block_size)] for i in range(tokens)])
    value_rows = torch.stack([value_pages[int(i // block_size), int(i % block_size)] for i in range(tokens)])
    decoded_keys = unpack_tq_rows(
        key_rows,
        rotation=rotation,
        centroids=centroids,
        bit_width=4,
        head_dim=dim,
    )
    decoded_values = unpack_tq_rows(
        value_rows,
        rotation=rotation,
        centroids=centroids,
        bit_width=4,
        head_dim=dim,
    )
    expected = torch.empty_like(query)
    for q_head in range(q_heads):
        kv_head = q_head // (q_heads // kv_heads)
        scores = torch.matmul(decoded_keys[:, kv_head], query[0, q_head]) / math.sqrt(dim)
        expected[0, q_head] = torch.matmul(torch.softmax(scores, dim=-1), decoded_values[:, kv_head])

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_decode_tq4_paged_sliding_window_changes_context():
    torch.manual_seed(13)
    dim = 16
    rotation = build_rotation(dim, RotationMode.RHT, torch.device("cpu"), seed=29)
    centroids, _ = get_codebook_tensors(dim, 4, torch.device("cpu"))
    keys = torch.randn(6, 1, dim)
    values = torch.randn(6, 1, dim)
    row_bytes = tq_row_bytes(dim, 4)
    key_pages = torch.zeros(2, 4, 1, row_bytes, dtype=torch.uint8)
    value_pages = torch.zeros_like(key_pages)
    write_tq4_kv(keys, key_pages, torch.arange(6), rotation=rotation, centroids=centroids)
    write_tq4_kv(values, value_pages, torch.arange(6), rotation=rotation, centroids=centroids)
    query = torch.randn(1, 1, dim)
    block_table = torch.tensor([[0, 1]], dtype=torch.int64)
    seq_lens = torch.tensor([6], dtype=torch.int64)

    all_out = decode_tq4_paged(
        query,
        key_pages,
        value_pages,
        block_table,
        seq_lens,
        rotation=rotation,
        centroids=centroids,
        num_kv_heads=1,
    )
    window_out = decode_tq4_paged(
        query,
        key_pages,
        value_pages,
        block_table,
        seq_lens,
        rotation=rotation,
        centroids=centroids,
        num_kv_heads=1,
        sliding_window=2,
    )

    assert not torch.allclose(all_out, window_out)

from __future__ import annotations

import torch

from torbuquant.core.codebook import get_codebook_tensors
from torbuquant.core.rotation import RotationMode, build_rotation
from torbuquant.triton import pack_tq_rows, tq_row_bytes, unpack_tq_rows, write_tq4_kv


def test_pack_tq_rows_and_slot_write_roundtrip():
    torch.manual_seed(5)
    dim = 16
    rotation = build_rotation(dim, RotationMode.RHT, torch.device("cpu"), seed=17)
    centroids, _ = get_codebook_tensors(dim, 4, torch.device("cpu"))
    raw = torch.randn(3, 2, dim)
    row_bytes = tq_row_bytes(dim, 4)
    pages = torch.zeros(2, 4, 2, row_bytes, dtype=torch.uint8)

    rows = write_tq4_kv(
        raw,
        pages,
        torch.tensor([0, -1, 5]),
        rotation=rotation,
        centroids=centroids,
        bit_width=4,
    )
    unpacked = unpack_tq_rows(
        rows,
        rotation=rotation,
        centroids=centroids,
        bit_width=4,
        head_dim=dim,
    )

    assert rows.shape == (3, 2, row_bytes)
    torch.testing.assert_close(pages[0, 0], rows[0])
    torch.testing.assert_close(pages[1, 1], rows[2])
    assert torch.count_nonzero(pages[0, 1]) == 0
    assert unpacked.shape == raw.shape
    assert torch.isfinite(unpacked).all()


def test_pack_tq_rows_matches_direct_writer():
    torch.manual_seed(7)
    dim = 32
    rotation = build_rotation(dim, RotationMode.RHT, torch.device("cpu"), seed=19)
    centroids, _ = get_codebook_tensors(dim, 3, torch.device("cpu"))
    raw = torch.randn(2, 1, dim)
    rows = pack_tq_rows(raw, rotation=rotation, centroids=centroids, bit_width=3)

    assert rows.shape[-1] == tq_row_bytes(dim, 3)
    assert rows.dtype == torch.uint8

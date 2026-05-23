import pytest
import torch

from torbuquant.packing import (
    pack_k2,
    pack_k3,
    pack_k4,
    pack_sign_bits,
    pack_unsigned,
    pack_v2,
    pack_v3,
    pack_v4,
    packed_length,
    packed_nbytes,
    unpack_k2,
    unpack_k3,
    unpack_k4,
    unpack_sign_bits,
    unpack_unsigned,
    unpack_v2,
    unpack_v3,
    unpack_v4,
)


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("width", [1, 2, 7, 8, 9, 17])
def test_pack_unsigned_round_trip(bits, width):
    torch.manual_seed(bits * 100 + width)
    values = torch.randint(0, 1 << bits, (3, 4, width), dtype=torch.int64)

    packed = pack_unsigned(values, bits)
    out = unpack_unsigned(packed, bits, width)

    assert packed.shape == (3, 4, packed_length(width, bits))
    assert packed.dtype == torch.uint8
    torch.testing.assert_close(out, values)


def test_named_kv_packers_round_trip():
    k = torch.tensor([[0, 1, 2, 15, 7, 8, 3]], dtype=torch.int64)
    k3 = torch.tensor([[0, 1, 7, 2, 4, 5, 6, 3, 1]], dtype=torch.int64)
    k2 = torch.tensor([[0, 1, 2, 3, 0, 2, 1]], dtype=torch.int64)
    v4 = torch.tensor([[0, 4, 15, 9, 1]], dtype=torch.int64)
    v2 = torch.tensor([[0, 1, 2, 3, 0, 2]], dtype=torch.int64)
    v3 = torch.tensor([[0, 1, 7, 6, 2, 3, 4]], dtype=torch.int64)

    torch.testing.assert_close(unpack_k4(pack_k4(k), k.shape[-1]), k)
    torch.testing.assert_close(unpack_k3(pack_k3(k3), k3.shape[-1]), k3)
    torch.testing.assert_close(unpack_k2(pack_k2(k2), k2.shape[-1]), k2)
    torch.testing.assert_close(unpack_v4(pack_v4(v4), v4.shape[-1]), v4)
    torch.testing.assert_close(unpack_v2(pack_v2(v2), v2.shape[-1]), v2)
    torch.testing.assert_close(unpack_v3(pack_v3(v3), v3.shape[-1]), v3)


def test_packed_nbytes_matches_packed_tensor_storage():
    values = torch.arange(2 * 3 * 13, dtype=torch.int64).reshape(2, 3, 13) % 8
    packed = pack_unsigned(values, 3)

    assert packed.numel() == packed_nbytes(values.shape, 3)


def test_pack_unsigned_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="values must be"):
        pack_unsigned(torch.tensor([4]), bits=2)


def test_sign_bit_packing_round_trip_bool():
    signs = torch.tensor([[True, False, True, True, False, False, True, False, True]])

    packed = pack_sign_bits(signs)
    out = unpack_sign_bits(packed, signs.shape[-1])

    assert packed.dtype == torch.uint8
    torch.testing.assert_close(out, signs)

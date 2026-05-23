import torch

from torbuquant.core.mse import TorbuquantMSE, pack_indices, unpack_indices
from torbuquant.core.rotation import RotationMode, build_rotation
from torbuquant.packing import packed_length


def test_pack_indices_round_trip_for_supported_bit_widths():
    torch.manual_seed(0)
    for bits in (1, 2, 3, 4, 8):
        high = 1 << bits
        x = torch.randint(0, high, (3, 5, 17), dtype=torch.long)

        packed = pack_indices(x, bits)
        out = unpack_indices(packed, bits, d=17)

        torch.testing.assert_close(out, x)
        assert packed.dtype == torch.uint8


def test_mse_pack_indices_uses_true_three_bit_layout():
    x = torch.randint(0, 8, (2, 17), dtype=torch.long)
    packed = pack_indices(x, 3)

    assert packed.shape[-1] == packed_length(17, 3)
    torch.testing.assert_close(unpack_indices(packed, 3, 17), x)


def test_mse_quantizer_round_trip_shape_and_dtype():
    torch.manual_seed(1)
    device = torch.device("cpu")
    rotation = build_rotation(8, RotationMode.QR, device=device, dtype=torch.float32, seed=11)
    quantizer = TorbuquantMSE(8, 2, rotation, device=device)
    x = torch.randn(4, 6, 8)

    q = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q)

    assert q.indices.dtype == torch.uint8
    assert q.norms.dtype == torch.float16
    assert x_hat.shape == x.shape
    assert x_hat.dtype == torch.float32


def test_mse_distortion_decreases_with_bit_width():
    torch.manual_seed(2)
    device = torch.device("cpu")
    x = torch.randn(48, 8)
    rotation = build_rotation(8, RotationMode.QR, device=device, dtype=torch.float32, seed=22)

    q1 = TorbuquantMSE(8, 1, rotation, device=device)
    q3 = TorbuquantMSE(8, 3, rotation, device=device)

    mse1 = ((q1(x) - x) ** 2).mean()
    mse3 = ((q3(x) - x) ** 2).mean()

    assert mse3 < mse1


def test_mse_norm_correction_restores_stored_norms():
    torch.manual_seed(4)
    device = torch.device("cpu")
    rotation = build_rotation(16, RotationMode.RHT, device=device, dtype=torch.float32, seed=41)
    quantizer = TorbuquantMSE(16, 2, rotation, device=device, norm_correction=True)
    x = torch.randn(8, 16)

    q = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q)

    torch.testing.assert_close(x_hat.norm(dim=-1), q.norms.float(), rtol=2e-3, atol=2e-3)


def test_mse_score_matches_dequantized_scores_for_query_batch():
    torch.manual_seed(5)
    device = torch.device("cpu")
    rotation = build_rotation(16, RotationMode.QR, device=device, dtype=torch.float32, seed=51)
    quantizer = TorbuquantMSE(16, 3, rotation, device=device, norm_correction=True)
    keys = torch.randn(2, 11, 16)
    query = torch.randn(2, 4, 16)
    scale = 0.25

    q = quantizer.quantize(keys)
    direct = quantizer.score(query, q, scale=scale)
    dense = torch.einsum("bqd,bnd->bqn", query.float(), quantizer.dequantize(q).float()) * scale

    torch.testing.assert_close(direct, dense, rtol=4e-3, atol=4e-3)


def test_mse_score_matches_dequantized_scores_for_single_query():
    torch.manual_seed(6)
    device = torch.device("cpu")
    rotation = build_rotation(32, RotationMode.RHT, device=device, dtype=torch.float32, seed=61)
    quantizer = TorbuquantMSE(32, 4, rotation, device=device, norm_correction=True)
    keys = torch.randn(3, 13, 32)
    query = torch.randn(3, 32)

    q = quantizer.quantize(keys)
    direct = quantizer.score(query, q)
    dense = torch.einsum("bd,bnd->bn", query.float(), quantizer.dequantize(q).float())

    torch.testing.assert_close(direct, dense, rtol=4e-3, atol=4e-3)

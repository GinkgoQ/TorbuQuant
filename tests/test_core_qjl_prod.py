import torch

from torbuquant.core.types import MSEData
from torbuquant.core.polar import TorbuquantProd
from torbuquant.core.qjl import pack_signs, unpack_signs
from torbuquant.core.rotation import RotationMode, build_rotation


def test_qjl_sign_packing_round_trip():
    projected = torch.tensor(
        [[-3.0, -0.2, 0.1, 4.0, 8.0, -1.0, 0.5, -0.7, 9.0]],
        dtype=torch.float32,
    )
    packed = pack_signs(projected)
    signs = unpack_signs(packed, d=projected.shape[-1])
    expected = torch.where(projected > 0, torch.ones_like(projected), -torch.ones_like(projected))

    assert packed.dtype == torch.uint8
    torch.testing.assert_close(signs, expected)


def test_prod_attention_score_matches_dequantized_dot_product():
    torch.manual_seed(3)
    device = torch.device("cpu")
    dim = 8
    rotation = build_rotation(dim, RotationMode.QR, device=device, dtype=torch.float32, seed=33)
    quantizer = TorbuquantProd(dim, 3, rotation, device=device, qjl_seed=44)

    keys = torch.randn(2, 5, dim)
    query = torch.randn(2, dim)

    q = quantizer.quantize(keys)
    scores = quantizer.attention_score(query, q, include_qjl=True)

    keys_hat = quantizer.dequantize(q)
    reference = (query.unsqueeze(1) * keys_hat).sum(dim=-1)

    torch.testing.assert_close(scores, reference, rtol=2e-4, atol=2e-4)


def test_prod_attention_score_default_uses_mse_part_only():
    torch.manual_seed(5)
    device = torch.device("cpu")
    dim = 8
    rotation = build_rotation(dim, RotationMode.QR, device=device, dtype=torch.float32, seed=55)
    quantizer = TorbuquantProd(dim, 3, rotation, device=device, qjl_seed=56)

    keys = torch.randn(2, 5, dim)
    query = torch.randn(2, dim)

    q = quantizer.quantize(keys)
    scores = quantizer.attention_score(query, q)
    mse_data = MSEData(q.mse_indices, q.norms, q.mse_bits, q.dim)
    keys_mse = quantizer.mse.dequantize(mse_data)
    reference = (query.unsqueeze(1) * keys_mse).sum(dim=-1)

    assert quantizer.qjl_for_attention is False
    torch.testing.assert_close(scores, reference, rtol=2e-4, atol=2e-4)


def test_reusing_transform_across_chunks_gives_same_scores():
    torch.manual_seed(6)
    device = torch.device("cpu")
    dim = 16
    rotation = build_rotation(dim, RotationMode.RHT, device=device, dtype=torch.float32, seed=66)
    quantizer = TorbuquantProd(dim, 4, rotation, device=device, qjl_seed=67)

    keys = torch.randn(3, 9, dim)
    query = torch.randn(3, dim)
    all_q = quantizer.quantize(keys)
    q0 = quantizer.quantize(keys[:, :4])
    q1 = quantizer.quantize(keys[:, 4:])

    score_all = quantizer.attention_score(query, all_q, include_qjl=True)
    score_chunks = torch.cat(
        [
            quantizer.attention_score(query, q0, include_qjl=True),
            quantizer.attention_score(query, q1, include_qjl=True),
        ],
        dim=1,
    )

    torch.testing.assert_close(score_chunks, score_all, rtol=0, atol=0)

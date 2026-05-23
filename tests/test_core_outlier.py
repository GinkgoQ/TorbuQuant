import torch

from torbuquant.core.outlier import TorbuquantChannelMSE, channel_energy_scores, split_channel_indices


def test_split_channel_indices_fractional_bits_without_calibration():
    high, low, high_bits, low_bits = split_channel_indices(128, 3.5, device=torch.device("cpu"))

    assert high_bits == 4
    assert low_bits == 3
    assert high.numel() == 64
    assert low.numel() == 64
    assert torch.equal(high, torch.arange(64))
    assert torch.equal(low, torch.arange(64, 128))


def test_split_channel_indices_uses_calibration_scores():
    scores = torch.arange(8, dtype=torch.float32)
    high, low, high_bits, low_bits = split_channel_indices(8, 2.25, calibration_scores=scores)

    assert high_bits == 3
    assert low_bits == 2
    assert torch.equal(high, torch.tensor([6, 7]))
    assert torch.equal(low, torch.tensor([0, 1, 2, 3, 4, 5]))


def test_channel_energy_scores_matches_mean_square():
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    scores = channel_energy_scores(x)

    torch.testing.assert_close(scores, torch.tensor([5.0, 4.0, 5.0]))


def test_channel_mse_round_trip_and_storage_ratio():
    torch.manual_seed(7)
    x = torch.randn(5, 128)
    scores = channel_energy_scores(x)
    codec = TorbuquantChannelMSE(
        dim=128,
        target_bits=3.5,
        device=torch.device("cpu"),
        seed=11,
        calibration_scores=scores,
        rotation_mode="rht",
    )

    q = codec.quantize(x)
    y = codec.dequantize(q)
    err = (x - y).float().pow(2).mean().sqrt()

    assert y.shape == x.shape
    assert codec.effective_bits == 3.5
    assert codec.storage_bits_per_vector() == 128 * 3 + 64 + 32
    assert codec.compression_ratio() > 4.0
    assert torch.isfinite(err)
    assert err < x.float().norm(dim=-1).mean()


def test_channel_mse_integer_target_uses_single_low_codec():
    x = torch.randn(2, 16)
    codec = TorbuquantChannelMSE(
        dim=16,
        target_bits=4.0,
        device=torch.device("cpu"),
        seed=3,
        rotation_mode="rht",
    )

    q = codec.quantize(x)
    y = codec.dequantize(q)

    assert q.high is None
    assert q.low is not None
    assert codec.high_index.numel() == 0
    assert codec.low_index.numel() == 16
    assert y.shape == x.shape

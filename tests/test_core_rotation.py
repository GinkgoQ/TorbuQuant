import pytest
import torch

from torbuquant.core.rotation import (
    RotationMode,
    build_rotation,
    derive_transform_seed,
    rotation_from_spec,
    rotate_backward,
    rotate_forward,
)


def _assert_round_trip(mode: RotationMode):
    device = torch.device("cpu")
    state = build_rotation(16, mode, device=device, dtype=torch.float32, seed=123)
    x = torch.randn(5, 7, 16)

    y = rotate_forward(x, state)
    x_back = rotate_backward(y, state)

    torch.testing.assert_close(x_back, x.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_qr_rotation_round_trip_and_norms():
    _assert_round_trip(RotationMode.QR)


def test_rht_rotation_round_trip_and_norms():
    _assert_round_trip(RotationMode.RHT)


def test_qr_matrix_is_orthogonal():
    device = torch.device("cpu")
    state = build_rotation(16, RotationMode.QR, device=device, dtype=torch.float32, seed=7)
    eye = torch.eye(16)

    torch.testing.assert_close(state.matrix.T @ state.matrix, eye, rtol=1e-5, atol=1e-5)


def test_rht_requires_power_of_two_dimension():
    with pytest.raises(ValueError, match="power-of-two"):
        build_rotation(12, RotationMode.RHT, device=torch.device("cpu"), seed=1)


def test_rotation_metadata_rebuilds_same_transform():
    device = torch.device("cpu")
    seed = derive_transform_seed(19, layer_idx=3, head_idx=2)
    state = build_rotation(16, RotationMode.RHT, device=device, seed=seed)
    rebuilt = rotation_from_spec(state.spec, device=device)

    x = torch.randn(4, 16)
    torch.testing.assert_close(
        rotate_forward(x, state),
        rotate_forward(x, rebuilt),
        rtol=0,
        atol=0,
    )
    assert state.to_metadata() == rebuilt.to_metadata()

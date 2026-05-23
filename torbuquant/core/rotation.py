"""Rotation helpers for TurboQuant core codecs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn.functional as F

from torbuquant.core.types import TransformSpec


# Enums and state.

class RotationMode(Enum):
    QR = auto()
    RHT = auto()


@dataclass
class RotationState:
    """Parameters needed to apply and invert a layer rotation."""

    mode: RotationMode
    d_orig: int
    seed: int
    matrix: Optional[torch.Tensor] = field(default=None)
    signs1: Optional[torch.Tensor] = field(default=None)
    signs2: Optional[torch.Tensor] = field(default=None)
    d_pad: int = 0

    @property
    def spec(self) -> TransformSpec:
        kind = "qr_rotation" if self.mode == RotationMode.QR else "rht"
        tensor = self.matrix if self.matrix is not None else self.signs1
        dtype = str(tensor.dtype).replace("torch.", "") if tensor is not None else "float32"
        device_type = tensor.device.type if tensor is not None else "cpu"
        return TransformSpec(
            kind=kind,
            dim=self.d_orig,
            seed=self.seed,
            dtype=dtype,
            device_type=device_type,
            pad_dim=self.d_pad,
        )

    def to_metadata(self) -> dict[str, object]:
        return self.spec.to_dict()


# Helpers.

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _generate_qr_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Haar-style random orthogonal matrix via QR of a Gaussian matrix."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(G)
    diag_signs = torch.sign(torch.diag(R))
    diag_signs[diag_signs == 0] = 1.0
    Q = Q * diag_signs.unsqueeze(0)
    sign, _ = torch.linalg.slogdet(Q)
    if sign < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(device=device, dtype=dtype)


def _generate_rht_signs(
    d: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Generate two sign vectors for the RHT rotation."""
    d_pad = _next_pow2(d)
    if d_pad != d:
        raise ValueError(
            "RHT currently requires a power-of-two dimension. "
            "Use QR for non-power-of-two dimensions."
        )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    s1 = torch.randint(0, 2, (d_pad,), generator=gen).float() * 2.0 - 1.0
    s2 = torch.randint(0, 2, (d_pad,), generator=gen).float() * 2.0 - 1.0
    return s1.to(device=device, dtype=dtype), s2.to(device=device, dtype=dtype), d_pad


# Walsh-Hadamard kernel.

def _walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform without materializing H."""
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"WHT requires power-of-2 length, got {n}"
    h = 1
    out = x.clone()
    while h < n:
        shape = out.shape[:-1] + (n // (2 * h), 2, h)
        v = out.reshape(shape)
        a = v[..., 0, :].clone()
        b = v[..., 1, :].clone()
        v[..., 0, :] = a + b
        v[..., 1, :] = a - b
        out = v.reshape(out.shape[:-1] + (n,))
        h *= 2
    return out * (1.0 / math.sqrt(n))


# Public factory.

def build_rotation(
    d: int,
    mode: RotationMode,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> RotationState:
    """Create a rotation state for a layer/head group."""
    if mode == RotationMode.QR:
        mat = _generate_qr_matrix(d, device, dtype, seed)
        return RotationState(mode=mode, d_orig=d, seed=seed, matrix=mat, d_pad=d)
    else:  # RHT
        s1, s2, d_pad = _generate_rht_signs(d, device, dtype, seed)
        return RotationState(mode=mode, d_orig=d, seed=seed, signs1=s1, signs2=s2, d_pad=d_pad)


def derive_transform_seed(base_seed: int, layer_idx: int, head_idx: int = 0) -> int:
    """Derive a deterministic transform seed for layer/head metadata."""
    return int(base_seed + 1_000_003 * layer_idx + 97_003 * head_idx)


def rotation_from_spec(
    spec: TransformSpec,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> RotationState:
    """Rebuild a rotation state from serialized metadata."""
    if spec.kind == "qr_rotation":
        mode = RotationMode.QR
    elif spec.kind == "rht":
        mode = RotationMode.RHT
    else:
        raise ValueError(f"Unsupported rotation spec kind: {spec.kind}")
    return build_rotation(spec.dim, mode, device=device, dtype=dtype, seed=spec.seed)


# Forward and backward transforms.

def rotate_forward(x: torch.Tensor, state: RotationState) -> torch.Tensor:
    """Apply the forward rotation to row-vector inputs."""
    x_f = x.float()

    if state.mode == RotationMode.QR:
        return torch.matmul(x_f, state.matrix.T)

    d_orig = state.d_orig
    d_pad  = state.d_pad
    if d_pad > d_orig:
        pad = d_pad - d_orig
        x_f = F.pad(x_f, (0, pad))
    x_f = x_f * state.signs1
    x_f = _walsh_hadamard_transform(x_f)
    x_f = x_f * state.signs2
    return x_f[..., :d_orig]


def rotate_backward(y: torch.Tensor, state: RotationState) -> torch.Tensor:
    """Apply the inverse rotation to row-vector inputs."""
    y_f = y.float()

    if state.mode == RotationMode.QR:
        return torch.matmul(y_f, state.matrix)

    d_orig = state.d_orig
    d_pad  = state.d_pad
    if d_pad > d_orig:
        pad = d_pad - d_orig
        y_f = F.pad(y_f, (0, pad))
    y_f = y_f * state.signs2
    y_f = _walsh_hadamard_transform(y_f)
    y_f = y_f * state.signs1
    return y_f[..., :d_orig]


# QJL projection matrix.

def build_qjl_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
) -> torch.Tensor:
    """Generate a QJL sketch matrix with i.i.d. standard normal entries."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    S = torch.randn(d, d, generator=gen, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)

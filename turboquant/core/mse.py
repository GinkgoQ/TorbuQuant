"""TurboQuant MSE quantizer, Algorithm 1 from the paper."""

from __future__ import annotations

from typing import Tuple

import torch

from turboquant.core.types import MSEData
from turboquant.core.rotation import RotationState, rotate_forward, rotate_backward
from turboquant.core.codebook import get_codebook_tensors
from turboquant.packing.bits import pack_unsigned, unpack_unsigned


# Bit packing.

def _packing_params(bits: int) -> Tuple[int, int]:
    """Return (eff_bits, vals_per_byte) for the given logical bit-width."""
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    return bits, 8 // bits if bits <= 4 else 1


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack integer indices into uint8 bytes."""
    _packing_params(bits)
    return pack_unsigned(indices, bits)


def unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Unpack uint8 bytes back to integer indices."""
    _packing_params(bits)
    return unpack_unsigned(packed, bits, d)


# MSE quantizer class.

class TorbuquantMSE(torch.nn.Module):
    """Algorithm 1 TurboQuant quantizer for reconstruction MSE."""

    def __init__(
        self,
        dim: int,
        bits: int,
        rotation: RotationState,
        device: torch.device,
        *,
        use_exact: bool = True,
        norm_correction: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.rotation = rotation
        self.use_exact = use_exact
        self.norm_correction = norm_correction

        centroids, decision_bnd = get_codebook_tensors(
            dim, bits, device, dtype=torch.float32, use_exact=use_exact
        )
        # Register as buffers so state_dict / .to() work correctly
        self.register_buffer("centroids", centroids)          # (2^bits,)
        self.register_buffer("decision_bnd", decision_bnd)   # (2^bits - 1,)

    # Quantize.

    def quantize(self, x: torch.Tensor) -> MSEData:
        """Quantize a batch of vectors."""
        norms = x.norm(dim=-1)

        x_unit = x / (norms.unsqueeze(-1).clamp(min=1e-12))

        y = rotate_forward(x_unit, self.rotation)

        y_c = y.contiguous()
        indices = torch.searchsorted(
            self.decision_bnd, y_c.reshape(-1, self.dim)
        ).reshape(y_c.shape)

        packed = pack_indices(indices, self.bits)

        return MSEData(
            indices=packed,
            norms=norms.half(),
            bits=self.bits,
            dim=self.dim,
        )

    # Dequantize.

    def dequantize(self, q: MSEData) -> torch.Tensor:
        """Reconstruct vectors from MSEData."""
        indices = unpack_indices(q.indices, q.bits, q.dim)

        y_hat = self.centroids[indices]

        x_hat = rotate_backward(y_hat, self.rotation)

        if self.norm_correction:
            x_hat = x_hat / x_hat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        x_hat = x_hat * q.norms.float().unsqueeze(-1)

        return x_hat

    def rotate_query(self, query: torch.Tensor) -> torch.Tensor:
        """Apply the quantizer rotation to query vectors."""
        if query.shape[-1] != self.dim:
            raise ValueError(f"query dim mismatch: expected {self.dim}, got {query.shape[-1]}")
        return rotate_forward(query, self.rotation)

    def score_rotated(self, query_rotated: torch.Tensor, q: MSEData, *, scale: float = 1.0) -> torch.Tensor:
        """Compute inner-product scores from rotated queries and packed indices.

        ``query_rotated`` may be shaped ``(..., dim)`` or ``(..., query_tokens,
        dim)``.  The packed data must be shaped ``(..., kv_tokens, packed_dim)``
        with matching leading dimensions.  The result has shape
        ``(..., kv_tokens)`` or ``(..., query_tokens, kv_tokens)``.
        """
        if q.dim != self.dim:
            raise ValueError(f"quantized dim mismatch: expected {self.dim}, got {q.dim}")
        if q.bits != self.bits:
            raise ValueError(f"quantized bit width mismatch: expected {self.bits}, got {q.bits}")
        if query_rotated.shape[-1] != self.dim:
            raise ValueError(f"query dim mismatch: expected {self.dim}, got {query_rotated.shape[-1]}")

        indices = unpack_indices(q.indices.to(query_rotated.device), q.bits, q.dim)
        values = self.centroids.to(query_rotated.device)[indices.long()].float()
        norms = q.norms.to(query_rotated.device).float()
        if self.norm_correction:
            unit_norm = values.norm(dim=-1).clamp(min=1e-12)
            norms = norms / unit_norm

        query_f = query_rotated.float()
        if query_f.ndim == values.ndim - 1:
            if query_f.shape[:-1] != values.shape[:-2]:
                raise ValueError("query and quantized prefixes do not match")
            scores = (query_f.unsqueeze(-2) * values).sum(dim=-1) * norms
            return scores * scale
        if query_f.ndim == values.ndim:
            if query_f.shape[:-2] != values.shape[:-2]:
                raise ValueError("query and quantized prefixes do not match")
            scores = torch.einsum("...qd,...nd->...qn", query_f, values)
            return scores * norms.unsqueeze(-2) * scale
        raise ValueError("query must be shaped (..., dim) or (..., query_tokens, dim)")

    def score(self, query: torch.Tensor, q: MSEData, *, scale: float = 1.0) -> torch.Tensor:
        """Compute inner-product scores from unrotated queries and packed indices."""
        return self.score_rotated(self.rotate_query(query), q, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))

"""
turboquant.core.qjl
===================
Quantized Johnson-Lindenstrauss (QJL) transform, with 1-bit residual correction.

Paper Section 2.2 (Definition 1 / Lemma 2):
  Given residual r = x - x_mse (x in S^{d-1}, x_mse from Algorithm 1),
  the QJL quantization is:

      z = sign(S * r) in {-1, +1}^d
      ||r||_2 stored as a scalar

  Dequantized QJL contribution:
      x_qjl = sqrt(pi/2) / d * ||r|| * S^T * z

  The combined estimator x_mse + x_qjl is unbiased:
      E[<y, x_mse + x_qjl>] = <y, x> for any y in R^d

  and has inner-product variance bounded by
      (pi/2) / d * ||y||^2 * D_mse(b-1).

Storage:
  Signs are packed 8 per byte (1 bit per coordinate) into uint8.
  One float32 scalar per vector stores ||r||.

Score computation (used during decode attention):
  Given pre-sketched query q_sk = q @ S^T:
      qjl_score = qjl_scale * ||r|| * <q_sk, z>
               = qjl_scale * ||r|| * sum_j q_sk[j] * z[j]

  where qjl_scale = sqrt(pi/2) / d.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# Sign packing.

def pack_signs(projected: torch.Tensor) -> torch.Tensor:
    """
    Convert real projections to packed sign bits (8 signs per byte).

    Args:
        projected : (..., d) float values from S * r.

    Returns:
        packed : (..., ceil(d/8))  uint8.
    """
    signs = (projected > 0).to(torch.uint8)   # 1 = positive, 0 = negative
    *batch, d = signs.shape
    pad = (-d) % 8
    if pad:
        signs = F.pad(signs, (0, pad), value=0)
    s = signs.reshape(*batch, -1, 8)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                          dtype=torch.uint8, device=projected.device)
    return (s * powers).sum(dim=-1, dtype=torch.uint8)  # (..., d//8)


def unpack_signs(packed: torch.Tensor, d: int) -> torch.Tensor:
    """
    Unpack sign bits back to float {-1, +1}.

    Args:
        packed : (..., ceil(d/8))  uint8.
        d      : original number of coordinates.

    Returns:
        signs : (..., d)  float32 with values in {-1, +1}.
    """
    *batch, _ = packed.shape
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                          dtype=torch.uint8, device=packed.device)
    bits = ((packed.unsqueeze(-1) & powers) > 0).reshape(*batch, -1)[..., :d]
    return bits.float() * 2.0 - 1.0


# QJL class.

class TorbuquantQJL(torch.nn.Module):
    """
    QJL residual quantizer (1 bit per coordinate of the residual).

    Args:
        dim    : head dimension d.
        S      : (d, d) float32 Gaussian random matrix generated once per layer
                 by build_qjl_matrix() in rotation.py.
        device : torch device.
    """

    def __init__(self, dim: int, S: torch.Tensor, device: torch.device):
        super().__init__()
        self.dim = dim
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim
        self.register_buffer("S", S.to(device=device, dtype=torch.float32))

    # Quantize.

    def quantize(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize residual r to 1-bit signs.

        Args:
            residual : (..., d) float, r = x - x_mse.

        Returns:
            signs_packed   : (..., ceil(d/8))  uint8.
            residual_norms : (...,)             float32.
        """
        residual_norms = residual.float().norm(dim=-1)         # (...,)
        projected = torch.matmul(residual.float(), self.S.T)   # (..., d)
        signs_packed = pack_signs(projected)
        return signs_packed, residual_norms

    # Dequantize contribution.

    def dequantize(
        self,
        signs_packed: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct QJL contribution: x_qjl = scale * ||r|| * S^T * z

        Returns:
            x_qjl : (..., d) float32.
        """
        z = unpack_signs(signs_packed, self.dim)                 # (..., d)
        x_qjl = torch.matmul(z, self.S)
        x_qjl = x_qjl * (self.qjl_scale * residual_norms.float().unsqueeze(-1))
        return x_qjl

    # Attention score contribution.

    def score(
        self,
        q_sketched: torch.Tensor,
        signs_packed: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute QJL attention score contribution without dequantizing.

        Instead of forming x_qjl and then dotting with query, use:
            qjl_score[n] = scale * ||r[n]|| * <q_sk, z[n]>
        where q_sk = q @ S^T is precomputed once per decode step.

        Args:
            q_sketched     : (BH, d) float32 pre-sketched query q @ S^T.
            signs_packed   : (BH, N, ceil(d/8))  uint8.
            residual_norms : (BH, N)   float32.

        Returns:
            scores : (BH, N)  float32.
        """
        BH, N, _ = signs_packed.shape
        z = unpack_signs(signs_packed.reshape(BH * N, -1), self.dim)  # (BH*N, d)
        z = z.reshape(BH, N, self.dim)                                # (BH, N, d)
        # <q_sk, z[n]> for each token n: (BH, N)
        dot = torch.bmm(z, q_sketched.unsqueeze(-1)).squeeze(-1)      # (BH, N)
        return self.qjl_scale * residual_norms * dot

    def sketch_query(self, query: torch.Tensor) -> torch.Tensor:
        """
        Pre-sketch query: q_sk = q @ S^T.

        Args:
            query : (BH, d) float query vectors.

        Returns:
            q_sketched : (BH, d)  float32.
        """
        return torch.matmul(query.float(), self.S.T)

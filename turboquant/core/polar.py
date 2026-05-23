"""TurboQuant MSE plus QJL residual codec, Algorithm 2 from the paper."""

from __future__ import annotations

import torch

from torbuquant.core.types import MSEData, ProdData
from torbuquant.core.mse import TorbuquantMSE
from torbuquant.core.qjl import TorbuquantQJL
from torbuquant.core.rotation import (
    RotationState, RotationMode,
    build_rotation, build_qjl_matrix,
    rotate_forward,
)
from torbuquant.core.codebook import get_codebook_tensors


class TorbuquantProd(torch.nn.Module):
    """
    Algorithm 2: unbiased inner-product TurboQuant quantizer.

    Args:
        dim        : head dimension d.
        bits       : total bits per coordinate (must be >= 2).
                     MSE uses (bits-1), QJL uses 1.
        rotation   : RotationState for the MSE stage (shared with layer).
        device     : torch device.
        qjl_seed   : RNG seed for the QJL Gaussian matrix S.
        use_exact  : use exact Beta codebook for MSE stage.
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        rotation: RotationState,
        device: torch.device,
        *,
        qjl_seed: int = 12345,
        use_exact: bool = True,
        qjl_for_attention: bool = False,
    ):
        super().__init__()
        if bits < 2:
            raise ValueError("TorbuquantProd requires bits >= 2")
        self.dim = dim
        self.bits = bits
        self.device = device
        self.qjl_for_attention = qjl_for_attention

        # Stage 1: MSE at (bits-1) bits
        self.mse = TorbuquantMSE(
            dim=dim,
            bits=bits - 1,
            rotation=rotation,
            device=device,
            use_exact=use_exact,
        )

        # Stage 2: QJL on the residual
        S = build_qjl_matrix(dim, device=device, dtype=torch.float32, seed=qjl_seed)
        self.qjl = TorbuquantQJL(dim=dim, S=S, device=device)

    # Quantize.

    def quantize(self, x: torch.Tensor) -> ProdData:
        """Two-stage quantization."""
        mse_q = self.mse.quantize(x)
        x_hat = self.mse.dequantize(mse_q)

        residual = x.float() - x_hat

        signs_packed, res_norms = self.qjl.quantize(residual)

        return ProdData(
            mse_indices=mse_q.indices,
            qjl_signs=signs_packed,
            residual_norms=res_norms.float(),
            norms=mse_q.norms,
            mse_bits=mse_q.bits,
            dim=self.dim,
        )

    # Dequantize.

    def dequantize(self, q: ProdData) -> torch.Tensor:
        """Reconstruct with MSE plus QJL contribution."""
        mse_data = MSEData(
            indices=q.mse_indices,
            norms=q.norms,
            bits=q.mse_bits,
            dim=q.dim,
        )
        x_mse = self.mse.dequantize(mse_data)
        x_qjl = self.qjl.dequantize(q.qjl_signs, q.residual_norms)
        return x_mse + x_qjl

    # Attention score.

    def attention_score(
        self,
        query: torch.Tensor,
        q: ProdData,
        *,
        include_qjl: bool | None = None,
    ) -> torch.Tensor:
        """Compute raw scores from compressed keys."""
        if query.dim() == 3:
            query = query.squeeze(1)

        BH, d = query.shape
        if include_qjl is None:
            include_qjl = self.qjl_for_attention

        q_rot = rotate_forward(query, self.mse.rotation)

        from torbuquant.core.mse import unpack_indices
        mse_idx = unpack_indices(q.mse_indices, q.mse_bits, d)

        centroids = self.mse.centroids
        c_vals = centroids[mse_idx]

        mse_scores = (c_vals * q_rot.unsqueeze(1)).sum(-1)
        mse_scores = mse_scores * q.norms.float()

        if not include_qjl:
            return mse_scores

        q_sk = self.qjl.sketch_query(query)
        qjl_scores = self.qjl.score(q_sk, q.qjl_signs, q.residual_norms)

        return mse_scores + qjl_scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize."""
        return self.dequantize(self.quantize(x))

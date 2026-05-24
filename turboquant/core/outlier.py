"""Channel-split TurboQuant codecs for fractional bit targets."""

from __future__ import annotations

import math

import torch

from turboquant.core.mse import TorbuquantMSE
from turboquant.core.rotation import RotationMode, build_rotation, derive_transform_seed
from turboquant.core.types import ChannelSplitData, MSEData


def split_channel_indices(
    dim: int,
    target_bits: float,
    *,
    calibration_scores: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return high-bit and low-bit channel indices for a fractional bit target."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    if target_bits < 1.0 or target_bits > 8.0:
        raise ValueError("target_bits must be in [1, 8]")

    low_bits = int(math.floor(target_bits))
    high_bits = int(math.ceil(target_bits))
    frac = target_bits - low_bits
    n_high = int(round(dim * frac))
    if low_bits == high_bits:
        n_high = 0
    if high_bits > 8:
        raise ValueError("high-bit channel width exceeds 8")

    dev = device if device is not None else (
        calibration_scores.device if calibration_scores is not None else torch.device("cpu")
    )
    all_idx = torch.arange(dim, device=dev, dtype=torch.long)
    if n_high == 0:
        return all_idx[:0], all_idx, high_bits, low_bits
    if n_high == dim:
        return all_idx, all_idx[:0], high_bits, low_bits

    if calibration_scores is not None:
        if calibration_scores.shape != (dim,):
            raise ValueError(f"calibration_scores must have shape ({dim},)")
        high_idx = torch.topk(calibration_scores.to(dev).float(), k=n_high, largest=True).indices.sort().values
    else:
        high_idx = all_idx[:n_high]

    mask = torch.ones(dim, device=dev, dtype=torch.bool)
    mask[high_idx] = False
    low_idx = all_idx[mask]
    return high_idx, low_idx, high_bits, low_bits


def channel_energy_scores(x: torch.Tensor) -> torch.Tensor:
    """Compute mean squared magnitude per channel from calibration tensors."""
    if x.ndim < 2:
        raise ValueError("calibration tensor must include a channel dimension")
    return x.float().reshape(-1, x.shape[-1]).pow(2).mean(dim=0)


class TorbuquantChannelMSE(torch.nn.Module):
    """MSE codec that assigns one extra bit to selected channels."""

    def __init__(
        self,
        *,
        dim: int,
        target_bits: float,
        device: torch.device,
        seed: int = 42,
        calibration_scores: torch.Tensor | None = None,
        rotation_mode: RotationMode = "rht",
        use_exact: bool = True,
        norm_correction: bool = True,
    ):
        super().__init__()
        high_idx, low_idx, high_bits, low_bits = split_channel_indices(
            dim,
            target_bits,
            calibration_scores=calibration_scores,
            device=device,
        )
        self.dim = dim
        self.target_bits = float(target_bits)
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.register_buffer("high_index", high_idx)
        self.register_buffer("low_index", low_idx)

        self.high_codec: TorbuquantMSE | None = None
        self.low_codec: TorbuquantMSE | None = None
        if int(high_idx.numel()) > 0:
            high_dim = int(high_idx.numel())
            high_rotation = build_rotation(
                high_dim,
                mode=rotation_mode,
                device=device,
                dtype=torch.float32,
                seed=derive_transform_seed(seed, 10_001, high_dim),
            )
            self.high_codec = TorbuquantMSE(
                dim=high_dim,
                bits=high_bits,
                rotation=high_rotation,
                device=device,
                use_exact=use_exact,
                norm_correction=norm_correction,
            )
        if int(low_idx.numel()) > 0:
            low_dim = int(low_idx.numel())
            low_rotation = build_rotation(
                low_dim,
                mode=rotation_mode,
                device=device,
                dtype=torch.float32,
                seed=derive_transform_seed(seed, 10_002, low_dim),
            )
            self.low_codec = TorbuquantMSE(
                dim=low_dim,
                bits=low_bits,
                rotation=low_rotation,
                device=device,
                use_exact=use_exact,
                norm_correction=norm_correction,
            )

    @property
    def effective_bits(self) -> float:
        high = int(self.high_index.numel()) * self.high_bits
        low = int(self.low_index.numel()) * self.low_bits
        return (high + low) / self.dim

    def quantize(self, x: torch.Tensor) -> ChannelSplitData:
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dimension must be {self.dim}")
        high_data: MSEData | None = None
        low_data: MSEData | None = None
        if self.high_codec is not None:
            high_data = self.high_codec.quantize(x.index_select(-1, self.high_index))
        if self.low_codec is not None:
            low_data = self.low_codec.quantize(x.index_select(-1, self.low_index))
        return ChannelSplitData(
            high=high_data,
            low=low_data,
            high_index=self.high_index,
            low_index=self.low_index,
            dim=self.dim,
            target_bits=self.target_bits,
        )

    def dequantize(self, q: ChannelSplitData) -> torch.Tensor:
        if q.dim != self.dim:
            raise ValueError(f"compressed dim {q.dim} does not match codec dim {self.dim}")
        shape = None
        high_recon = None
        low_recon = None
        if q.high is not None:
            if self.high_codec is None:
                raise ValueError("compressed data contains high channels but codec has none")
            high_recon = self.high_codec.dequantize(q.high)
            shape = high_recon.shape[:-1]
        if q.low is not None:
            if self.low_codec is None:
                raise ValueError("compressed data contains low channels but codec has none")
            low_recon = self.low_codec.dequantize(q.low)
            shape = low_recon.shape[:-1]
        if shape is None:
            raise ValueError("compressed data contains no channels")
        out = torch.zeros(*shape, self.dim, device=self.high_index.device, dtype=torch.float32)
        if high_recon is not None:
            out.index_copy_(-1, self.high_index, high_recon.float())
        if low_recon is not None:
            out.index_copy_(-1, self.low_index, low_recon.float())
        return out

    def storage_bits_per_vector(self) -> int:
        """Return packed index and norm bits per vector for this codec."""
        index_bits = int(self.high_index.numel()) * self.high_bits + int(self.low_index.numel()) * self.low_bits
        norm_count = int(self.high_index.numel() > 0) + int(self.low_index.numel() > 0)
        return index_bits + 16 * norm_count

    def compression_ratio(self, *, original_bits_per_value: int = 16) -> float:
        compressed = self.storage_bits_per_vector()
        return (self.dim * original_bits_per_value) / compressed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))

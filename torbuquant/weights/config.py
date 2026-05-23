"""Configuration for weight quantization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantConfig:
    """Configuration for TurboQuant weight quantization.

    Attributes:
        group_size: Group size for quantization (default: 128).
        outlier_keep_ratio: Fraction of columns to keep in fp16 (default: 0.02).
        rank: Rank for SVD residual correction (0 = disabled).
        activation_aware: Use activation statistics for importance (default: False).
        symmetric: Use symmetric quantization (default: True).
        bits: Number of bits for quantization (default: 4).
    """
    group_size: int = 128
    outlier_keep_ratio: float = 0.02
    rank: int = 0
    activation_aware: bool = False
    symmetric: bool = True
    bits: int = 4

    def __post_init__(self):
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if not 0 <= self.outlier_keep_ratio <= 1:
            raise ValueError(f"outlier_keep_ratio must be in [0, 1], got {self.outlier_keep_ratio}")
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative, got {self.rank}")
        if self.bits not in (2, 3, 4, 8):
            raise ValueError(f"bits must be 2, 3, 4, or 8, got {self.bits}")

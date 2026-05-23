"""Weight quantization module for TurboQuant.

This module provides int4 weight quantization with:
- Activation-aware importance scoring
- Outlier channel protection
- SVD low-rank residual correction
- Group-wise quantization

Example:
    ```python
    from torbuquant.weights import QuantizedLinear, QuantConfig

    # Quantize an existing linear layer
    config = QuantConfig(group_size=128, outlier_keep_ratio=0.02)
    q_linear = QuantizedLinear.from_linear(linear, config=config)

    # Use in forward pass
    output = q_linear(input)
    ```
"""

from torbuquant.weights.config import QuantConfig
from torbuquant.weights.core import (
    CompressedWeights,
    compute_channel_importance,
    identify_outliers,
    quantize_group_wise,
    dequantize_group_wise,
    compute_awq_scales,
    svd_low_rank_correction,
    pack_int4,
    unpack_int4,
    turboquant_compress,
    turboquant_decompress,
    compute_metrics,
)
from torbuquant.weights.linear import (
    QuantizedLinear,
    TurboQuantLinear,
)

__all__ = [
    "QuantConfig",
    "CompressedWeights",
    "compute_channel_importance",
    "identify_outliers",
    "quantize_group_wise",
    "dequantize_group_wise",
    "compute_awq_scales",
    "svd_low_rank_correction",
    "pack_int4",
    "unpack_int4",
    "turboquant_compress",
    "turboquant_decompress",
    "compute_metrics",
    "QuantizedLinear",
    "TurboQuantLinear",
]

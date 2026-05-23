"""PyTorch QuantizedLinear layer for TurboQuant weight quantization."""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import QuantConfig
from .core import (
    CompressedWeights,
    turboquant_compress,
    turboquant_decompress,
    pack_int4,
    unpack_int4,
)


class QuantizedLinear(nn.Module):
    """Quantized linear layer with TurboQuant compression.

    Supports:
    - Int4 weight quantization with per-group scales
    - Outlier channel protection (kept in fp16)
    - SVD low-rank residual correction
    - Activation-aware importance scoring

    Example:
        ```python
        # Create from existing linear layer
        q_linear = QuantizedLinear.from_linear(
            linear,
            config=QuantConfig(group_size=128),
        )

        # Forward pass
        output = q_linear(input)
        ```
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[QuantConfig] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantConfig()

        self.register_buffer("packed_int4", torch.zeros(0, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("zero_points", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("protected_channels", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("protected_indices", torch.zeros(0, dtype=torch.long))
        self.register_buffer("svd_u", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("svd_v", torch.zeros(0, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        self.group_size = self.config.group_size
        self.outlier_keep_ratio = self.config.outlier_keep_ratio
        self.activation_aware = self.config.activation_aware
        self._is_quantized = False
        self._original_shape: Optional[Tuple[int, int]] = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Optional[QuantConfig] = None,
        activations: Optional[torch.Tensor] = None,
    ) -> "QuantizedLinear":
        """Create a quantized linear layer from an existing nn.Linear.

        Args:
            linear: Source linear layer.
            config: Quantization configuration.
            activations: Optional activation statistics for importance.

        Returns:
            Quantized linear layer.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear)}")

        config = config or QuantConfig()

        quantized = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )

        W = linear.weight.data.detach().float().cpu().numpy()

        if activations is not None:
            activations_np = activations.detach().float().cpu().numpy()
        else:
            activations_np = None

        comp = turboquant_compress(W, config, activations_np)

        quantized._load_from_compressed(comp)

        if linear.bias is not None:
            quantized.bias = nn.Parameter(
                linear.bias.data.detach().half().clone(),
                requires_grad=False,
            )

        return quantized

    def _load_from_compressed(self, comp: CompressedWeights):
        """Load weights from compressed representation."""
        self._original_shape = comp.shape

        packed = torch.from_numpy(comp.packed_int4.copy())
        self.packed_int4 = nn.Parameter(packed, requires_grad=False)

        scales = torch.from_numpy(comp.scales.copy())
        self.scales = nn.Parameter(scales, requires_grad=False)

        if comp.zero_points is not None:
            zero_points = torch.from_numpy(comp.zero_points.copy())
            self.zero_points = nn.Parameter(zero_points, requires_grad=False)
        else:
            self.zero_points = nn.Parameter(
                torch.zeros(len(scales), dtype=torch.float16),
                requires_grad=False,
            )

        if comp.protected_channels is not None:
            protected = torch.from_numpy(comp.protected_channels.T.copy())
            self.protected_channels = nn.Parameter(protected, requires_grad=False)
            protected_indices = torch.from_numpy(comp.protected_indices.copy())
            self.protected_indices = nn.Parameter(protected_indices, requires_grad=False)
        else:
            self.protected_channels = nn.Parameter(
                torch.zeros(0, dtype=torch.float16),
                requires_grad=False,
            )
            self.protected_indices = nn.Parameter(
                torch.zeros(0, dtype=torch.long),
                requires_grad=False,
            )

        if comp.svd_u is not None and comp.svd_v is not None:
            svd_u = torch.from_numpy(comp.svd_u.copy())
            svd_v = torch.from_numpy(comp.svd_v.copy())
            self.svd_u = nn.Parameter(svd_u, requires_grad=False)
            self.svd_v = nn.Parameter(svd_v, requires_grad=False)
        else:
            self.svd_u = nn.Parameter(torch.zeros(0, dtype=torch.float16), requires_grad=False)
            self.svd_v = nn.Parameter(torch.zeros(0, dtype=torch.float16), requires_grad=False)

        self.group_size = comp.group_size
        self.outlier_keep_ratio = comp.outlier_keep_ratio
        self.activation_aware = comp.activation_aware
        self._is_quantized = True

    def _dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights to float."""
        if not self._is_quantized:
            raise RuntimeError("Layer is not quantized. Call from_linear() first.")

        shape = self._original_shape or (self.out_features, self.in_features)
        out_dim, in_dim = shape
        groups = (in_dim + self.group_size - 1) // self.group_size

        W_rec = torch.zeros(shape, dtype=torch.float32, device=self.scales.device)

        if self.packed_int4.dtype == object or (
            hasattr(self.packed_int4, 'numpy') and
            self.packed_int4.cpu().numpy().dtype == object
        ):
            # Handle object array (packed row format)
            packed_np = self.packed_int4.cpu().numpy() if hasattr(self.packed_int4, 'cpu') else self.packed_int4
            for r in range(out_dim):
                for g in range(groups):
                    start, end, packed = packed_np[r][g]
                    scale = float(self.scales[r, g].item())
                    length = end - start
                    q = unpack_int4(packed, length)
                    W_rec[r, start:end] = torch.from_numpy(q.astype(np.float32)) * scale

            if self.protected_indices.numel() > 0:
                for i, idx in enumerate(self.protected_indices):
                    W_rec[:, idx] = self.protected_channels[i].float()
        else:
            # Fallback for simple packed format
            n_elements = np.prod(shape)
            packed = self.packed_int4.flatten().cpu().numpy()
            unpacked = unpack_int4(packed, n_elements)
            W_quant = torch.from_numpy(unpacked.astype(np.float32)).reshape(shape)

            for g in range(groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_dim)
                scale = self.scales.flatten()[g].float().item()
                W_rec[:, start:end] = W_quant[:, start:end] * scale

        # Apply SVD correction
        if self.svd_u.numel() > 0 and self.svd_v.numel() > 0:
            W_rec = W_rec + self.svd_u.float() @ self.svd_v.T.float()

        return W_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        W_dequant = self._dequantize_weights().to(x.device)

        output = x.float() @ W_dequant.T.half()

        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)

        return output.half()

    def get_weight_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if not self._is_quantized:
            return {"quantized": False}

        original_size = self.out_features * self.in_features * 4
        packed_size = self.packed_int4.numel() if isinstance(self.packed_int4, torch.Tensor) else 0
        scales_size = self.scales.numel() * 2
        zero_points_size = self.zero_points.numel() * 2
        svd_size = self.svd_u.numel() * 2 + self.svd_v.numel() * 2
        protected_size = self.protected_channels.numel() * 2

        total_quantized = packed_size + scales_size + zero_points_size + svd_size + protected_size

        return {
            "quantized": True,
            "original_size_bytes": original_size,
            "quantized_size_bytes": total_quantized,
            "compression_ratio": original_size / max(1, total_quantized),
            "group_size": self.group_size,
            "outlier_keep_ratio": self.outlier_keep_ratio,
            "has_svd_correction": self.svd_u.numel() > 0,
            "has_protected_channels": self.protected_channels.numel() > 0,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quantized={self._is_quantized}"
        )


class TurboQuantLinear(nn.Module):
    """GPTQ-compatible quantized linear layer.

    This variant uses a format compatible with existing GPTQ/AWQ tooling.
    Requires CUDA kernels for efficient forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 4,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        self.register_buffer("qweight", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("qzeros", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("g_idx", torch.zeros(0, dtype=torch.long))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        self._is_quantized = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
    ) -> "TurboQuantLinear":
        """Create from existing linear layer."""
        config = QuantConfig(group_size=group_size)

        quantized = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            group_size=group_size,
        )

        W = linear.weight.data.detach().float().cpu().numpy()
        comp = turboquant_compress(W, config)

        qweight = torch.from_numpy(comp.packed_int4.copy())
        if qweight.dtype == torch.uint8:
            qweight = qweight.to(torch.int32)

        scales = torch.from_numpy(comp.scales.copy())

        if comp.zero_points is not None:
            qzeros = torch.from_numpy(comp.zero_points.copy()).to(torch.int32)
        else:
            qzeros = torch.zeros(len(scales), dtype=torch.int32)

        n_in = W.shape[1]
        n_groups = (n_in + group_size - 1) // group_size
        g_idx = torch.arange(n_groups, dtype=torch.long).repeat_interleave(group_size)[:n_in]

        quantized.qweight = nn.Parameter(qweight, requires_grad=False)
        quantized.scales = nn.Parameter(scales, requires_grad=False)
        quantized.qzeros = nn.Parameter(qzeros, requires_grad=False)
        quantized.g_idx = nn.Parameter(g_idx, requires_grad=False)
        quantized._is_quantized = True

        if linear.bias is not None:
            quantized.bias = nn.Parameter(
                linear.bias.data.detach().half().clone(),
                requires_grad=False,
            )

        return quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (requires CUDA kernel)."""
        if not self._is_quantized:
            raise RuntimeError("Layer is not quantized")

        raise NotImplementedError(
            "TurboQuantLinear requires CUDA kernel for forward pass. "
            "Use QuantizedLinear for CPU/fallback inference."
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bits={self.bits}, "
            f"group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )

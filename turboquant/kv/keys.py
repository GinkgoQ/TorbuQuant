"""Key storage codecs for cache write paths."""

from __future__ import annotations

from typing import NamedTuple

import torch

from torbuquant.core import RotationMode, TorbuquantMSE, build_rotation, derive_transform_seed
from torbuquant.core.types import MSEData, ValueData
from torbuquant.kv.values import dequantize_values, quantize_values, value_data_nbytes


class DenseKeyData(NamedTuple):
    data: torch.Tensor
    format: str = "K16"


class QuantizedKeyData(NamedTuple):
    data: ValueData
    format: str = "K8"


class TurboKeyData(NamedTuple):
    data: MSEData
    format: str = "K4"


KeyPayload = DenseKeyData | QuantizedKeyData | TurboKeyData


def quantize_k8(keys: torch.Tensor, *, group_size: int = 32) -> QuantizedKeyData:
    return QuantizedKeyData(
        data=quantize_values(keys, bits=8, group_size=group_size),
        format="K8",
    )


def dequantize_k8(payload: QuantizedKeyData, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return dequantize_values(payload.data, dtype=dtype)


def key_payload_nbytes(payload: KeyPayload) -> dict[str, int]:
    if isinstance(payload, DenseKeyData):
        return {
            "compressed_k_bytes": payload.data.numel() * payload.data.element_size(),
            "scales_bytes": 0,
            "zeros_bytes": 0,
            "norms_bytes": 0,
        }
    if isinstance(payload, QuantizedKeyData):
        counts = value_data_nbytes(payload.data)
        return {
            "compressed_k_bytes": counts["compressed_v_bytes"],
            "scales_bytes": counts["scales_bytes"],
            "zeros_bytes": counts["zeros_bytes"],
            "norms_bytes": 0,
        }
    return {
        "compressed_k_bytes": payload.data.indices.numel() * payload.data.indices.element_size(),
        "scales_bytes": 0,
        "zeros_bytes": 0,
        "norms_bytes": payload.data.norms.numel() * payload.data.norms.element_size(),
    }


def build_k4_quantizer(
    *,
    head_dim: int,
    layer_idx: int,
    device: torch.device,
    seed: int,
) -> TorbuquantMSE:
    transform_seed = derive_transform_seed(seed, layer_idx=layer_idx, head_idx=0)
    rotation = build_rotation(head_dim, RotationMode.RHT, device=device, dtype=torch.float32, seed=transform_seed)
    return TorbuquantMSE(head_dim, 4, rotation, device=device, use_exact=True, norm_correction=True)

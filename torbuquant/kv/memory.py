"""KV cache byte accounting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch

from torbuquant.packing.bits import tensor_nbytes


@dataclass(frozen=True)
class MemoryReport:
    dense_kv_bytes: int = 0
    compressed_k_bytes: int = 0
    compressed_v_bytes: int = 0
    scales_bytes: int = 0
    zeros_bytes: int = 0
    norms_bytes: int = 0
    centroids_bytes: int = 0
    rotation_bytes: int = 0
    metadata_bytes: int = 0
    recent_window_bytes: int = 0
    boundary_bytes: int = 0
    sparse_bytes: int = 0
    outlier_bytes: int = 0
    workspace_bytes: int = 0
    peak_allocated_bytes: int = 0
    formula_total_bytes: int = 0

    @property
    def compressed_total_bytes(self) -> int:
        return (
            self.compressed_k_bytes
            + self.compressed_v_bytes
            + self.scales_bytes
            + self.zeros_bytes
            + self.norms_bytes
            + self.centroids_bytes
            + self.rotation_bytes
            + self.metadata_bytes
            + self.recent_window_bytes
            + self.boundary_bytes
            + self.sparse_bytes
            + self.outlier_bytes
            + self.workspace_bytes
        )

    @property
    def measured_total_bytes(self) -> int:
        return self.compressed_total_bytes + self.peak_allocated_bytes

    @property
    def compression_ratio(self) -> float:
        if self.compressed_total_bytes == 0:
            return float("inf")
        return self.dense_kv_bytes / self.compressed_total_bytes

    @property
    def formula_compression_ratio(self) -> float:
        if self.formula_total_bytes == 0:
            return float("inf")
        return self.dense_kv_bytes / self.formula_total_bytes

    def as_dict(self) -> dict[str, int | float]:
        data = {
            "dense_kv_bytes": self.dense_kv_bytes,
            "compressed_k_bytes": self.compressed_k_bytes,
            "compressed_v_bytes": self.compressed_v_bytes,
            "scales_bytes": self.scales_bytes,
            "zeros_bytes": self.zeros_bytes,
            "norms_bytes": self.norms_bytes,
            "centroids_bytes": self.centroids_bytes,
            "rotation_bytes": self.rotation_bytes,
            "metadata_bytes": self.metadata_bytes,
            "recent_window_bytes": self.recent_window_bytes,
            "boundary_bytes": self.boundary_bytes,
            "sparse_bytes": self.sparse_bytes,
            "outlier_bytes": self.outlier_bytes,
            "workspace_bytes": self.workspace_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "formula_total_bytes": self.formula_total_bytes,
            "compressed_total_bytes": self.compressed_total_bytes,
            "measured_total_bytes": self.measured_total_bytes,
            "compression_ratio": self.compression_ratio,
            "formula_compression_ratio": self.formula_compression_ratio,
        }
        return data


def bytes_from_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Return a byte ledger for named tensors."""
    return {name: tensor_nbytes(tensor) for name, tensor in tensors.items()}


class ByteLedger:
    """Mutable builder for MemoryReport."""

    def __init__(self, dense_kv_bytes: int = 0):
        self._report = MemoryReport(dense_kv_bytes=dense_kv_bytes)

    def add_bytes(self, field: str, value: int) -> None:
        if not hasattr(self._report, field):
            raise ValueError(f"unknown MemoryReport field: {field}")
        current = getattr(self._report, field)
        self._report = replace(self._report, **{field: current + int(value)})

    def add_tensor(self, field: str, tensor: torch.Tensor) -> None:
        self.add_bytes(field, tensor_nbytes(tensor))

    def add_tensors(self, field: str, tensors: Mapping[str, torch.Tensor]) -> None:
        for tensor in tensors.values():
            self.add_tensor(field, tensor)

    def to_report(self) -> MemoryReport:
        return self._report


def report_from_components(
    *,
    dense_kv_bytes: int,
    compressed_k: torch.Tensor | None = None,
    compressed_v: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    zeros: torch.Tensor | None = None,
    norms: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    centroids: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    rotation: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    metadata_bytes: int = 0,
    recent_window: torch.Tensor | None = None,
    boundary: torch.Tensor | None = None,
    sparse: torch.Tensor | None = None,
    outlier: torch.Tensor | None = None,
    workspace: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    peak_allocated_bytes: int = 0,
    formula_total_bytes: int = 0,
) -> MemoryReport:
    """Build a MemoryReport from actual tensors."""
    ledger = ByteLedger(dense_kv_bytes=dense_kv_bytes)
    optional = [
        ("compressed_k_bytes", compressed_k),
        ("compressed_v_bytes", compressed_v),
        ("scales_bytes", scales),
        ("zeros_bytes", zeros),
        ("recent_window_bytes", recent_window),
        ("boundary_bytes", boundary),
        ("sparse_bytes", sparse),
        ("outlier_bytes", outlier),
    ]
    for field, tensor in optional:
        if tensor is not None:
            ledger.add_tensor(field, tensor)

    for field, item in [
        ("norms_bytes", norms),
        ("centroids_bytes", centroids),
        ("rotation_bytes", rotation),
        ("workspace_bytes", workspace),
    ]:
        if isinstance(item, torch.Tensor):
            ledger.add_tensor(field, item)
        elif item is not None:
            ledger.add_tensors(field, item)

    ledger.add_bytes("metadata_bytes", metadata_bytes)
    ledger.add_bytes("peak_allocated_bytes", peak_allocated_bytes)
    ledger.add_bytes("formula_total_bytes", formula_total_bytes)
    return ledger.to_report()

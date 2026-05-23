"""Reporting helpers for vLLM TurboQuant runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CacheByteReport:
    dense_kv_bytes: int
    packed_kv_bytes: int
    metadata_bytes: int = 0
    workspace_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.packed_kv_bytes + self.metadata_bytes + self.workspace_bytes

    @property
    def compression_ratio(self) -> float:
        if self.total_bytes == 0:
            return float("inf")
        return self.dense_kv_bytes / self.total_bytes

    def to_dict(self) -> dict[str, int | float]:
        data = asdict(self)
        data["total_bytes"] = self.total_bytes
        data["compression_ratio"] = self.compression_ratio
        return data


@dataclass(frozen=True)
class RuntimeReport:
    cache_dtype: str
    path: str
    bytes: CacheByteReport
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_dtype": self.cache_dtype,
            "path": self.path,
            "bytes": self.bytes.to_dict(),
            "details": self.details,
        }

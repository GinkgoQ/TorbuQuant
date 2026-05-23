"""Dense recent-window storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RecentWindow:
    capacity: int
    num_kv_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {self.capacity}")
        self._keys = torch.empty(self.capacity, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        self._values = torch.empty_like(self._keys)
        self._size = 0
        self._total_written = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def total_written(self) -> int:
        return self._total_written

    @property
    def bytes(self) -> int:
        return self._keys.numel() * self._keys.element_size() + self._values.numel() * self._values.element_size()

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if keys.shape != values.shape:
            raise ValueError(f"key/value shape mismatch: {keys.shape} vs {values.shape}")
        if keys.ndim != 3:
            raise ValueError("recent window expects (tokens, num_kv_heads, head_dim)")
        if self.capacity == 0:
            self._total_written += int(keys.shape[0])
            return keys, values

        keys = keys.to(device=self.device, dtype=self.dtype)
        values = values.to(device=self.device, dtype=self.dtype)
        current_k, current_v = self.peek()
        if current_k is None:
            all_k, all_v = keys, values
        else:
            all_k = torch.cat([current_k, keys], dim=0)
            all_v = torch.cat([current_v, values], dim=0)

        overflow = max(0, all_k.shape[0] - self.capacity)
        overflow_k = all_k[:overflow].contiguous() if overflow else None
        overflow_v = all_v[:overflow].contiguous() if overflow else None
        keep_k = all_k[overflow:].contiguous()
        keep_v = all_v[overflow:].contiguous()
        self._size = int(keep_k.shape[0])
        self._keys[: self._size].copy_(keep_k)
        self._values[: self._size].copy_(keep_v)
        self._total_written += int(keys.shape[0])
        return overflow_k, overflow_v

    def peek(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._size == 0:
            return None, None
        return self._keys[: self._size], self._values[: self._size]

    def drain(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        keys, values = self.peek()
        if keys is None:
            return None, None
        out_k = keys.clone()
        out_v = values.clone()
        self._size = 0
        return out_k, out_v

    def to_report(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "total_written": self._total_written,
            "bytes": self.bytes,
        }

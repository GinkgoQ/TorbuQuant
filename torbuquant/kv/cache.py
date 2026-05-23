"""Compressed KV cache ownership and write paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from torbuquant.core import TorbuquantMSE
from torbuquant.kv.formats import KVFormatSpec, validate_k_format, validate_v_format
from torbuquant.kv.keys import (
    DenseKeyData,
    KeyPayload,
    QuantizedKeyData,
    TurboKeyData,
    build_k4_quantizer,
    dequantize_k8,
    key_payload_nbytes,
    quantize_k8,
)
from torbuquant.kv.memory import MemoryReport
from torbuquant.kv.policy import KVQuantPolicy
from torbuquant.kv.recent import RecentWindow
from torbuquant.kv.values import dequantize_values, quantize_values, value_data_nbytes


class DiagnosticDenseKV(NamedTuple):
    label: str
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class BoundaryPolicy:
    tokens: int = 0
    preserve_first_layers: int = 0
    preserve_last_layers: int = 0

    def preserves_layer(self, layer_idx: int, num_layers: int | None = None) -> bool:
        if layer_idx < self.preserve_first_layers:
            return True
        if num_layers is not None and layer_idx >= num_layers - self.preserve_last_layers:
            return True
        return False


@dataclass(frozen=True)
class SparseOutlierStore:
    indices: torch.Tensor | None = None
    key_values: torch.Tensor | None = None
    value_values: torch.Tensor | None = None

    @property
    def bytes(self) -> int:
        total = 0
        for tensor in (self.indices, self.key_values, self.value_values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    def to_report(self) -> dict[str, int]:
        return {
            "bytes": self.bytes,
            "num_indices": 0 if self.indices is None else int(self.indices.numel()),
        }


@dataclass(frozen=True)
class CompressedKVBlock:
    key: KeyPayload
    value: Any
    start: int
    length: int
    k_format: str
    v_format: str
    diagnostic_label: str | None = None

    def to_report(self) -> dict[str, Any]:
        key_bytes = key_payload_nbytes(self.key)
        value_bytes = value_data_nbytes(self.value)
        return {
            "start": self.start,
            "length": self.length,
            "k_format": self.k_format,
            "v_format": self.v_format,
            "diagnostic_label": self.diagnostic_label,
            **key_bytes,
            **value_bytes,
        }


@dataclass(frozen=True)
class CompressedKVPage:
    block: CompressedKVBlock
    page_id: int
    block_size: int

    def to_report(self) -> dict[str, Any]:
        data = self.block.to_report()
        data.update({"page_id": self.page_id, "block_size": self.block_size})
        return data


class ProductionKVHandle:
    def __init__(self, cache: "CompressedKVCache"):
        self._cache = cache

    @property
    def blocks(self) -> tuple[CompressedKVBlock, ...]:
        return tuple(self._cache.blocks)

    @property
    def recent(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._cache.recent_window.peek()

    def dense(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("production cache handle cannot expose dense historical K/V")

    def to_report(self) -> dict[str, Any]:
        return self._cache.to_report()


class CompressedKVCache:
    def __init__(
        self,
        *,
        head_dim: int,
        num_kv_heads: int,
        layer_idx: int,
        policy: KVQuantPolicy,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        chunk_size: int = 128,
        num_layers: int | None = None,
        seed: int = 42,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_idx = layer_idx
        self.policy = policy
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.seed = seed
        self.k_spec: KVFormatSpec = validate_k_format(
            policy.k_format,
            allow_low_bit_keys=policy.allow_k_low_bits,
        )
        self.v_spec: KVFormatSpec = validate_v_format(policy.v_format, allow_v3=policy.allow_v3)
        self.boundary_policy = BoundaryPolicy(
            tokens=policy.boundary_tokens,
            preserve_first_layers=policy.preserve_first_layers,
            preserve_last_layers=policy.preserve_last_layers,
        )
        self.recent_window = RecentWindow(
            capacity=policy.recent_window,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        self.blocks: list[CompressedKVBlock] = []
        self.boundary_keys: torch.Tensor | None = None
        self.boundary_values: torch.Tensor | None = None
        self.layer_preserved = self.boundary_policy.preserves_layer(layer_idx, num_layers)
        self.layer_keys: torch.Tensor | None = None
        self.layer_values: torch.Tensor | None = None
        self.outliers = SparseOutlierStore()
        self.seq_len = 0
        self._compressed_tokens = 0
        self._fallback_events: list[dict[str, Any]] = []
        self._k4_quantizer: TorbuquantMSE | None = None

    @property
    def compressed_tokens(self) -> int:
        return self._compressed_tokens

    @property
    def recent_tokens(self) -> int:
        return self.recent_window.size

    def prefill(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        keys, values = self._validate_kv(keys, values)
        total = int(keys.shape[0])
        self.seq_len += total
        if self.layer_preserved:
            self._append_layer_dense(keys, values)
            return
        cursor = 0
        boundary = self.boundary_policy.tokens
        if boundary and total > 0:
            n = min(boundary, total)
            self.boundary_keys = keys[:n].to(device=self.device, dtype=self.dtype).contiguous()
            self.boundary_values = values[:n].to(device=self.device, dtype=self.dtype).contiguous()
            cursor = n

        remaining = keys[cursor:]
        remaining_v = values[cursor:]
        if remaining.shape[0] <= self.recent_window.capacity:
            self.recent_window.append(remaining, remaining_v)
            return

        n_compress = remaining.shape[0] - self.recent_window.capacity
        self._append_compressed_chunks(remaining[:n_compress], remaining_v[:n_compress])
        self.recent_window.append(remaining[n_compress:], remaining_v[n_compress:])

    def append_decode(self, key: torch.Tensor, value: torch.Tensor) -> None:
        key, value = self._validate_kv(key, value)
        self.seq_len += int(key.shape[0])
        if self.layer_preserved:
            self._append_layer_dense(key, value)
            return
        overflow_k, overflow_v = self.recent_window.append(key, value)
        if overflow_k is not None:
            self._append_compressed_chunks(overflow_k, overflow_v)

    def flush_recent(self) -> None:
        keys, values = self.recent_window.drain()
        if keys is not None and not self.layer_preserved:
            self._append_compressed_chunks(keys, values)

    def production_handle(self) -> ProductionKVHandle:
        return ProductionKVHandle(self)

    def diagnostic_dense(self) -> DiagnosticDenseKV:
        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []
        if self.layer_keys is not None:
            return DiagnosticDenseKV(
                "diagnostic_dense_layer",
                self.layer_keys.float(),
                self.layer_values.float(),
            )
        if self.boundary_keys is not None:
            parts_k.append(self.boundary_keys.float())
            parts_v.append(self.boundary_values.float())
        for block in self.blocks:
            k, v = self._dequantize_block(block)
            parts_k.append(k)
            parts_v.append(v)
        recent_k, recent_v = self.recent_window.peek()
        if recent_k is not None:
            parts_k.append(recent_k.float())
            parts_v.append(recent_v.float())
        empty_shape = (0, self.num_kv_heads, self.head_dim)
        keys = (
            torch.cat(parts_k, dim=0)
            if parts_k
            else torch.empty(empty_shape, device=self.device, dtype=torch.float32)
        )
        values = (
            torch.cat(parts_v, dim=0)
            if parts_v
            else torch.empty(empty_shape, device=self.device, dtype=torch.float32)
        )
        return DiagnosticDenseKV("diagnostic_dequant", keys, values)

    def memory_report(self) -> MemoryReport:
        dense_bytes = (
            2
            * self.seq_len
            * self.num_kv_heads
            * self.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
        )
        compressed_k = 0
        compressed_v = 0
        scales = 0
        zeros = 0
        norms = 0
        for block in self.blocks:
            kb = key_payload_nbytes(block.key)
            vb = value_data_nbytes(block.value)
            compressed_k += kb["compressed_k_bytes"]
            compressed_v += vb["compressed_v_bytes"]
            scales += kb["scales_bytes"] + vb["scales_bytes"]
            zeros += kb["zeros_bytes"] + vb["zeros_bytes"]
            norms += kb["norms_bytes"]
        boundary_bytes = 0
        if self.boundary_keys is not None:
            boundary_bytes += self.boundary_keys.numel() * self.boundary_keys.element_size()
            boundary_bytes += self.boundary_values.numel() * self.boundary_values.element_size()
        if self.layer_keys is not None:
            boundary_bytes += self.layer_keys.numel() * self.layer_keys.element_size()
            boundary_bytes += self.layer_values.numel() * self.layer_values.element_size()
        centroids_bytes, rotation_bytes = self._k4_metadata_bytes()
        return MemoryReport(
            dense_kv_bytes=dense_bytes,
            compressed_k_bytes=compressed_k,
            compressed_v_bytes=compressed_v,
            scales_bytes=scales,
            zeros_bytes=zeros,
            norms_bytes=norms,
            centroids_bytes=centroids_bytes,
            rotation_bytes=rotation_bytes,
            recent_window_bytes=self.recent_window.bytes,
            boundary_bytes=boundary_bytes,
            outlier_bytes=self.outliers.bytes,
            metadata_bytes=len(str(self.to_report()).encode("utf-8")),
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads,
            "layer_idx": self.layer_idx,
            "seq_len": self.seq_len,
            "compressed_tokens": self.compressed_tokens,
            "recent_tokens": self.recent_tokens,
            "layer_preserved": self.layer_preserved,
            "layer_dense_tokens": 0 if self.layer_keys is None else int(self.layer_keys.shape[0]),
            "k_format": self.policy.k_format,
            "v_format": self.policy.v_format,
            "policy": self.policy.to_dict(),
            "k4_metadata": self._k4_metadata_report(),
            "recent_window": self.recent_window.to_report(),
            "boundary": {
                "tokens": self.boundary_policy.tokens,
                "stored_tokens": 0 if self.boundary_keys is None else int(self.boundary_keys.shape[0]),
            },
            "blocks": [block.to_report() for block in self.blocks],
            "outliers": self.outliers.to_report(),
            "fallback_events": list(self._fallback_events),
        }

    def _validate_kv(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.shape != values.shape:
            raise ValueError(f"key/value shape mismatch: {keys.shape} vs {values.shape}")
        if keys.ndim != 3:
            raise ValueError("cache expects (tokens, num_kv_heads, head_dim)")
        if keys.shape[1] != self.num_kv_heads or keys.shape[2] != self.head_dim:
            raise ValueError("cache tensor shape does not match cache geometry")
        return keys.to(self.device), values.to(self.device)

    def _append_layer_dense(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        keys = keys.to(device=self.device, dtype=self.dtype).contiguous()
        values = values.to(device=self.device, dtype=self.dtype).contiguous()
        if self.layer_keys is None:
            self.layer_keys = keys
            self.layer_values = values
            return
        self.layer_keys = torch.cat([self.layer_keys, keys], dim=0).contiguous()
        self.layer_values = torch.cat([self.layer_values, values], dim=0).contiguous()

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor | None) -> int:
        if tensor is None:
            return 0
        return tensor.numel() * tensor.element_size()

    def _k4_metadata_bytes(self) -> tuple[int, int]:
        if self._k4_quantizer is None:
            return 0, 0
        centroids_bytes = (
            self._tensor_bytes(self._k4_quantizer.centroids)
            + self._tensor_bytes(self._k4_quantizer.decision_bnd)
        )
        rotation = self._k4_quantizer.rotation
        rotation_bytes = (
            self._tensor_bytes(rotation.matrix)
            + self._tensor_bytes(rotation.signs1)
            + self._tensor_bytes(rotation.signs2)
        )
        return centroids_bytes, rotation_bytes

    def _k4_metadata_report(self) -> dict[str, Any] | None:
        if self._k4_quantizer is None:
            return None
        centroids_bytes, rotation_bytes = self._k4_metadata_bytes()
        return {
            "bits": self._k4_quantizer.bits,
            "dim": self._k4_quantizer.dim,
            "centroids_bytes": centroids_bytes,
            "rotation_bytes": rotation_bytes,
            "rotation": self._k4_quantizer.rotation.to_metadata(),
        }

    def _append_compressed_chunks(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        for start in range(0, int(keys.shape[0]), self.chunk_size):
            k = keys[start : start + self.chunk_size]
            v = values[start : start + self.chunk_size]
            block = self._compress_block(k, v, start_token=self._compressed_tokens)
            self.blocks.append(block)
            self._compressed_tokens += int(k.shape[0])

    def _compress_block(self, keys: torch.Tensor, values: torch.Tensor, *, start_token: int) -> CompressedKVBlock:
        if self.policy.k_format == "K16":
            key_payload: KeyPayload = DenseKeyData(keys.to(dtype=self.dtype).contiguous(), "K16")
        elif self.policy.k_format == "K8":
            key_payload = quantize_k8(keys, group_size=32)
        elif self.policy.k_format == "K4":
            if self._k4_quantizer is None:
                self._k4_quantizer = build_k4_quantizer(
                    head_dim=self.head_dim,
                    layer_idx=self.layer_idx,
                    device=self.device,
                    seed=self.seed,
                )
            flat = keys.reshape(-1, self.head_dim)
            q = self._k4_quantizer.quantize(flat)
            shaped = q._replace(
                indices=q.indices.reshape(keys.shape[0], self.num_kv_heads, -1),
                norms=q.norms.reshape(keys.shape[0], self.num_kv_heads),
            )
            key_payload = TurboKeyData(shaped, "K4")
        else:
            raise ValueError(f"unsupported key format in cache write path: {self.policy.k_format}")

        value_bits = int(self.policy.v_format[1:])
        value_payload = quantize_values(
            values,
            bits=value_bits,
            group_size=32,
            allow_v3=self.policy.allow_v3,
        )
        return CompressedKVBlock(
            key=key_payload,
            value=value_payload,
            start=start_token,
            length=int(keys.shape[0]),
            k_format=self.policy.k_format,
            v_format=self.policy.v_format,
        )

    def _dequantize_block(self, block: CompressedKVBlock) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(block.key, DenseKeyData):
            keys = block.key.data.float()
        elif isinstance(block.key, QuantizedKeyData):
            keys = dequantize_k8(block.key).float()
        else:
            if self._k4_quantizer is None:
                self._k4_quantizer = build_k4_quantizer(
                    head_dim=self.head_dim,
                    layer_idx=self.layer_idx,
                    device=self.device,
                    seed=self.seed,
                )
            flat_q = block.key.data._replace(
                indices=block.key.data.indices.reshape(-1, block.key.data.indices.shape[-1]),
                norms=block.key.data.norms.reshape(-1),
            )
            keys = self._k4_quantizer.dequantize(flat_q).reshape(
                block.length,
                self.num_kv_heads,
                self.head_dim,
            )
        values = dequantize_values(block.value).float()
        return keys, values

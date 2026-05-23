"""TurboQuant compressed KV store - owns the quantized historical segment.

Design rules:
  - Chunks are stored in lists; concatenation is deferred (lazy flatten).
  - Flat cache is materialized on first read and invalidated on write.
  - No per-token overhead; all writes are chunk-based.
"""

from __future__ import annotations

import torch
from typing import Optional, NamedTuple
from dataclasses import dataclass

from torbuquant.kv.compressors import (
    TurboQuantCompressorV2,
    TurboQuantCompressorMSE,
    CompressedKeys,
    CompressedValues,
)


@dataclass
class ValueQuantized:
    """Quantized value cache (bit-packed)."""
    data: torch.Tensor       # (..., n_tokens, packed_d) bit-packed quantized values
    scales: torch.Tensor     # (..., n_tokens, n_groups) scale per group
    zeros: torch.Tensor      # (..., n_tokens, n_groups) zero point per group
    bits: int = 2            # quantization bits (for unpacking)


def unpack_value_data(vq: ValueQuantized) -> torch.Tensor:
    """Unpack bit-packed value data to uint8 per-element."""
    bits = vq.bits
    packed = vq.data
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    elif bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        return torch.stack([v0, v1], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return packed


def quantize_value_chunk(
    v: torch.Tensor,
    bits: int = 2,
    group_size: int = 32,
) -> ValueQuantized:
    """Symmetric group quantization for value vectors.

    Args:
        v: (..., seq_len, d) value vectors
        bits: quantization bits (2 or 4)
        group_size: number of elements per quantization group
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    if d % group_size != 0:
        # Pad to group_size
        pad_size = group_size - (d % group_size)
        v = torch.nn.functional.pad(v, (0, pad_size))
        d = v.shape[-1]
        n_groups = d // group_size

    # Reshape to groups
    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)

    # Compute scale and zero per group (asymmetric)
    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values

    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)
    zero = v_min

    # Quantize
    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    # Bit-pack
    if bits == 2:
        assert d % 4 == 0
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = v_4[..., 0] | (v_4[..., 1] << 2) | (v_4[..., 2] << 4) | (v_4[..., 3] << 6)
        v_q_flat = packed
    elif bits == 4:
        assert d % 2 == 0
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = v_2[..., 0] | (v_2[..., 1] << 4)
        v_q_flat = packed

    return ValueQuantized(
        data=v_q_flat,
        scales=scale.squeeze(-1),
        zeros=zero.squeeze(-1),
        bits=bits,
    )


def dequantize_value_chunk(
    vq: ValueQuantized,
    group_size: int = 32,
) -> torch.Tensor:
    """Dequantize value vectors from bit-packed format."""
    data = unpack_value_data(vq).float()
    d = data.shape[-1]
    batch_shape = data.shape[:-1]

    n_groups = d // group_size
    data = data.reshape(*batch_shape, n_groups, group_size)
    scales = vq.scales.unsqueeze(-1)
    zeros = vq.zeros.unsqueeze(-1)

    v = data * scales + zeros
    return v.reshape(*batch_shape, d)


class FlatCache(NamedTuple):
    """Flattened view of compressed KV for fast read access."""
    key_compressed: CompressedKeys   # Compressed keys
    value_compressed: CompressedValues  # Compressed values
    num_tokens: int


class CompressedKVStore:
    """Chunked compressed KV store with lazy flattening.

    Keys are quantized via TurboQuantCompressorV2 (unbiased inner-product estimator).
    Values use TurboQuantCompressorMSE.
    Chunks are kept in lists until a flat view is requested.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        device: torch.device = None,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = min(value_group_size, head_dim)
        self.device = device or torch.device("cuda")
        self.layer_idx = layer_idx

        self.key_compressor = TurboQuantCompressorV2(
            head_dim=head_dim,
            bits=key_bits,
            seed=42 + layer_idx * 7,
            device=str(self.device),
        )
        self.value_compressor = TurboQuantCompressorMSE(
            head_dim=head_dim,
            bits=value_bits,
            seed=42 + layer_idx * 7 + 500,
            device=str(self.device),
        )

        self._key_chunks: list[CompressedKeys] = []
        self._value_chunks: list[CompressedValues] = []
        self._chunk_lengths: list[int] = []

        self._flat: Optional[FlatCache] = None

    @property
    def num_tokens(self) -> int:
        """Total number of compressed tokens."""
        return sum(self._chunk_lengths)

    @property
    def num_chunks(self) -> int:
        """Number of stored chunks."""
        return len(self._chunk_lengths)

    def append_chunk(self, key: torch.Tensor, value: torch.Tensor):
        """Quantize and store a chunk of KV pairs.

        key/value: (chunk_len, num_kv_heads, head_dim)
        """
        chunk_len = key.shape[0]

        # Reshape to (1, num_kv_heads, chunk_len, head_dim) for compressor
        k = key.transpose(0, 1).unsqueeze(0)  # (1, H, T, D)
        v = value.transpose(0, 1).unsqueeze(0)

        key_compressed = self.key_compressor.compress(k)
        value_compressed = self.value_compressor.compress(v)

        self._key_chunks.append(key_compressed)
        self._value_chunks.append(value_compressed)
        self._chunk_lengths.append(chunk_len)
        self._flat = None  # Invalidate cached flat view

    def get_flat_cache(self) -> Optional[FlatCache]:
        """Return a flattened view of all compressed tokens. Cached until next write."""
        if not self._key_chunks:
            return None

        if self._flat is not None:
            return self._flat

        if len(self._key_chunks) == 1:
            flat_k = self._flatten_keys(self._key_chunks[0])
            flat_v = self._flatten_values(self._value_chunks[0])
        else:
            flat_k = self._concat_keys([self._flatten_keys(c) for c in self._key_chunks])
            flat_v = self._concat_values([self._flatten_values(c) for c in self._value_chunks])

        self._flat = FlatCache(
            key_compressed=flat_k,
            value_compressed=flat_v,
            num_tokens=self.num_tokens,
        )
        return self._flat

    def _flatten_keys(self, ck: CompressedKeys) -> CompressedKeys:
        """Collapse batch dim: (1, H, T, ...) -> (H, T, ...)."""
        return CompressedKeys(
            indices=ck.indices.reshape(-1, ck.indices.shape[-2], ck.indices.shape[-1]).contiguous(),
            norms=ck.norms.reshape(-1, ck.norms.shape[-1]).contiguous(),
            qjl_signs=ck.qjl_signs.reshape(-1, ck.qjl_signs.shape[-2], ck.qjl_signs.shape[-1]).contiguous(),
            residual_norms=ck.residual_norms.reshape(-1, ck.residual_norms.shape[-1]).contiguous(),
            original_dtype=ck.original_dtype,
        )

    def _flatten_values(self, cv: CompressedValues) -> CompressedValues:
        """Collapse batch dim: (1, H, T, ...) -> (H, T, ...)."""
        return CompressedValues(
            indices=cv.indices.reshape(-1, cv.indices.shape[-2], cv.indices.shape[-1]).contiguous(),
            norms=cv.norms.reshape(-1, cv.norms.shape[-1]).contiguous(),
            original_dtype=cv.original_dtype,
        )

    def _concat_keys(self, chunks: list[CompressedKeys]) -> CompressedKeys:
        """Concatenate multiple flattened CompressedKeys along the token dimension."""
        return CompressedKeys(
            indices=torch.cat([c.indices for c in chunks], dim=-2),
            norms=torch.cat([c.norms for c in chunks], dim=-1),
            qjl_signs=torch.cat([c.qjl_signs for c in chunks], dim=-2),
            residual_norms=torch.cat([c.residual_norms for c in chunks], dim=-1),
            original_dtype=chunks[0].original_dtype,
        )

    def _concat_values(self, chunks: list[CompressedValues]) -> CompressedValues:
        """Concatenate multiple flattened CompressedValues along the token dimension."""
        return CompressedValues(
            indices=torch.cat([c.indices for c in chunks], dim=-2),
            norms=torch.cat([c.norms for c in chunks], dim=-1),
            original_dtype=chunks[0].original_dtype,
        )

    def memory_bytes(self) -> int:
        """Estimate GPU memory used by compressed data."""
        total = 0
        for ck in self._key_chunks:
            total += ck.indices.nelement()  # uint8
            total += ck.norms.nelement() * 2  # fp16
            total += ck.qjl_signs.nelement()  # int8
            total += ck.residual_norms.nelement() * 2  # fp16
        for cv in self._value_chunks:
            total += cv.indices.nelement()  # uint8
            total += cv.norms.nelement() * 2  # fp16
        return total

    def reset(self):
        """Clear all stored chunks."""
        self._key_chunks.clear()
        self._value_chunks.clear()
        self._chunk_lengths.clear()
        self._flat = None


class TurboQuantKVCache:
    """KV cache using TurboQuant for keys and group quantization for values.

    Drop-in replacement concept for a standard KV cache with compression.

    Usage:
        cache = TurboQuantKVCache(head_dim=128, key_bits=3, value_bits=2)

        # During prefill:
        cache.prefill(key_states, value_states)

        # During decode (one token at a time):
        cache.append(new_key, new_value)

        # Compute attention:
        scores = cache.attention_scores(query_states)
        output = cache.attend(query_states, scores_after_softmax)
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self.device = device or torch.device("cuda")
        self.dtype = dtype
        self.layer_idx = layer_idx

        self.key_compressor = TurboQuantCompressorV2(
            head_dim=head_dim,
            bits=key_bits,
            seed=42 + layer_idx * 7,
            device=str(self.device),
        )
        self.value_compressor = TurboQuantCompressorMSE(
            head_dim=head_dim,
            bits=value_bits,
            seed=42 + layer_idx * 7 + 500,
            device=str(self.device),
        )

        # State
        self.seq_len: int = 0
        self.key_quantized: Optional[CompressedKeys] = None
        self.value_quantized: Optional[CompressedValues] = None

        # Buffer for recent unquantized tokens
        self.key_buffer: Optional[torch.Tensor] = None
        self.value_buffer: Optional[torch.Tensor] = None

    def prefill(self, keys: torch.Tensor, values: torch.Tensor):
        """Process prefill tokens.

        Args:
            keys: (batch, n_heads, seq_len, head_dim)
            values: (batch, n_heads, seq_len, head_dim)
        """
        seq_len = keys.shape[-2]
        self.seq_len = seq_len

        if seq_len <= self.buffer_size:
            # Everything fits in buffer
            self.key_buffer = keys
            self.value_buffer = values
            return

        # Split into quantized portion and buffer
        n_quant = seq_len - self.buffer_size

        keys_to_quant = keys[..., :n_quant, :]
        values_to_quant = values[..., :n_quant, :]

        self.key_buffer = keys[..., n_quant:, :]
        self.value_buffer = values[..., n_quant:, :]

        # Quantize
        self.key_quantized = self.key_compressor.compress(keys_to_quant)
        self.value_quantized = self.value_compressor.compress(values_to_quant)

    def append(self, key: torch.Tensor, value: torch.Tensor):
        """Append a single decode token.

        Args:
            key: (batch, n_heads, 1, head_dim)
            value: (batch, n_heads, 1, head_dim)
        """
        self.seq_len += 1

        if self.key_buffer is not None:
            self.key_buffer = torch.cat([self.key_buffer, key], dim=-2)
            self.value_buffer = torch.cat([self.value_buffer, value], dim=-2)
        else:
            self.key_buffer = key
            self.value_buffer = value

        # If buffer exceeds size, flush oldest chunk
        if self.key_buffer.shape[-2] > self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Move oldest tokens from buffer to quantized storage."""
        n_flush = self.key_buffer.shape[-2] - self.buffer_size

        keys_flush = self.key_buffer[..., :n_flush, :]
        values_flush = self.value_buffer[..., :n_flush, :]

        self.key_buffer = self.key_buffer[..., n_flush:, :]
        self.value_buffer = self.value_buffer[..., n_flush:, :]

        # Quantize flushed tokens
        new_key_q = self.key_compressor.compress(keys_flush)
        new_val_q = self.value_compressor.compress(values_flush)

        if self.key_quantized is None:
            self.key_quantized = new_key_q
            self.value_quantized = new_val_q
        else:
            # Concatenate along sequence dimension
            self.key_quantized = CompressedKeys(
                indices=torch.cat([self.key_quantized.indices, new_key_q.indices], dim=-2),
                norms=torch.cat([self.key_quantized.norms, new_key_q.norms], dim=-1),
                qjl_signs=torch.cat([self.key_quantized.qjl_signs, new_key_q.qjl_signs], dim=-2),
                residual_norms=torch.cat([self.key_quantized.residual_norms, new_key_q.residual_norms], dim=-1),
                original_dtype=new_key_q.original_dtype,
            )
            self.value_quantized = CompressedValues(
                indices=torch.cat([self.value_quantized.indices, new_val_q.indices], dim=-2),
                norms=torch.cat([self.value_quantized.norms, new_val_q.norms], dim=-1),
                original_dtype=new_val_q.original_dtype,
            )

    def attention_scores(self, query: torch.Tensor, scale: float = None) -> torch.Tensor:
        """Compute attention logits using asymmetric estimator.

        Args:
            query: (batch, n_heads, n_q, head_dim)
            scale: attention scale factor (default: 1/sqrt(head_dim))

        Returns:
            scores: (batch, n_heads, n_q, seq_len)
        """
        import math
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores_parts = []

        # Quantized portion - use asymmetric estimator
        if self.key_quantized is not None:
            scores_quant = self.key_compressor.asymmetric_attention_scores(query, self.key_quantized)
            scores_parts.append(scores_quant * scale)

        # Buffer portion (full precision)
        if self.key_buffer is not None:
            scores_buf = torch.matmul(query, self.key_buffer.transpose(-2, -1))
            scores_parts.append(scores_buf * scale)

        return torch.cat(scores_parts, dim=-1)

    def attend(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute attention output: out = softmax(scores) @ values.

        Args:
            attn_weights: (batch, n_heads, n_q, seq_len) - already softmaxed

        Returns:
            output: (batch, n_heads, n_q, head_dim)
        """
        output_parts = []
        col_offset = 0

        # Quantized values
        if self.value_quantized is not None:
            n_quant = self.value_quantized.indices.shape[-2]
            w_quant = attn_weights[..., col_offset:col_offset + n_quant]
            v_dequant = self.value_compressor.decompress(self.value_quantized)
            output_parts.append(torch.matmul(w_quant, v_dequant))
            col_offset += n_quant

        # Buffer values (full precision)
        if self.value_buffer is not None:
            n_buf = self.value_buffer.shape[-2]
            w_buf = attn_weights[..., col_offset:col_offset + n_buf]
            output_parts.append(torch.matmul(w_buf, self.value_buffer))

        return sum(output_parts)

    def memory_bytes(self) -> dict:
        """Estimate memory usage of the cache."""
        info = {"quantized_keys": 0, "quantized_values": 0, "buffer": 0, "total": 0}

        if self.key_quantized is not None:
            info["quantized_keys"] += self.key_quantized.indices.nelement()
            info["quantized_keys"] += self.key_quantized.qjl_signs.nelement()
            info["quantized_keys"] += self.key_quantized.residual_norms.nelement() * 2
            info["quantized_keys"] += self.key_quantized.norms.nelement() * 2

        if self.value_quantized is not None:
            info["quantized_values"] += self.value_quantized.indices.nelement()
            info["quantized_values"] += self.value_quantized.norms.nelement() * 2

        if self.key_buffer is not None:
            info["buffer"] += self.key_buffer.nelement() * 2  # fp16
        if self.value_buffer is not None:
            info["buffer"] += self.value_buffer.nelement() * 2

        info["total"] = info["quantized_keys"] + info["quantized_values"] + info["buffer"]
        return info

    def get_seq_length(self) -> int:
        """Return current sequence length."""
        return self.seq_len

    def reset(self):
        """Clear all state."""
        self.seq_len = 0
        self.key_quantized = None
        self.value_quantized = None
        self.key_buffer = None
        self.value_buffer = None

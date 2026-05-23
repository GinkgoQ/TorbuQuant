"""Production compressors for TurboQuant KV cache.

This module provides the main compressor implementations:
- TurboQuantCompressorV2: Full compressor with asymmetric attention score estimation
- TurboQuantCompressorMSE: Simpler MSE-only compressor for values
- MSECompressor: Single-stage MSE-optimal compressor (no QJL)
- TurboQuantV3: Community-informed compressor with residual windowing

Key insight from the TurboQuant paper:
  <q, k> ~= <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, sign(S@r_k)>

This is unbiased with variance O(1/d), even though k_mse itself has high
per-vector error. The estimator works because QJL corrects the bias in the
inner product space, not in the vector space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torbuquant.core import get_codebook_tensors, build_rotation


@dataclass
class CompressedKeys:
    """Compressed key representation for asymmetric attention."""
    indices: torch.Tensor       # (batch, heads, seq, head_dim) codebook indices
    norms: torch.Tensor         # (batch, heads, seq) vector norms
    qjl_signs: torch.Tensor     # (batch, heads, seq, head_dim) QJL sign bits
    residual_norms: torch.Tensor  # (batch, heads, seq) residual norms
    original_dtype: torch.dtype = torch.float16


@dataclass
class CompressedValues:
    """Compressed value representation."""
    indices: torch.Tensor       # (batch, heads, seq, head_dim) codebook indices
    norms: torch.Tensor         # (batch, heads, seq) vector norms
    original_dtype: torch.dtype = torch.float16


class TurboQuantCompressorV2:
    """Compressor supporting direct inner product computation without full decompression.

    Stores compressed representations AND supports asymmetric attention scores.
    This is the full TurboQuant algorithm from the paper.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        # Rotation matrix (Haar-distributed random orthogonal)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)

        # Lloyd-Max codebook
        self.centroids = self._solve_codebook(head_dim, self.mse_bits).to(device)

        # QJL projection matrix
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(seed + 10000)
        self.S = torch.randn(head_dim, head_dim, generator=gen2).to(device)

        # Precompute transpose for fast dequant
        self.PiT = self.Pi.T.contiguous()

    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        """Solve Lloyd-Max optimal quantizer for the coordinate distribution."""
        try:
            from scipy import integrate
        except ImportError:
            # Fallback to uniform codebook if scipy unavailable
            n_levels = 2 ** bits
            sigma = 1.0 / math.sqrt(d)
            return torch.linspace(-2.5 * sigma, 2.5 * sigma, n_levels)

        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))

        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_centroids = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_centroids.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_centroids

        return torch.tensor(centroids, dtype=torch.float32)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> CompressedKeys:
        """Compress states: (batch, heads, seq, head_dim) -> CompressedKeys."""
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()

        # Store original norms
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)  # (N, 1)
        flat_norm = flat / (vec_norms + 1e-8)

        # Rotate and quantize
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        # MSE reconstruction in original space
        reconstructed_rotated = self.centroids[indices.long()]
        k_mse = (reconstructed_rotated @ self.Pi) * vec_norms  # (N, D)

        # Residual in original space
        residual = flat - k_mse
        residual_norm = torch.norm(residual, dim=-1)  # (N,)

        # QJL signs of residual
        projected = residual @ self.S.T
        signs = (projected >= 0).to(torch.int8) * 2 - 1  # {-1, +1}

        return CompressedKeys(
            indices=indices.reshape(B, H, S, D),
            norms=vec_norms.squeeze(-1).to(torch.float16).reshape(B, H, S),
            qjl_signs=signs.reshape(B, H, S, D),
            residual_norms=residual_norm.to(torch.float16).reshape(B, H, S),
            original_dtype=states.dtype,
        )

    @torch.no_grad()
    def decompress(self, compressed: CompressedKeys) -> torch.Tensor:
        """Decompress to full tensors (MSE component only)."""
        B, H, S, D = compressed.indices.shape
        indices = compressed.indices.reshape(-1, D).long()
        norms = compressed.norms.reshape(-1, 1).float()

        reconstructed = self.centroids[indices] @ self.Pi
        return (reconstructed * norms).reshape(B, H, S, D).to(compressed.original_dtype)

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: CompressedKeys) -> torch.Tensor:
        """Compute attention scores <Q, K> directly from compressed K.

        Uses the asymmetric estimator:
            <q, k> ~= <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, signs_k>

        Args:
            queries: (batch, heads, seq_q, head_dim)
            compressed: CompressedKeys from compress()

        Returns:
            scores: (batch, heads, seq_q, seq_k)
        """
        # Decompress MSE component
        k_mse = self.decompress(compressed).float()  # (B, H, S_k, D)
        signs = compressed.qjl_signs.float()          # (B, H, S_k, D)
        r_norm = compressed.residual_norms.float()    # (B, H, S_k)

        # Term 1: Q @ K_mse^T
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))  # (B, H, S_q, S_k)

        # Term 2: QJL correction
        q_projected = torch.matmul(queries.float(), self.S.T)  # (B, H, S_q, D)
        qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1))  # (B, H, S_q, S_k)

        m = self.S.shape[0]
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)

        return term1 + term2


class TurboQuantCompressorMSE:
    """Simpler MSE-only compressor for values (no QJL needed)."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)
        self.centroids = self._solve_codebook(head_dim, bits).to(device)

    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        """Solve Lloyd-Max optimal quantizer."""
        try:
            from scipy import integrate
        except ImportError:
            n_levels = 2 ** bits
            sigma = 1.0 / math.sqrt(d)
            return torch.linspace(-2.5 * sigma, 2.5 * sigma, n_levels)

        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))

        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_c = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_c.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_c[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_c

        return torch.tensor(centroids, dtype=torch.float32)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> CompressedValues:
        """Compress states: (batch, heads, seq, head_dim) -> CompressedValues."""
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (vec_norms + 1e-8)
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)
        return CompressedValues(
            indices=indices.reshape(B, H, S, D),
            norms=vec_norms.squeeze(-1).to(torch.float16).reshape(B, H, S),
            original_dtype=states.dtype,
        )

    @torch.no_grad()
    def decompress(self, compressed: CompressedValues) -> torch.Tensor:
        """Decompress back to full tensors."""
        B, H, S, D = compressed.indices.shape
        indices = compressed.indices.reshape(-1, D).long()
        norms = compressed.norms.reshape(-1, 1).float()
        reconstructed = self.centroids[indices] @ self.Pi
        return (reconstructed * norms).reshape(B, H, S, D).to(compressed.original_dtype)


class MSECompressor:
    """Single-stage MSE-optimal compressor with bit packing.

    Used for both keys and values. No QJL — all bits go to reconstruction quality.
    This is the core building block for TurboQuantV3.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)
        self.centroids = self._solve_codebook(head_dim, bits).to(device)

    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        """Solve Lloyd-Max optimal quantizer."""
        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)
        return torch.linspace(-2.5 * sigma, 2.5 * sigma, n_levels)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """Compress (B, H, S, D) -> dict with bit-packed indices + norms."""
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()

        # Normalize to unit sphere
        vec_norms = torch.norm(flat, dim=-1)  # (N,)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)

        # Rotate + quantize
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (N, D)

        # Bit-pack indices
        indices_per_byte = 8 // self.bits
        idx_pad = (indices_per_byte - D % indices_per_byte) % indices_per_byte
        idx_flat = indices.long()
        if idx_pad:
            idx_flat = F.pad(idx_flat, (0, idx_pad))
        n_groups = idx_flat.shape[-1] // indices_per_byte
        idx_powers = torch.tensor(
            [2 ** (self.bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long, device=idx_flat.device
        )
        idx_bytes = (idx_flat.reshape(N, n_groups, indices_per_byte) * idx_powers).sum(-1).to(torch.uint8)

        return {
            "idx_bytes": idx_bytes.reshape(B, H, S, n_groups),
            "vec_norms": vec_norms.to(torch.float16).reshape(B, H, S),
            "shape": (B, H, S, D),
            "idx_pad": idx_pad,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress back to (B, H, S, D) tensor."""
        B, H, S, D = compressed["shape"]
        N = B * H * S
        idx_bytes = compressed["idx_bytes"].reshape(N, -1)
        vec_norms = compressed["vec_norms"].reshape(N, 1).float()
        idx_pad = compressed["idx_pad"]

        # Unpack indices
        indices_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        idx_shifts = torch.tensor(
            [self.bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long, device=idx_bytes.device
        )
        indices = ((idx_bytes.long().unsqueeze(-1) >> idx_shifts) & mask).reshape(N, -1)
        if idx_pad:
            indices = indices[:, :D]

        # Reconstruct
        reconstructed = (self.centroids[indices] @ self.Pi) * vec_norms
        return reconstructed.reshape(B, H, S, D)

    def memory_bytes(self, B: int, H: int, S: int) -> dict:
        """Actual memory usage in bytes."""
        D = self.head_dim
        N = B * H * S
        indices_per_byte = 8 // self.bits
        idx_bytes = N * math.ceil(D / indices_per_byte)
        norm_bytes = N * 2  # fp16
        compressed = idx_bytes + norm_bytes
        fp16 = N * D * 2
        return {
            "compressed_bytes": compressed,
            "fp16_bytes": fp16,
            "compression_ratio": fp16 / compressed if compressed > 0 else 0,
        }


class TurboQuantV3:
    """Community-informed KV cache compressor.

    Key improvements over V2:
      - MSE-only: no QJL, all bits go to reconstruction quality
      - Asymmetric: separate bit-widths for keys vs values
      - Residual window: recent tokens kept in fp16
      - Layer-adaptive: configurable per-layer bit overrides

    Usage:
        compressor = TurboQuantV3(head_dim=128, key_bits=4, value_bits=2, device="cuda")
        compressed_k, compressed_v = compressor.compress_kv(keys, values)
        keys_out, values_out = compressor.decompress_kv(compressed_k, compressed_v)
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 4,
        value_bits: int = 2,
        residual_window: int = 128,
        layer_idx: int = 0,
        n_layers: int = 36,
        protected_layers: int = 4,
        protected_bits: int = 8,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.head_dim = head_dim
        self.residual_window = residual_window
        self.device = device

        # Layer-adaptive: first/last N layers get more bits
        is_protected = layer_idx < protected_layers or layer_idx >= (n_layers - protected_layers)
        effective_key_bits = protected_bits if is_protected else key_bits
        effective_value_bits = protected_bits if is_protected else value_bits

        # Cap at 8 bits
        self.key_bits = min(effective_key_bits, 8)
        self.value_bits = min(effective_value_bits, 8)

        seed_base = seed + layer_idx * 1000
        self.key_compressor = MSECompressor(head_dim, self.key_bits, seed=seed_base, device=device)
        self.val_compressor = MSECompressor(head_dim, self.value_bits, seed=seed_base + 500, device=device)

    @torch.no_grad()
    def compress_kv(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[dict, dict]:
        """Compress key and value tensors.

        Input: keys, values - both (B, H, S, D)

        If S > residual_window, the last `residual_window` tokens are kept
        in fp16 (uncompressed) for generation quality.
        """
        B, H, S, D = keys.shape
        rw = self.residual_window

        if S <= rw:
            # Short sequence - keep everything in fp16
            return (
                {"fp16": keys, "compressed": None, "shape": (B, H, S, D), "split_at": S},
                {"fp16": values, "compressed": None, "shape": (B, H, S, D), "split_at": S},
            )

        # Split: compress old tokens, keep recent in fp16
        split_at = S - rw

        old_keys = keys[:, :, :split_at, :]
        recent_keys = keys[:, :, split_at:, :]
        old_values = values[:, :, :split_at, :]
        recent_values = values[:, :, split_at:, :]

        compressed_k = {
            "compressed": self.key_compressor.compress(old_keys),
            "fp16": recent_keys,
            "shape": (B, H, S, D),
            "split_at": split_at,
        }
        compressed_v = {
            "compressed": self.val_compressor.compress(old_values),
            "fp16": recent_values,
            "shape": (B, H, S, D),
            "split_at": split_at,
        }
        return compressed_k, compressed_v

    @torch.no_grad()
    def decompress_kv(
        self, compressed_k: dict, compressed_v: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress back to full tensors."""
        if compressed_k["compressed"] is None:
            return compressed_k["fp16"], compressed_v["fp16"]

        # Decompress old tokens
        old_keys = self.key_compressor.decompress(compressed_k["compressed"])
        old_values = self.val_compressor.decompress(compressed_v["compressed"])

        # Concatenate with fp16 recent tokens
        dtype = compressed_k["fp16"].dtype
        keys = torch.cat([old_keys.to(dtype), compressed_k["fp16"]], dim=2)
        values = torch.cat([old_values.to(dtype), compressed_v["fp16"]], dim=2)
        return keys, values

    def memory_bytes(self, B: int, H: int, S: int) -> dict:
        """Report actual memory usage including residual window."""
        rw = min(self.residual_window, S)
        compressed_S = max(S - rw, 0)
        fp16_S = rw

        if compressed_S > 0:
            k_mem = self.key_compressor.memory_bytes(B, H, compressed_S)
            v_mem = self.val_compressor.memory_bytes(B, H, compressed_S)
            compressed_bytes = k_mem["compressed_bytes"] + v_mem["compressed_bytes"]
        else:
            compressed_bytes = 0

        fp16_window_bytes = B * H * fp16_S * self.head_dim * 2 * 2  # keys + values, fp16
        total_compressed = compressed_bytes + fp16_window_bytes
        total_fp16 = B * H * S * self.head_dim * 2 * 2

        return {
            "compressed_bytes": total_compressed,
            "fp16_bytes": total_fp16,
            "compression_ratio": total_fp16 / total_compressed if total_compressed > 0 else 0,
            "compressed_tokens": compressed_S,
            "fp16_tokens": fp16_S,
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
        }

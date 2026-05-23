"""Triton kernels for contiguous compressed decode blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import state depends on environment
    triton = None
    tl = None

from torbuquant.core.rotation import rotate_forward
from torbuquant.kv.cache import CompressedKVBlock
from torbuquant.kv.keys import DenseKeyData, QuantizedKeyData, TurboKeyData


KernelName = Literal[
    "score_k16",
    "score_k8",
    "score_k4",
    "weighted_v4",
    "sparse_weighted_v4",
    "decode_k16v4",
    "decode_k8v4",
    "fused_decode_k4v4",
]


class UnsupportedFormatError(RuntimeError):
    """Raised when a requested compressed attention kernel is unavailable."""


@dataclass
class KernelCounters:
    score_calls: int = 0
    v_accumulation_calls: int = 0
    fused_calls: int = 0
    fallback_calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class KernelReport:
    name: KernelName
    k_format: str
    v_format: str
    q_len: int
    kv_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    counters: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KernelResult:
    output: torch.Tensor
    scores: torch.Tensor | None
    report: KernelReport


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _require_cuda(*tensors: torch.Tensor) -> None:
    if triton is None:
        raise UnsupportedFormatError("Triton is not importable")
    for tensor in tensors:
        if not tensor.is_cuda:
            raise UnsupportedFormatError("Triton kernels require CUDA tensors")


def _check_query(query: torch.Tensor) -> tuple[int, int, int]:
    if query.ndim != 3:
        raise ValueError("query must have shape (tokens, query_heads, head_dim)")
    return int(query.shape[0]), int(query.shape[1]), int(query.shape[2])


def _block_n(kv_len: int) -> int:
    if kv_len <= 32:
        return 32
    if kv_len <= 64:
        return 64
    return 128


if tl is not None:

    @triton.jit
    def _score_k16_kernel(
        Q,
        K,
        OUT,
        q_len: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        kv_len,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        stride_q_t,
        stride_q_h,
        stride_q_d,
        stride_k_n,
        stride_k_h,
        stride_k_d,
        stride_o_t,
        stride_o_h,
        stride_o_n,
        BLOCK_N: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)
        gqa: tl.constexpr = num_q_heads // num_kv_heads
        kv_h = pid_h // gqa
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d in range(0, head_dim):
            q = tl.load(Q + pid_t * stride_q_t + pid_h * stride_q_h + d * stride_q_d).to(tl.float32)
            k = tl.load(
                K + offs_n * stride_k_n + kv_h * stride_k_h + d * stride_k_d,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            acc += q * k
        tl.store(
            OUT + pid_t * stride_o_t + pid_h * stride_o_h + offs_n * stride_o_n,
            acc * scale,
            mask=mask_n,
        )

    @triton.jit
    def _score_k8_kernel(
        Q,
        K_DATA,
        K_SCALES,
        K_ZEROS,
        OUT,
        q_len: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        kv_len,
        head_dim: tl.constexpr,
        group_size: tl.constexpr,
        scale: tl.constexpr,
        stride_q_t,
        stride_q_h,
        stride_q_d,
        stride_k_n,
        stride_k_h,
        stride_k_d,
        stride_s_n,
        stride_s_h,
        stride_s_g,
        stride_z_n,
        stride_z_h,
        stride_z_g,
        stride_o_t,
        stride_o_h,
        stride_o_n,
        BLOCK_N: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)
        gqa: tl.constexpr = num_q_heads // num_kv_heads
        kv_h = pid_h // gqa
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d in range(0, head_dim):
            group = d // group_size
            q = tl.load(Q + pid_t * stride_q_t + pid_h * stride_q_h + d * stride_q_d).to(tl.float32)
            idx = tl.load(
                K_DATA + offs_n * stride_k_n + kv_h * stride_k_h + d * stride_k_d,
                mask=mask_n,
                other=0,
            ).to(tl.float32)
            s = tl.load(
                K_SCALES + offs_n * stride_s_n + kv_h * stride_s_h + group * stride_s_g,
                mask=mask_n,
                other=1.0,
            ).to(tl.float32)
            z = tl.load(
                K_ZEROS + offs_n * stride_z_n + kv_h * stride_z_h + group * stride_z_g,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            acc += q * (idx * s + z)
        tl.store(
            OUT + pid_t * stride_o_t + pid_h * stride_o_h + offs_n * stride_o_n,
            acc * scale,
            mask=mask_n,
        )

    @triton.jit
    def _score_k4_kernel(
        Q_ROT,
        K_IDX,
        K_NORMS,
        CENTROIDS,
        OUT,
        q_len: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        kv_len,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        stride_q_t,
        stride_q_h,
        stride_q_d,
        stride_k_n,
        stride_k_h,
        stride_k_b,
        stride_norm_n,
        stride_norm_h,
        stride_o_t,
        stride_o_h,
        stride_o_n,
        BLOCK_N: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)
        gqa: tl.constexpr = num_q_heads // num_kv_heads
        kv_h = pid_h // gqa
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len
        dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        norm2 = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(0, head_dim // 2):
            packed = tl.load(
                K_IDX + offs_n * stride_k_n + kv_h * stride_k_h + byte_idx * stride_k_b,
                mask=mask_n,
                other=0,
            ).to(tl.int32)
            lo = packed & 15
            hi = (packed >> 4) & 15
            c0 = tl.load(CENTROIDS + lo).to(tl.float32)
            c1 = tl.load(CENTROIDS + hi).to(tl.float32)
            q0 = tl.load(Q_ROT + pid_t * stride_q_t + pid_h * stride_q_h + (byte_idx * 2) * stride_q_d).to(tl.float32)
            q1 = tl.load(Q_ROT + pid_t * stride_q_t + pid_h * stride_q_h + (byte_idx * 2 + 1) * stride_q_d).to(tl.float32)
            dot += q0 * c0 + q1 * c1
            norm2 += c0 * c0 + c1 * c1
        norms = tl.load(
            K_NORMS + offs_n * stride_norm_n + kv_h * stride_norm_h,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        score = dot * norms / tl.sqrt(tl.maximum(norm2, 1.0e-20)) * scale
        tl.store(
            OUT + pid_t * stride_o_t + pid_h * stride_o_h + offs_n * stride_o_n,
            score,
            mask=mask_n,
        )

    @triton.jit
    def _weighted_v4_kernel(
        WEIGHTS,
        V_DATA,
        V_SCALES,
        V_ZEROS,
        OUT,
        q_len: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        kv_len,
        head_dim: tl.constexpr,
        group_size: tl.constexpr,
        stride_w_t,
        stride_w_h,
        stride_w_n,
        stride_v_n,
        stride_v_h,
        stride_v_b,
        stride_s_n,
        stride_s_h,
        stride_s_g,
        stride_z_n,
        stride_z_h,
        stride_z_g,
        stride_o_t,
        stride_o_h,
        stride_o_d,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        SPARSE_V: tl.constexpr,
        SPARSE_V_THRESHOLD: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_d = tl.program_id(2)
        gqa: tl.constexpr = num_q_heads // num_kv_heads
        kv_h = pid_h // gqa
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for block_idx in range(0, NUM_BLOCKS):
            offs_n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < kv_len
            w = tl.load(
                WEIGHTS + pid_t * stride_w_t + pid_h * stride_w_h + offs_n * stride_w_n,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            active_n = mask_n
            if SPARSE_V:
                active_n = mask_n & (w >= SPARSE_V_THRESHOLD)
                w = tl.where(active_n, w, 0.0)
            packed = tl.load(
                V_DATA
                + offs_n[:, None] * stride_v_n
                + kv_h * stride_v_h
                + (offs_d[None, :] // 2) * stride_v_b,
                mask=active_n[:, None] & mask_d[None, :],
                other=0,
            ).to(tl.int32)
            idx = tl.where((offs_d[None, :] & 1) == 0, packed & 15, (packed >> 4) & 15).to(tl.float32)
            group = offs_d // group_size
            s = tl.load(
                V_SCALES
                + offs_n[:, None] * stride_s_n
                + kv_h * stride_s_h
                + group[None, :] * stride_s_g,
                mask=active_n[:, None] & mask_d[None, :],
                other=1.0,
            ).to(tl.float32)
            z = tl.load(
                V_ZEROS
                + offs_n[:, None] * stride_z_n
                + kv_h * stride_z_h
                + group[None, :] * stride_z_g,
                mask=active_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            values = idx * s + z
            acc += tl.sum(w[:, None] * values, axis=0)
        tl.store(
            OUT + pid_t * stride_o_t + pid_h * stride_o_h + offs_d * stride_o_d,
            acc,
            mask=mask_d,
        )

    @triton.jit
    def _fused_decode_k4v4_kernel(
        Q_ROT,
        K_IDX,
        K_NORMS,
        CENTROIDS,
        V_DATA,
        V_SCALES,
        V_ZEROS,
        OUT,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        kv_len,
        head_dim: tl.constexpr,
        group_size: tl.constexpr,
        scale: tl.constexpr,
        stride_q_h,
        stride_q_d,
        stride_k_n,
        stride_k_h,
        stride_k_b,
        stride_norm_n,
        stride_norm_h,
        stride_v_n,
        stride_v_h,
        stride_v_b,
        stride_s_n,
        stride_s_h,
        stride_s_g,
        stride_z_n,
        stride_z_h,
        stride_z_g,
        stride_o_h,
        stride_o_d,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_d = tl.program_id(1)
        gqa: tl.constexpr = num_q_heads // num_kv_heads
        kv_h = pid_h // gqa
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        m_i = tl.full([1], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for block_idx in range(0, NUM_BLOCKS):
            offs_n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < kv_len
            dot = tl.zeros([BLOCK_N], dtype=tl.float32)
            norm2 = tl.zeros([BLOCK_N], dtype=tl.float32)
            for byte_idx in range(0, head_dim // 2):
                packed_k = tl.load(
                    K_IDX + offs_n * stride_k_n + kv_h * stride_k_h + byte_idx * stride_k_b,
                    mask=mask_n,
                    other=0,
                ).to(tl.int32)
                lo = packed_k & 15
                hi = (packed_k >> 4) & 15
                c0 = tl.load(CENTROIDS + lo).to(tl.float32)
                c1 = tl.load(CENTROIDS + hi).to(tl.float32)
                q0 = tl.load(Q_ROT + pid_h * stride_q_h + (byte_idx * 2) * stride_q_d).to(tl.float32)
                q1 = tl.load(Q_ROT + pid_h * stride_q_h + (byte_idx * 2 + 1) * stride_q_d).to(tl.float32)
                dot += q0 * c0 + q1 * c1
                norm2 += c0 * c0 + c1 * c1
            norms = tl.load(
                K_NORMS + offs_n * stride_norm_n + kv_h * stride_norm_h,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            scores = dot * norms / tl.sqrt(tl.maximum(norm2, 1.0e-20)) * scale
            scores = tl.where(mask_n, scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha
            packed_v = tl.load(
                V_DATA
                + offs_n[:, None] * stride_v_n
                + kv_h * stride_v_h
                + (offs_d[None, :] // 2) * stride_v_b,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0,
            ).to(tl.int32)
            idx = tl.where((offs_d[None, :] & 1) == 0, packed_v & 15, (packed_v >> 4) & 15).to(tl.float32)
            group = offs_d // group_size
            s = tl.load(
                V_SCALES
                + offs_n[:, None] * stride_s_n
                + kv_h * stride_s_h
                + group[None, :] * stride_s_g,
                mask=mask_n[:, None] & mask_d[None, :],
                other=1.0,
            ).to(tl.float32)
            z = tl.load(
                V_ZEROS
                + offs_n[:, None] * stride_z_n
                + kv_h * stride_z_h
                + group[None, :] * stride_z_g,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            values = idx * s + z
            acc += tl.sum(p[:, None] * values, axis=0)
            m_i = m_new
        tl.store(
            OUT + pid_h * stride_o_h + offs_d * stride_o_d,
            acc / l_i,
            mask=mask_d,
        )


def score_k16(query: torch.Tensor, keys: torch.Tensor, *, num_kv_heads: int, scale: float) -> torch.Tensor:
    q_len, num_q_heads, head_dim = _check_query(query)
    _require_cuda(query, keys)
    query_c = query.contiguous()
    keys_c = keys.contiguous()
    kv_len = int(keys.shape[0])
    out = torch.empty(q_len, num_q_heads, kv_len, device=query.device, dtype=torch.float32)
    block_n = _block_n(kv_len)
    grid = (q_len, num_q_heads, triton.cdiv(kv_len, block_n))
    _score_k16_kernel[grid](
        query_c,
        keys_c,
        out,
        q_len,
        num_q_heads,
        num_kv_heads,
        kv_len,
        head_dim,
        float(scale),
        query_c.stride(0),
        query_c.stride(1),
        query_c.stride(2),
        keys_c.stride(0),
        keys_c.stride(1),
        keys_c.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=block_n,
    )
    return out


def score_k8(query: torch.Tensor, key_data: Any, *, num_kv_heads: int, scale: float) -> torch.Tensor:
    q_len, num_q_heads, head_dim = _check_query(query)
    _require_cuda(query, key_data.data, key_data.scales, key_data.zeros)
    query_c = query.contiguous()
    data_c = key_data.data.contiguous()
    scales_c = key_data.scales.contiguous()
    zeros_c = key_data.zeros.contiguous()
    kv_len = int(key_data.data.shape[0])
    out = torch.empty(q_len, num_q_heads, kv_len, device=query.device, dtype=torch.float32)
    block_n = _block_n(kv_len)
    grid = (q_len, num_q_heads, triton.cdiv(kv_len, block_n))
    _score_k8_kernel[grid](
        query_c,
        data_c,
        scales_c,
        zeros_c,
        out,
        q_len,
        num_q_heads,
        num_kv_heads,
        kv_len,
        head_dim,
        key_data.group_size,
        float(scale),
        query_c.stride(0),
        query_c.stride(1),
        query_c.stride(2),
        data_c.stride(0),
        data_c.stride(1),
        data_c.stride(2),
        scales_c.stride(0),
        scales_c.stride(1),
        scales_c.stride(2),
        zeros_c.stride(0),
        zeros_c.stride(1),
        zeros_c.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=block_n,
    )
    return out


def score_k4(
    query: torch.Tensor,
    block: CompressedKVBlock,
    *,
    num_kv_heads: int,
    scale: float,
    quantizer: Any,
) -> torch.Tensor:
    q_len, num_q_heads, head_dim = _check_query(query)
    if not isinstance(block.key, TurboKeyData):
        raise UnsupportedFormatError("score_k4 requires K4 key payload")
    data = block.key.data
    q_rot = rotate_forward(query.reshape(-1, head_dim), quantizer.rotation).reshape(q_len, num_q_heads, head_dim)
    _require_cuda(q_rot, data.indices, data.norms, quantizer.centroids)
    q_rot_c = q_rot.contiguous()
    idx_c = data.indices.contiguous()
    norms_c = data.norms.contiguous()
    centroids_c = quantizer.centroids.contiguous()
    kv_len = int(data.indices.shape[0])
    out = torch.empty(q_len, num_q_heads, kv_len, device=query.device, dtype=torch.float32)
    block_n = _block_n(kv_len)
    grid = (q_len, num_q_heads, triton.cdiv(kv_len, block_n))
    _score_k4_kernel[grid](
        q_rot_c,
        idx_c,
        norms_c,
        centroids_c,
        out,
        q_len,
        num_q_heads,
        num_kv_heads,
        kv_len,
        head_dim,
        float(scale),
        q_rot_c.stride(0),
        q_rot_c.stride(1),
        q_rot_c.stride(2),
        idx_c.stride(0),
        idx_c.stride(1),
        idx_c.stride(2),
        norms_c.stride(0),
        norms_c.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=block_n,
    )
    return out


def weighted_v4(
    weights: torch.Tensor,
    value_data: Any,
    *,
    num_kv_heads: int,
    sparse_v_threshold: float | None = None,
) -> torch.Tensor:
    if weights.ndim != 3:
        raise ValueError("weights must have shape (tokens, query_heads, kv_tokens)")
    if sparse_v_threshold is not None and sparse_v_threshold < 0:
        raise ValueError("sparse_v_threshold must be non-negative")
    q_len, num_q_heads, kv_len = [int(x) for x in weights.shape]
    _require_cuda(weights, value_data.data, value_data.scales, value_data.zeros)
    weights_c = weights.contiguous()
    data_c = value_data.data.contiguous()
    scales_c = value_data.scales.contiguous()
    zeros_c = value_data.zeros.contiguous()
    head_dim = int(value_data.dim)
    out = torch.empty(q_len, num_q_heads, head_dim, device=weights.device, dtype=torch.float32)
    block_n = _block_n(kv_len)
    block_d = 32
    grid = (q_len, num_q_heads, triton.cdiv(head_dim, block_d))
    _weighted_v4_kernel[grid](
        weights_c,
        data_c,
        scales_c,
        zeros_c,
        out,
        q_len,
        num_q_heads,
        num_kv_heads,
        kv_len,
        head_dim,
        value_data.group_size,
        weights_c.stride(0),
        weights_c.stride(1),
        weights_c.stride(2),
        data_c.stride(0),
        data_c.stride(1),
        data_c.stride(2),
        scales_c.stride(0),
        scales_c.stride(1),
        scales_c.stride(2),
        zeros_c.stride(0),
        zeros_c.stride(1),
        zeros_c.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        NUM_BLOCKS=triton.cdiv(kv_len, block_n),
        SPARSE_V=sparse_v_threshold is not None,
        SPARSE_V_THRESHOLD=0.0 if sparse_v_threshold is None else float(sparse_v_threshold),
    )
    return out


def fused_decode_k4v4(
    query: torch.Tensor,
    block: CompressedKVBlock,
    *,
    num_kv_heads: int,
    scale: float,
    quantizer: Any,
) -> torch.Tensor:
    q_len, num_q_heads, head_dim = _check_query(query)
    if q_len != 1:
        raise UnsupportedFormatError("fused_decode_k4v4 supports one-token decode")
    if not isinstance(block.key, TurboKeyData) or block.v_format != "V4":
        raise UnsupportedFormatError("fused_decode_k4v4 requires K4V4 block")
    k_data = block.key.data
    v_data = block.value
    q_rot = rotate_forward(query.reshape(num_q_heads, head_dim), quantizer.rotation)
    _require_cuda(q_rot, k_data.indices, k_data.norms, quantizer.centroids, v_data.data, v_data.scales, v_data.zeros)
    q_rot_c = q_rot.contiguous()
    k_idx_c = k_data.indices.contiguous()
    k_norms_c = k_data.norms.contiguous()
    centroids_c = quantizer.centroids.contiguous()
    v_data_c = v_data.data.contiguous()
    v_scales_c = v_data.scales.contiguous()
    v_zeros_c = v_data.zeros.contiguous()
    kv_len = int(k_data.indices.shape[0])
    out = torch.empty(num_q_heads, head_dim, device=query.device, dtype=torch.float32)
    block_n = _block_n(kv_len)
    block_d = 32
    grid = (num_q_heads, triton.cdiv(head_dim, block_d))
    _fused_decode_k4v4_kernel[grid](
        q_rot_c,
        k_idx_c,
        k_norms_c,
        centroids_c,
        v_data_c,
        v_scales_c,
        v_zeros_c,
        out,
        num_q_heads,
        num_kv_heads,
        kv_len,
        head_dim,
        v_data.group_size,
        float(scale),
        q_rot_c.stride(0),
        q_rot_c.stride(1),
        k_idx_c.stride(0),
        k_idx_c.stride(1),
        k_idx_c.stride(2),
        k_norms_c.stride(0),
        k_norms_c.stride(1),
        v_data_c.stride(0),
        v_data_c.stride(1),
        v_data_c.stride(2),
        v_scales_c.stride(0),
        v_scales_c.stride(1),
        v_scales_c.stride(2),
        v_zeros_c.stride(0),
        v_zeros_c.stride(1),
        v_zeros_c.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        NUM_BLOCKS=triton.cdiv(kv_len, block_n),
    )
    return out.unsqueeze(0)


def decode_block(
    query: torch.Tensor,
    block: CompressedKVBlock,
    *,
    num_kv_heads: int,
    scale: float,
    quantizer: Any | None = None,
    counters: KernelCounters | None = None,
    sparse_v_threshold: float | None = None,
) -> KernelResult:
    q_len, num_q_heads, head_dim = _check_query(query)
    counters = KernelCounters() if counters is None else counters
    if block.v_format != "V4":
        counters.fallback_calls += 1
        raise UnsupportedFormatError(f"unsupported value format for Triton decode: {block.v_format}")
    if isinstance(block.key, TurboKeyData):
        if sparse_v_threshold is not None:
            counters.fallback_calls += 1
            raise UnsupportedFormatError("sparse V for K4 fused decode requires a two-pass kernel")
        if quantizer is None:
            counters.fallback_calls += 1
            raise UnsupportedFormatError("K4 decode requires a quantizer")
        if q_len != 1:
            counters.fallback_calls += 1
            raise UnsupportedFormatError("K4V4 fused decode supports one-token decode")
        out = fused_decode_k4v4(
            query,
            block,
            num_kv_heads=num_kv_heads,
            scale=scale,
            quantizer=quantizer,
        )
        counters.fused_calls += 1
        return KernelResult(
            output=out,
            scores=None,
            report=_report("fused_decode_k4v4", block, query, num_kv_heads, counters),
        )
    elif isinstance(block.key, DenseKeyData):
        scores = score_k16(query, block.key.data, num_kv_heads=num_kv_heads, scale=scale)
        counters.score_calls += 1
    elif isinstance(block.key, QuantizedKeyData):
        scores = score_k8(query, block.key.data, num_kv_heads=num_kv_heads, scale=scale)
        counters.score_calls += 1
    else:
        counters.fallback_calls += 1
        raise UnsupportedFormatError(f"unsupported key payload: {type(block.key)!r}")
    weights = torch.softmax(scores.float(), dim=-1)
    out = weighted_v4(
        weights,
        block.value,
        num_kv_heads=num_kv_heads,
        sparse_v_threshold=sparse_v_threshold,
    )
    counters.v_accumulation_calls += 1
    name: KernelName = "decode_k16v4"
    if isinstance(block.key, QuantizedKeyData):
        name = "decode_k8v4"
    elif isinstance(block.key, TurboKeyData):
        name = "fused_decode_k4v4"
    return KernelResult(output=out, scores=scores, report=_report(name, block, query, num_kv_heads, counters))


def _report(
    name: KernelName,
    block: CompressedKVBlock,
    query: torch.Tensor,
    num_kv_heads: int,
    counters: KernelCounters,
) -> KernelReport:
    _q_len, num_q_heads, head_dim = _check_query(query)
    return KernelReport(
        name=name,
        k_format=block.k_format,
        v_format=block.v_format,
        q_len=int(query.shape[0]),
        kv_len=block.length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        counters=counters.to_dict(),
    )

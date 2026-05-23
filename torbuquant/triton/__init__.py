"""Triton kernel entry points."""

from torbuquant.triton.kernels import (
    KernelCounters,
    KernelName,
    KernelReport,
    KernelResult,
    UnsupportedFormatError,
    decode_block,
    fused_decode_k4v4,
    score_k16,
    score_k4,
    score_k8,
    triton_available,
    weighted_v4,
)
from torbuquant.triton.tq4_decode import decode_tq4_paged, unpack_tq_rows
from torbuquant.triton.tq4_update import pack_tq_rows, tq_row_bytes, write_tq4_kv

# Advanced fused attention kernels (Phase 3)
try:
    from torbuquant.triton.flash_attention_tq4_kv import triton_flash_attention_tq4_kv
    from torbuquant.triton.fused_paged_tq4_attention import fused_paged_tq4_decode
    _FUSED_ATTENTION_AVAILABLE = True
except ImportError:
    _FUSED_ATTENTION_AVAILABLE = False
    triton_flash_attention_tq4_kv = None
    fused_paged_tq4_decode = None

__all__ = [
    # Core kernels
    "KernelCounters",
    "KernelName",
    "KernelReport",
    "KernelResult",
    "UnsupportedFormatError",
    "decode_block",
    "decode_tq4_paged",
    "fused_decode_k4v4",
    "pack_tq_rows",
    "score_k16",
    "score_k4",
    "score_k8",
    "triton_available",
    "tq_row_bytes",
    "unpack_tq_rows",
    "weighted_v4",
    "write_tq4_kv",
    # Advanced fused attention (Phase 3)
    "triton_flash_attention_tq4_kv",
    "fused_paged_tq4_decode",
]

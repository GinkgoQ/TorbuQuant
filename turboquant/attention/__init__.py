"""Attention reference paths and hybrid attention."""

from turboquant.attention.reference import (
    AttentionLabel,
    AttentionReport,
    AttentionResult,
    SparseVReport,
    apply_causal_mask,
    default_scale,
    dense_attention,
    dense_scores,
    dense_sdpa_attention,
    dense_value_accumulation,
    diagnostic_dequant_attention,
    direct_qk_attention,
    direct_qk_scores,
    fallback_dense_attention,
    gqa_kv_heads,
    online_softmax,
    weighted_packed_v_accumulation,
)
from turboquant.attention.hybrid import (
    FlatCache,
    HybridAttentionResult,
    compute_hybrid_attention,
    compute_hybrid_attention_store,
    online_softmax_attention,
)

__all__ = [
    # Reference paths
    "AttentionLabel",
    "AttentionReport",
    "AttentionResult",
    "SparseVReport",
    "apply_causal_mask",
    "default_scale",
    "dense_attention",
    "dense_scores",
    "dense_sdpa_attention",
    "dense_value_accumulation",
    "diagnostic_dequant_attention",
    "direct_qk_attention",
    "direct_qk_scores",
    "fallback_dense_attention",
    "gqa_kv_heads",
    "online_softmax",
    "weighted_packed_v_accumulation",
    # Hybrid attention
    "FlatCache",
    "HybridAttentionResult",
    "compute_hybrid_attention",
    "compute_hybrid_attention_store",
    "online_softmax_attention",
]

"""Model-aware cache policy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


PresetName = Literal[
    "k8_v4",
    "k16_v4",
    "k4_v8",
    "k4_v4",
    "k4_v3",
    "k4_v2",
    "v_compress",
    "boundary_v",
    "recent_window",
    "sparse_v",
]
AutoBackend = Literal["dense", "torbuquant"]


@dataclass(frozen=True)
class BackendCapability:
    name: Literal["diagnostic", "hf", "vllm"]
    fused_decode: bool = False
    paged_cache: bool = False
    supports_v3: bool = False
    supports_sparse_v: bool = False


@dataclass(frozen=True)
class KVQuantPolicy:
    preset: str
    k_format: str
    v_format: str
    recent_window: int = 128
    boundary_tokens: int = 0
    preserve_first_layers: int = 0
    preserve_last_layers: int = 0
    sparse_v: bool = False
    sparse_v_threshold: float | None = None
    allow_v3: bool = False
    allow_k_low_bits: bool = False
    fallback_mode: str | None = None
    fallback_count: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutoPolicyInput:
    batch_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    context_tokens: int
    dtype_bytes: int = 2
    available_kv_memory_bytes: int | None = None
    model_id: str | None = None
    num_q_heads: int | None = None
    prefer_low_latency: bool = False
    allow_quantization: bool = True
    backend: BackendCapability = BackendCapability("vllm", fused_decode=True)


@dataclass(frozen=True)
class AutoPolicyDecision:
    backend: AutoBackend
    policy: KVQuantPolicy | None
    dense_bytes: int
    cache_bytes: int
    compression_ratio: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy"] = None if self.policy is None else self.policy.to_dict()
        return data


def estimate_dense_kv_bytes(
    *,
    batch_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    dtype_bytes: int = 2,
) -> int:
    if min(batch_size, num_layers, num_kv_heads, head_dim, context_tokens, dtype_bytes) <= 0:
        raise ValueError("KV byte estimate inputs must be positive")
    return int(batch_size * num_layers * num_kv_heads * context_tokens * head_dim * 2 * dtype_bytes)


def estimate_policy_kv_bytes(
    policy: KVQuantPolicy,
    *,
    batch_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    dtype_bytes: int = 2,
) -> int:
    if min(batch_size, num_layers, num_kv_heads, head_dim, context_tokens, dtype_bytes) <= 0:
        raise ValueError("KV byte estimate inputs must be positive")
    k_row = _k_row_bytes(policy.k_format, head_dim=head_dim, dtype_bytes=dtype_bytes)
    v_row = _v_row_bytes(policy.v_format, head_dim=head_dim, dtype_bytes=dtype_bytes)
    recent = min(max(0, policy.recent_window), context_tokens)
    boundary = min(max(0, policy.boundary_tokens), max(0, context_tokens - recent))
    compressed_tokens = max(0, context_tokens - recent - boundary)
    compressed_rows = int(batch_size * num_layers * num_kv_heads * compressed_tokens)
    dense_rows = int(batch_size * num_layers * num_kv_heads * (recent + boundary))
    compressed = compressed_rows * (k_row + v_row)
    dense_tail = dense_rows * head_dim * 2 * dtype_bytes
    return int(compressed + dense_tail)


def choose_auto_kv_policy(inputs: AutoPolicyInput) -> AutoPolicyDecision:
    dense = estimate_dense_kv_bytes(
        batch_size=inputs.batch_size,
        num_layers=inputs.num_layers,
        num_kv_heads=inputs.num_kv_heads,
        head_dim=inputs.head_dim,
        context_tokens=inputs.context_tokens,
        dtype_bytes=inputs.dtype_bytes,
    )
    budget = inputs.available_kv_memory_bytes
    if not inputs.allow_quantization:
        return AutoPolicyDecision("dense", None, dense, dense, 1.0, "quantization disabled")
    if budget is None:
        return AutoPolicyDecision("dense", None, dense, dense, 1.0, "no KV memory budget was provided")
    if dense <= int(0.90 * budget):
        return AutoPolicyDecision("dense", None, dense, dense, 1.0, "dense KV is within the memory budget")

    model_id = (inputs.model_id or "").lower()
    gqa_ratio = (
        inputs.num_q_heads / inputs.num_kv_heads
        if inputs.num_q_heads is not None and inputs.num_kv_heads
        else 1.0
    )
    key_sensitive = "qwen" in model_id or gqa_ratio >= 4.0
    presets: tuple[PresetName, ...] = (
        ("k16_v4", "k8_v4", "k4_v8", "k4_v4")
        if key_sensitive
        else ("k4_v8", "k8_v4", "k4_v4")
    )
    if inputs.prefer_low_latency and key_sensitive:
        presets = ("k16_v4", "k8_v4", "k4_v8", "k4_v4")

    last_policy: KVQuantPolicy | None = None
    last_bytes = dense
    for preset in presets:
        policy = qwen25_3b_policy(
            preset=preset,
            backend=inputs.backend,
            context_length=inputs.context_tokens,
            batch_size=inputs.batch_size,
            num_q_heads=inputs.num_q_heads or inputs.num_kv_heads,
            num_kv_heads=inputs.num_kv_heads,
            head_dim=inputs.head_dim,
            quality_priority="quality",
        )
        cache_bytes = estimate_policy_kv_bytes(
            policy,
            batch_size=inputs.batch_size,
            num_layers=inputs.num_layers,
            num_kv_heads=inputs.num_kv_heads,
            head_dim=inputs.head_dim,
            context_tokens=inputs.context_tokens,
            dtype_bytes=inputs.dtype_bytes,
        )
        last_policy = policy
        last_bytes = cache_bytes
        if cache_bytes <= budget:
            return AutoPolicyDecision(
                "torbuquant",
                policy,
                dense,
                cache_bytes,
                dense / max(1, cache_bytes),
                f"{preset} fits the KV memory budget",
            )

    assert last_policy is not None
    return AutoPolicyDecision(
        "torbuquant",
        last_policy,
        dense,
        last_bytes,
        dense / max(1, last_bytes),
        "no listed compressed preset fits the KV memory budget",
    )


def qwen25_3b_policy(
    *,
    preset: PresetName = "k16_v4",
    backend: BackendCapability = BackendCapability("diagnostic"),
    context_length: int = 4096,
    batch_size: int = 1,
    num_q_heads: int = 16,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    vram_budget_bytes: int | None = None,
    quality_priority: Literal["quality", "memory"] = "quality",
) -> KVQuantPolicy:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("Qwen-like GQA requires query heads divisible by KV heads")
    if head_dim != 128:
        raise ValueError(f"Qwen2.5-3B policy expects head_dim=128, got {head_dim}")

    mapping = {
        "k8_v4": ("K8", "V4"),
        "k16_v4": ("K16", "V4"),
        "k4_v8": ("K4", "V8"),
        "k4_v4": ("K4", "V4"),
        "k4_v3": ("K4", "V3"),
        "k4_v2": ("K4", "V2"),
        "v_compress": ("K16", "V4"),
        "boundary_v": ("K16", "V2"),
        "recent_window": ("K16", "V4"),
        "sparse_v": ("K16", "V4"),
    }
    k_format, v_format = mapping[preset]
    fallback_mode = None
    fallback_count = 0
    reason = "preset"

    if quality_priority == "quality" and k_format == "K4" and not backend.fused_decode:
        k_format = "K16"
        fallback_mode = "k16_key_fallback"
        fallback_count = 1
        reason = "K4 requires compressed score kernels before promotion"

    if v_format == "V3" and not backend.supports_v3:
        v_format = "V4"
        fallback_mode = "v4_value_fallback"
        fallback_count += 1
        reason = "V3 requires a backend format gate"

    if preset == "sparse_v" and not backend.supports_sparse_v:
        raise ValueError("sparse_v preset requires sparse V backend support")

    if backend.name == "hf" and k_format == "K4" and not backend.fused_decode:
        raise ValueError("HF K4 production mode requires compressed attention kernels")

    recent = 128 if context_length >= 256 else min(32, context_length)
    if preset == "recent_window":
        recent = max(recent, 256)

    boundary_tokens = 0
    if v_format in ("V2", "V3") or preset == "boundary_v":
        boundary_tokens = 4

    if vram_budget_bytes is not None and quality_priority == "memory" and preset in ("k16_v4", "v_compress"):
        k_format = "K8"
        reason = "memory budget selected K8 plus compressed V"

    return KVQuantPolicy(
        preset=preset,
        k_format=k_format,
        v_format=v_format,
        recent_window=recent,
        boundary_tokens=boundary_tokens,
        sparse_v=preset == "sparse_v",
        sparse_v_threshold=1e-6 if preset == "sparse_v" else None,
        allow_v3=backend.supports_v3,
        allow_k_low_bits=backend.fused_decode,
        fallback_mode=fallback_mode,
        fallback_count=fallback_count,
        reason=reason,
    )


def _k_row_bytes(format_name: str, *, head_dim: int, dtype_bytes: int) -> int:
    if format_name == "K16":
        return head_dim * dtype_bytes
    if format_name == "K8":
        return head_dim + _group_count(head_dim, 32) * 4
    if format_name == "K4":
        return ((head_dim * 4 + 7) // 8) + 2
    raise ValueError(f"unsupported key format for byte estimate: {format_name}")


def _v_row_bytes(format_name: str, *, head_dim: int, dtype_bytes: int) -> int:
    if format_name == "V16":
        return head_dim * dtype_bytes
    if not format_name.startswith("V"):
        raise ValueError(f"unsupported value format for byte estimate: {format_name}")
    bits = int(format_name[1:])
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"unsupported value bits for byte estimate: {bits}")
    return ((head_dim * bits + 7) // 8) + _group_count(head_dim, 32) * 4


def _group_count(dim: int, group_size: int) -> int:
    return (dim + group_size - 1) // group_size

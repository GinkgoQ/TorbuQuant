"""Validation run for memory, quality, speed, and context scaling.

The script uses real Qwen projection activations captured from local text. It
does not use synthetic random tensors for the reported quality or timing rows.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torbuquant.attention import dense_attention, dense_sdpa_attention, direct_qk_attention
from torbuquant.integration import KVQuantConfig, detect_backend
from torbuquant.integration.hf import capture_qwen_layer
from torbuquant.kv import BackendCapability, CompressedKVCache, qwen25_3b_policy
from torbuquant.quality import capacity_from_bytes, cuda_time_ms, distribution_kl, tensor_error
from torbuquant.triton import KernelCounters, decode_block, triton_available


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_local_long_text(max_chars: int = 120_000) -> str:
    candidates = [
        ROOT / "paper" / "main.tex",
        ROOT / "RESEARCH_NOTES.md",
        ROOT / "ARCHITECTURE.md",
        ROOT / "plan.md",
    ]
    parts: list[str] = []
    for path in candidates:
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n\n".join(parts)
    if not text.strip():
        raise RuntimeError("no local text sources found for real activation capture")
    while len(text) < max_chars:
        text = text + "\n\n" + text
    return text[:max_chars]


def build_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    preset: str,
    device: torch.device,
    dtype: torch.dtype,
) -> CompressedKVCache:
    backend = BackendCapability("vllm", fused_decode=True) if preset.startswith("k4") else BackendCapability("diagnostic")
    policy = replace(
        qwen25_3b_policy(
            preset=preset,
            backend=backend,
            context_length=int(keys.shape[0]),
            num_kv_heads=int(keys.shape[1]),
            head_dim=int(keys.shape[2]),
        ),
        recent_window=0,
    )
    cache = CompressedKVCache(
        head_dim=int(keys.shape[2]),
        num_kv_heads=int(keys.shape[1]),
        layer_idx=0,
        policy=policy,
        device=device,
        dtype=dtype,
        chunk_size=int(keys.shape[0]),
    )
    cache.prefill(keys, values)
    cache.flush_recent()
    return cache


def _run_preset(
    *,
    preset: str,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    dense_ref: Any,
    dense_timing_ms: dict[str, Any],
    repeats: int,
    warmup: int,
    memory_budget_bytes: int,
    device: torch.device,
) -> dict[str, Any]:
    cache = build_cache(keys, values, preset=preset, device=device, dtype=torch.float16)
    direct = direct_qk_attention(query.float(), cache, scale=1.0 / math.sqrt(int(keys.shape[-1])))
    output_error = tensor_error(direct.output, dense_ref.output)
    score_kl = distribution_kl(direct.scores, dense_ref.scores) if direct.scores is not None else None
    block = cache.blocks[0]
    counters = KernelCounters()
    kernel_result, kernel_timing = cuda_time_ms(
        lambda: decode_block(
            query,
            block,
            num_kv_heads=cache.num_kv_heads,
            scale=1.0 / math.sqrt(int(keys.shape[-1])),
            quantizer=cache._k4_quantizer,
            counters=counters,
        ),
        warmup=warmup,
        repeats=repeats,
    )
    kernel_error = tensor_error(kernel_result.output, direct.output)
    memory = cache.memory_report().as_dict()
    dense_per_token = memory["dense_kv_bytes"] / int(keys.shape[0])
    compressed_per_token = memory["compressed_total_bytes"] / int(keys.shape[0])
    dense_capacity = capacity_from_bytes(memory_budget_bytes=memory_budget_bytes, bytes_per_token=dense_per_token)
    compressed_capacity = capacity_from_bytes(memory_budget_bytes=memory_budget_bytes, bytes_per_token=compressed_per_token)
    return {
        "preset": preset,
        "kv_len": int(keys.shape[0]),
        "memory": memory,
        "bytes_per_token": {
            "dense": dense_per_token,
            "compressed": compressed_per_token,
        },
        "capacity_under_budget": {
            "budget_bytes": memory_budget_bytes,
            "dense_tokens": dense_capacity,
            "compressed_tokens": compressed_capacity,
            "gain": compressed_capacity / dense_capacity if dense_capacity else 0.0,
        },
        "quality": {
            "output_vs_dense": output_error.to_dict(),
            "attention_kl_vs_dense": score_kl,
            "direct_label": direct.report.label,
        },
        "speed": {
            "dense_sdpa_ms": dense_timing_ms,
            "compressed_decode_ms": kernel_timing.to_dict(),
            "speed_ratio_vs_dense_sdpa": (
                dense_timing_ms["median_ms"] / kernel_timing.median_ms
                if kernel_timing.median_ms > 0
                else None
            ),
            "kernel": kernel_result.report.to_dict(),
            "kernel_vs_direct": kernel_error.to_dict(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--contexts", default="128,512,1024")
    parser.add_argument("--presets", default="k16_v4,k8_v4,k4_v4")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--memory-budget-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="reports/validation_run.json")
    args = parser.parse_args()

    if not triton_available():
        raise RuntimeError("validation run requires Triton with CUDA")
    set_seed(args.seed)
    device = torch.device("cuda")
    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    presets = [x.strip() for x in args.presets.split(",") if x.strip()]
    if not contexts:
        raise ValueError("at least one context length is required")
    max_context = max(contexts)
    text = load_local_long_text()
    config = KVQuantConfig(
        model_id=args.model_id,
        backend="hf",
        mode="diagnostic",
        preset="k16_v4",
        context_length=max(args.max_length, max_context + 1),
        local_files_only=True,
        recent_window=0,
    )
    capture = capture_qwen_layer(
        config=config,
        texts=[text],
        device=device,
        max_length=max(args.max_length, max_context + 1),
    )
    mask = capture.attention_mask.bool()
    q = capture.q[mask].to(device=device, dtype=torch.float16)
    k = capture.k[mask].to(device=device, dtype=torch.float16)
    v = capture.v[mask].to(device=device, dtype=torch.float16)
    available = int(k.shape[0])
    contexts = [length for length in contexts if length + 1 <= available]
    if not contexts:
        raise RuntimeError(f"not enough captured tokens for requested contexts; captured {available}")

    rows: list[dict[str, Any]] = []
    memory_budget_bytes = int(args.memory_budget_gib * 1024**3)
    for length in contexts:
        keys = k[:length].contiguous()
        values = v[:length].contiguous()
        query = q[length : length + 1].contiguous()
        dense_ref = dense_attention(query.float(), keys.float(), values.float(), scale=1.0 / math.sqrt(int(keys.shape[-1])))
        _dense_out, dense_timing = cuda_time_ms(
            lambda: dense_sdpa_attention(query, keys, values, scale=1.0 / math.sqrt(int(keys.shape[-1]))),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        dense_timing_ms = dense_timing.to_dict()
        for preset in presets:
            rows.append(
                _run_preset(
                    preset=preset,
                    query=query,
                    keys=keys,
                    values=values,
                    dense_ref=dense_ref,
                    dense_timing_ms=dense_timing_ms,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    memory_budget_bytes=memory_budget_bytes,
                    device=device,
                )
            )

    quality_rows = [
        row for row in rows
        if row["quality"]["output_vs_dense"]["cosine"] > 0.995
        and row["memory"]["compression_ratio"] > 1.0
    ]
    speed_quality_rows = [
        row for row in quality_rows
        if (row["speed"]["speed_ratio_vs_dense_sdpa"] or 0.0) > 1.0
    ]
    max_memory_row = max(rows, key=lambda row: row["memory"]["compression_ratio"])
    max_quality_row = max(rows, key=lambda row: row["quality"]["output_vs_dense"]["cosine"])
    top_speed_quality_row = (
        max(speed_quality_rows, key=lambda row: row["speed"]["speed_ratio_vs_dense_sdpa"])
        if speed_quality_rows
        else None
    )
    gates = {
        "memory_reduction_all_rows": all(row["memory"]["compression_ratio"] > 1.0 for row in rows),
        "quality_output_cosine_all_rows": all(row["quality"]["output_vs_dense"]["cosine"] > 0.98 for row in rows),
        "has_quality_preserving_row": bool(quality_rows),
        "has_quality_preserving_speedup_row": bool(speed_quality_rows),
        "large_context_measured": max(row["kv_len"] for row in rows) >= min(max_context, 1024),
        "has_speedup_row": any(
            (row["speed"]["speed_ratio_vs_dense_sdpa"] or 0.0) > 1.0
            for row in rows
        ),
    }
    report = {
        "phase": "validation",
        "device": "cuda",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "model_id": args.model_id,
        "capture": capture.to_report(),
        "capabilities": {
            "hf": detect_backend("hf").to_dict(),
            "vllm": detect_backend("vllm").to_dict(),
        },
        "contexts_requested": [int(x) for x in args.contexts.split(",") if x.strip()],
        "contexts_measured": contexts,
        "presets": presets,
        "measurement": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "memory_budget_gib": args.memory_budget_gib,
            "baseline": "PyTorch SDPA dense KV for timing; dense_attention for score/weight reference",
        },
        "gates": gates,
        "selection": {
            "quality_rows": [
                {"preset": row["preset"], "kv_len": row["kv_len"]}
                for row in quality_rows
            ],
            "speed_quality_rows": [
                {"preset": row["preset"], "kv_len": row["kv_len"]}
                for row in speed_quality_rows
            ],
            "max_memory_row": {
                "preset": max_memory_row["preset"],
                "kv_len": max_memory_row["kv_len"],
                "compression_ratio": max_memory_row["memory"]["compression_ratio"],
                "output_cosine": max_memory_row["quality"]["output_vs_dense"]["cosine"],
            },
            "max_quality_row": {
                "preset": max_quality_row["preset"],
                "kv_len": max_quality_row["kv_len"],
                "compression_ratio": max_quality_row["memory"]["compression_ratio"],
                "output_cosine": max_quality_row["quality"]["output_vs_dense"]["cosine"],
            },
            "top_speed_quality_row": None
            if top_speed_quality_row is None
            else {
                "preset": top_speed_quality_row["preset"],
                "kv_len": top_speed_quality_row["kv_len"],
                "speed_ratio_vs_dense_sdpa": top_speed_quality_row["speed"]["speed_ratio_vs_dense_sdpa"],
                "compression_ratio": top_speed_quality_row["memory"]["compression_ratio"],
                "output_cosine": top_speed_quality_row["quality"]["output_vs_dense"]["cosine"],
            },
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

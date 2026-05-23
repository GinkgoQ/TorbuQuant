"""Phase 4 real-data checks for cache ownership, write path, and policy."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_activation_capture import collect_qwen_qkv, set_seed
from torbuquant.kv import BackendCapability, CompressedKVCache, qwen25_3b_policy


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.float() - b.float()).pow(2))).item())


def _reshape_kv(flat: torch.Tensor, *, num_kv_heads: int, max_tokens: int) -> torch.Tensor:
    tokens = min(max_tokens, int(flat.shape[0]) // num_kv_heads)
    usable = tokens * num_kv_heads
    return flat[:usable].reshape(tokens, num_kv_heads, flat.shape[-1]).contiguous()


def _production_dense_denied(cache: CompressedKVCache) -> bool:
    try:
        cache.production_handle().dense()
    except RuntimeError:
        return True
    return False


def _run_cache_case(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    preset: str,
    backend: BackendCapability,
    recent_window: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    policy = replace(
        qwen25_3b_policy(
            preset=preset,
            backend=backend,
            context_length=int(keys.shape[0]),
            num_kv_heads=int(keys.shape[1]),
            head_dim=int(keys.shape[2]),
        ),
        recent_window=recent_window,
    )
    cache = CompressedKVCache(
        head_dim=int(keys.shape[2]),
        num_kv_heads=int(keys.shape[1]),
        layer_idx=0,
        policy=policy,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        chunk_size=chunk_size,
    )
    split = max(1, int(keys.shape[0]) - 8)
    cache.prefill(keys[:split], values[:split])
    for idx in range(split, int(keys.shape[0])):
        cache.append_decode(keys[idx : idx + 1], values[idx : idx + 1])

    dense = cache.diagnostic_dense()
    memory = cache.memory_report().as_dict()
    report = cache.to_report()
    return {
        "preset": preset,
        "policy": policy.to_dict(),
        "seq_len": cache.seq_len,
        "compressed_tokens": cache.compressed_tokens,
        "recent_tokens": cache.recent_tokens,
        "blocks": [block["length"] for block in report["blocks"]],
        "production_dense_denied": _production_dense_denied(cache),
        "diagnostic_label": dense.label,
        "diagnostic_shape": list(dense.keys.shape),
        "key_rmse_vs_dense": _rmse(dense.keys.cpu(), keys[: dense.keys.shape[0]].cpu()),
        "value_rmse_vs_dense": _rmse(dense.values.cpu(), values[: dense.values.shape[0]].cpu()),
        "memory": memory,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-texts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--recent-window", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", default="reports/phase4_real_data_check.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _q, k_flat, v_flat, meta = collect_qwen_qkv(
        model_id=args.model,
        max_length=args.max_length,
        max_texts=args.max_texts,
        layer_idx=args.layer,
        device=device,
        local_files_only=not args.allow_download,
    )
    keys = _reshape_kv(k_flat, num_kv_heads=int(meta["num_kv_heads"]), max_tokens=args.max_tokens)
    values = _reshape_kv(v_flat, num_kv_heads=int(meta["num_kv_heads"]), max_tokens=args.max_tokens)

    cases = [
        _run_cache_case(
            keys,
            values,
            preset="k16_v4",
            backend=BackendCapability("diagnostic"),
            recent_window=args.recent_window,
            chunk_size=args.chunk_size,
            device=device,
        ),
        _run_cache_case(
            keys,
            values,
            preset="boundary_v",
            backend=BackendCapability("diagnostic"),
            recent_window=args.recent_window,
            chunk_size=args.chunk_size,
            device=device,
        ),
        _run_cache_case(
            keys,
            values,
            preset="k4_v4",
            backend=BackendCapability("vllm", fused_decode=True),
            recent_window=args.recent_window,
            chunk_size=args.chunk_size,
            device=device,
        ),
    ]
    gates = {
        "production_dense_denied": all(case["production_dense_denied"] for case in cases),
        "sequence_lengths_match": all(case["seq_len"] == int(keys.shape[0]) for case in cases),
        "recent_window_bounded": all(case["recent_tokens"] <= args.recent_window for case in cases),
        "has_compressed_blocks": all(case["compressed_tokens"] > 0 for case in cases),
        "boundary_case_stores_tokens": cases[1]["report"]["boundary"]["stored_tokens"] == 4,
        "k4_case_counts_metadata": cases[2]["memory"]["centroids_bytes"] > 0
        and cases[2]["memory"]["rotation_bytes"] > 0,
    }

    report = {
        "phase": 4,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "activation_meta": meta,
        "tokens_checked": int(keys.shape[0]),
        "num_kv_heads": int(keys.shape[1]),
        "head_dim": int(keys.shape[2]),
        "gates": gates,
        "cases": cases,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

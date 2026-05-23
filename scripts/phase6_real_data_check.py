"""Phase 6 real-data checks for Triton compressed decode kernels."""

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
from torbuquant.attention import direct_qk_attention
from torbuquant.kv import BackendCapability, CompressedKVCache, qwen25_3b_policy
from torbuquant.triton import KernelCounters, decode_block, triton_available


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.float() - b.float()).pow(2))).item())


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _reshape_heads(flat: torch.Tensor, *, heads: int, max_tokens: int) -> torch.Tensor:
    tokens = min(max_tokens, int(flat.shape[0]) // heads)
    usable = tokens * heads
    return flat[:usable].reshape(tokens, heads, flat.shape[-1]).contiguous()


def _build_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    preset: str,
    device: torch.device,
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
        dtype=torch.float32,
        chunk_size=int(keys.shape[0]),
    )
    cache.prefill(keys, values)
    return cache


def _case_report(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    preset: str,
    device: torch.device,
) -> dict[str, Any]:
    cache = _build_cache(keys, values, preset=preset, device=device)
    block = cache.blocks[0]
    counters = KernelCounters()
    result = decode_block(
        query,
        block,
        num_kv_heads=cache.num_kv_heads,
        scale=1.0 / (int(keys.shape[2]) ** 0.5),
        quantizer=cache._k4_quantizer,
        counters=counters,
    )
    reference = direct_qk_attention(query, cache, softmax_block_size=32)
    score_max_abs = None
    score_rmse = None
    if result.scores is not None:
        score_max_abs = _max_abs(result.scores, reference.scores)
        score_rmse = _rmse(result.scores, reference.scores)
    return {
        "preset": preset,
        "kernel_report": result.report.to_dict(),
        "cache_report": cache.to_report(),
        "memory": cache.memory_report().as_dict(),
        "score_max_abs": score_max_abs,
        "score_rmse": score_rmse,
        "output_max_abs": _max_abs(result.output, reference.output),
        "output_rmse": _rmse(result.output, reference.output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-texts", type=int, default=4)
    parser.add_argument("--kv-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", default="reports/phase6_real_data_check.json")
    args = parser.parse_args()

    if not triton_available():
        raise SystemExit("Phase 6 real-data check requires Triton with CUDA")

    set_seed(args.seed)
    device = torch.device("cuda")
    q_flat, k_flat, v_flat, meta = collect_qwen_qkv(
        model_id=args.model,
        max_length=args.max_length,
        max_texts=args.max_texts,
        layer_idx=args.layer,
        device=device,
        local_files_only=not args.allow_download,
    )
    keys = _reshape_heads(k_flat, heads=int(meta["num_kv_heads"]), max_tokens=args.kv_tokens).to(device=device)
    values = _reshape_heads(v_flat, heads=int(meta["num_kv_heads"]), max_tokens=args.kv_tokens).to(device=device)
    query = _reshape_heads(q_flat, heads=int(meta["num_q_heads"]), max_tokens=1).to(device=device)

    cases = [
        _case_report(query, keys, values, preset="k16_v4", device=device),
        _case_report(query, keys, values, preset="k8_v4", device=device),
        _case_report(query, keys, values, preset="k4_v4", device=device),
    ]
    gates = {
        "k16v4_output_matches_reference": cases[0]["output_max_abs"] < 2e-4,
        "k8v4_output_matches_reference": cases[1]["output_max_abs"] < 2e-4,
        "k4v4_output_matches_reference": cases[2]["output_max_abs"] < 2e-3,
        "k16v4_score_matches_reference": cases[0]["score_max_abs"] is not None
        and cases[0]["score_max_abs"] < 2e-4,
        "k8v4_score_matches_reference": cases[1]["score_max_abs"] is not None
        and cases[1]["score_max_abs"] < 2e-4,
        "k4v4_uses_fused_kernel": cases[2]["kernel_report"]["counters"]["fused_calls"] == 1,
    }

    report = {
        "phase": 6,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "activation_meta": meta,
        "kv_tokens": int(keys.shape[0]),
        "query_tokens": int(query.shape[0]),
        "num_q_heads": int(query.shape[1]),
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

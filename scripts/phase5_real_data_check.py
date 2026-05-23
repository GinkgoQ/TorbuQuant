"""Phase 5 real-data checks for reference attention paths."""

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
from torbuquant.attention import (
    dense_attention,
    dense_sdpa_attention,
    diagnostic_dequant_attention,
    direct_qk_attention,
    direct_qk_scores,
    online_softmax,
)
from torbuquant.kv import BackendCapability, CompressedKVCache, qwen25_3b_policy


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.float() - b.float()).pow(2))).item())


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _rmse_scores(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    finite = torch.isfinite(a_f) & torch.isfinite(b_f)
    mismatch = torch.isfinite(a_f) ^ torch.isfinite(b_f)
    if bool(mismatch.any().item()):
        return float("inf")
    if not bool(finite.any().item()):
        return 0.0
    return _rmse(a_f[finite], b_f[finite])


def _max_abs_scores(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    finite = torch.isfinite(a_f) & torch.isfinite(b_f)
    mismatch = torch.isfinite(a_f) ^ torch.isfinite(b_f)
    if bool(mismatch.any().item()):
        return float("inf")
    if not bool(finite.any().item()):
        return 0.0
    return _max_abs(a_f[finite], b_f[finite])


def _reshape_heads(flat: torch.Tensor, *, heads: int, max_tokens: int) -> torch.Tensor:
    tokens = min(max_tokens, int(flat.shape[0]) // heads)
    usable = tokens * heads
    return flat[:usable].reshape(tokens, heads, flat.shape[-1]).contiguous()


def _build_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    preset: str,
    recent_window: int,
    chunk_size: int,
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
    cache.prefill(keys, values)
    return cache


def _case_report(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    preset: str,
    recent_window: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    cache = _build_cache(
        keys,
        values,
        preset=preset,
        recent_window=recent_window,
        chunk_size=chunk_size,
        device=device,
    )
    q_start_pos = int(keys.shape[0] - query.shape[0])
    direct = direct_qk_attention(query, cache, causal=True, q_start_pos=q_start_pos, softmax_block_size=32)
    diagnostic = diagnostic_dequant_attention(query, cache, causal=True, q_start_pos=q_start_pos)
    scores, score_report = direct_qk_scores(query, cache, causal=True, q_start_pos=q_start_pos)
    dense_original = dense_attention(query, keys, values, causal=True, q_start_pos=q_start_pos)
    online = online_softmax(scores, block_size=32)
    torch_softmax = torch.softmax(scores.float(), dim=-1)
    return {
        "preset": preset,
        "seq_len": int(keys.shape[0]),
        "q_len": int(query.shape[0]),
        "direct_report": direct.report.to_dict(),
        "score_report": score_report.to_dict(),
        "cache_report": cache.to_report(),
        "memory": cache.memory_report().as_dict(),
        "direct_vs_diagnostic_score_rmse": _rmse_scores(direct.scores, diagnostic.scores),
        "direct_vs_diagnostic_score_max_abs": _max_abs_scores(direct.scores, diagnostic.scores),
        "direct_vs_diagnostic_output_rmse": _rmse(direct.output, diagnostic.output),
        "direct_vs_diagnostic_output_max_abs": _max_abs(direct.output, diagnostic.output),
        "direct_vs_original_dense_output_rmse": _rmse(direct.output, dense_original.output),
        "direct_vs_original_dense_output_max_abs": _max_abs(direct.output, dense_original.output),
        "online_softmax_max_abs": _max_abs(online, torch_softmax),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-texts", type=int, default=4)
    parser.add_argument("--kv-tokens", type=int, default=256)
    parser.add_argument("--query-tokens", type=int, default=2)
    parser.add_argument("--recent-window", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", default="reports/phase5_real_data_check.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    q_all = _reshape_heads(q_flat, heads=int(meta["num_q_heads"]), max_tokens=max(args.query_tokens, 1))
    query = q_all[-args.query_tokens :].to(device=device)

    sdpa = dense_sdpa_attention(query, keys, values, causal=False)
    dense_ref = dense_attention(query, keys, values, causal=False)
    cases = [
        _case_report(
            query,
            keys,
            values,
            preset="k16_v4",
            recent_window=args.recent_window,
            chunk_size=args.chunk_size,
            device=device,
        ),
        _case_report(
            query,
            keys,
            values,
            preset="k4_v4",
            recent_window=args.recent_window,
            chunk_size=args.chunk_size,
            device=device,
        ),
    ]
    gates = {
        "sdpa_matches_dense_reference": _max_abs(sdpa.output, dense_ref.output) < 1e-3,
        "online_softmax_matches_torch": all(case["online_softmax_max_abs"] < 1e-6 for case in cases),
        "direct_matches_diagnostic_scores": all(
            case["direct_vs_diagnostic_score_max_abs"] < 2e-3 for case in cases
        ),
        "direct_matches_diagnostic_outputs": all(
            case["direct_vs_diagnostic_output_max_abs"] < 2e-3 for case in cases
        ),
        "labels_present": all(
            case["direct_report"]["label"] == "direct_qk"
            and case["direct_report"]["value_label"] == "weighted_packed_v"
            for case in cases
        ),
    }

    report = {
        "phase": 5,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "activation_meta": meta,
        "kv_tokens": int(keys.shape[0]),
        "query_tokens": int(query.shape[0]),
        "num_q_heads": int(query.shape[1]),
        "num_kv_heads": int(keys.shape[1]),
        "head_dim": int(keys.shape[2]),
        "sdpa_vs_dense_output_max_abs": _max_abs(sdpa.output, dense_ref.output),
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

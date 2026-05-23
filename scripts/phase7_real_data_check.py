"""Phase 7 real-data integration check using Qwen activations."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torbuquant.attention.reference import direct_qk_attention
from torbuquant.integration import KVQuantConfig, detect_backend, policy_from_config
from torbuquant.integration.hf import HFDiagnosticCacheAdapter, build_layer_cache_from_capture, capture_qwen_layer
from torbuquant.integration.vllm import PagedCacheSpec, PagedCompressedKVCache
from torbuquant.triton import KernelCounters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wikitext_texts(max_texts: int) -> list[str]:
    ds = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="validation",
        download_mode="reuse_dataset_if_exists",
    )
    texts: list[str] = []
    for row in ds:
        text = row["text"].strip()
        if len(text) >= 80 and not text.startswith("="):
            texts.append(text)
        if len(texts) >= max_texts:
            break
    if not texts:
        texts = [Path("paper/main.tex").read_text(encoding="utf-8")[:4000]]
    return texts


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-texts", type=int, default=4)
    parser.add_argument("--kv-tokens", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--out", default="reports/phase7_real_data_check.json")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7 real-data check requires CUDA")
    set_seed(args.seed)
    device = torch.device("cuda")
    texts = load_wikitext_texts(args.max_texts)
    capture_config = KVQuantConfig(
        model_id=args.model_id,
        backend="hf",
        mode="diagnostic",
        preset="k16_v4",
        context_length=args.max_length,
        layer_idx=args.layer_idx,
        recent_window=0,
        local_files_only=True,
    )
    capture = capture_qwen_layer(
        config=capture_config,
        texts=texts,
        device=device,
        max_length=args.max_length,
    )
    mask = capture.attention_mask.bool()
    q = capture.q[mask].to(device=device, dtype=torch.float16)
    k = capture.k[mask].to(device=device, dtype=torch.float16)
    v = capture.v[mask].to(device=device, dtype=torch.float16)
    if k.shape[0] < args.kv_tokens + 1:
        raise RuntimeError(f"not enough captured tokens: got {k.shape[0]}, need {args.kv_tokens + 1}")
    k = k[: args.kv_tokens].contiguous()
    v = v[: args.kv_tokens].contiguous()
    query = q[args.kv_tokens : args.kv_tokens + 1].contiguous()

    hf_cache = build_layer_cache_from_capture(
        config=capture_config,
        keys=k,
        values=v,
        device=device,
    )
    adapter = HFDiagnosticCacheAdapter(config=capture_config, device=device)
    adapter_k, adapter_v = adapter.update(
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        layer_idx=args.layer_idx,
    )
    adapter_dense = adapter.layers[args.layer_idx].diagnostic_dense()
    adapter_k_error = _max_abs(adapter_k[0].transpose(0, 1), adapter_dense.keys)
    adapter_v_error = _max_abs(adapter_v[0].transpose(0, 1), adapter_dense.values)

    vllm_config = KVQuantConfig(
        model_id=args.model_id,
        backend="vllm",
        mode="production",
        preset="k16_v4",
        context_length=args.max_length,
        layer_idx=args.layer_idx,
        recent_window=0,
        local_files_only=True,
    )
    policy = policy_from_config(vllm_config)
    spec = PagedCacheSpec(
        block_size=16,
        num_kv_heads=int(capture.meta["num_kv_heads"]),
        head_dim=int(capture.meta["head_dim"]),
        k_format=policy.k_format,
        v_format=policy.v_format,
    )
    paged = PagedCompressedKVCache(config=vllm_config, spec=spec, device=device)
    slots = torch.arange(args.kv_tokens, device=device, dtype=torch.long)
    paged.write(k, v, slots)
    counters = KernelCounters()
    paged_result = paged.decode(
        query,
        slots,
        num_q_heads=int(capture.meta["num_q_heads"]),
        scale=1.0 / math.sqrt(int(capture.meta["head_dim"])),
        counters=counters,
    )
    reference = direct_qk_attention(
        query.float(),
        hf_cache,
        scale=1.0 / math.sqrt(int(capture.meta["head_dim"])),
    )
    output_error = _max_abs(paged_result.output, reference.output)
    gates = {
        "hf_adapter_returns_diagnostic_dense": adapter_k_error < 1e-6 and adapter_v_error < 1e-3,
        "vllm_paged_decode_matches_reference": output_error < 2e-4,
        "vllm_owns_paged_slots": paged.to_report()["cache"]["slots"] == args.kv_tokens,
    }
    report = {
        "phase": 7,
        "device": "cuda",
        "seed": args.seed,
        "capture": capture.to_report(),
        "capabilities": {
            "hf": detect_backend("hf").to_dict(),
            "vllm": detect_backend("vllm").to_dict(),
        },
        "kv_tokens": args.kv_tokens,
        "gates": gates,
        "hf_cache": hf_cache.to_report(),
        "hf_adapter": adapter.report(),
        "vllm_cache": paged.to_report(),
        "kernel": paged_result.report.to_dict(),
        "adapter_k_error": adapter_k_error,
        "adapter_v_error": adapter_v_error,
        "paged_output_max_abs": output_error,
        "reference_label": reference.report.label,
    }
    if not all(gates.values()):
        raise RuntimeError(json.dumps(report, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

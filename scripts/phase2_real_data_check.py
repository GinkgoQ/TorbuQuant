"""Phase 2 real-data checks for TorbuQuant core math.

This script uses real text and real Qwen/Qwen2.5-3B activations. It does not
claim LLM quality. It checks only Phase 2 math behavior:

- Lloyd-Max paper-level distortion reference values.
- MSE quantization on real key vectors.
- QR and RHT norm preservation on real key vectors.
- Direct compressed score behavior on real query/key vectors.
- QJL finite-seed behavior on real query/key vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torbuquant.core import (
    RotationMode,
    TorbuquantMSE,
    TorbuquantProd,
    build_rotation,
    compute_lloyd_max,
    rotate_backward,
    rotate_forward,
)
from torbuquant.core.types import MSEData


PAPER_MSE = {
    1: 0.36,
    2: 0.117,
    3: 0.03,
    4: 0.009,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_texts(max_texts: int) -> list[str]:
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
        local = Path("paper/main.tex").read_text()
        texts = [local[:4000]]
    return texts


def _collect_qwen_vectors(
    model_id: str,
    max_length: int,
    max_texts: int,
    layer_idx: int,
    device: torch.device,
    local_files_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    config = model.config
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size // config.num_attention_heads)

    layer = model.model.layers[layer_idx].self_attn
    captured: dict[str, torch.Tensor] = {}

    def save_q(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["q"] = output.detach()

    def save_k(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["k"] = output.detach()

    hooks = [
        layer.q_proj.register_forward_hook(save_q),
        layer.k_proj.register_forward_hook(save_k),
    ]

    texts = _load_texts(max_texts)
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        model(**batch, use_cache=False)

    for hook in hooks:
        hook.remove()

    q_raw = captured["q"].float().cpu()
    k_raw = captured["k"].float().cpu()
    q = q_raw.reshape(q_raw.shape[0], q_raw.shape[1], num_q_heads, head_dim)
    k = k_raw.reshape(k_raw.shape[0], k_raw.shape[1], num_kv_heads, head_dim)

    mask = batch["attention_mask"].bool().cpu()
    k_vectors = k[mask].reshape(-1, num_kv_heads, head_dim).reshape(-1, head_dim)
    q_vectors = q[mask].reshape(-1, num_q_heads, head_dim).reshape(-1, head_dim)

    pair_count = min(k_vectors.shape[0], q_vectors.shape[0])
    k_vectors = k_vectors[:pair_count].contiguous()
    q_vectors = q_vectors[:pair_count].contiguous()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta = {
        "model_id": model_id,
        "layer_idx": layer_idx,
        "texts": len(texts),
        "tokens": int(mask.sum().item()),
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "vectors": int(pair_count),
    }
    return q_vectors, k_vectors, meta


def _mse_real_data_check(
    vectors: torch.Tensor,
    bits_list: list[int],
    mode: RotationMode,
    device: torch.device,
    max_vectors: int,
) -> list[dict[str, Any]]:
    x = vectors[:max_vectors].to(device=device, dtype=torch.float32)
    dim = x.shape[-1]
    x_unit = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    rotation = build_rotation(dim, mode, device=device, dtype=torch.float32, seed=2026)

    rows: list[dict[str, Any]] = []
    previous = None
    for bits in bits_list:
        quantizer = TorbuquantMSE(dim, bits, rotation, device=device, use_exact=True)
        q = quantizer.quantize(x_unit)
        x_hat = quantizer.dequantize(q)
        mse_per_vector = ((x_hat - x_unit) ** 2).sum(dim=-1)
        measured = float(mse_per_vector.mean().item())
        paper_ref = PAPER_MSE.get(bits)
        upper_bound = math.sqrt(3.0) * math.pi / 2.0 * (4.0 ** (-bits))
        rows.append(
            {
                "mode": mode.name.lower(),
                "bits": bits,
                "measured_mse": measured,
                "paper_reference_mse": paper_ref,
                "paper_upper_bound": upper_bound,
                "under_upper_bound": bool(measured <= upper_bound),
                "decreases_from_previous": None if previous is None else bool(measured < previous),
            }
        )
        previous = measured
    return rows


def _rotation_real_data_check(vectors: torch.Tensor, device: torch.device, max_vectors: int) -> list[dict[str, Any]]:
    x = vectors[:max_vectors].to(device=device, dtype=torch.float32)
    dim = x.shape[-1]
    rows: list[dict[str, Any]] = []
    for mode in (RotationMode.QR, RotationMode.RHT):
        rotation = build_rotation(dim, mode, device=device, dtype=torch.float32, seed=2030)
        y = rotate_forward(x, rotation)
        x_back = rotate_backward(y, rotation)
        rows.append(
            {
                "mode": mode.name.lower(),
                "max_round_trip_error": float((x_back - x).abs().max().item()),
                "max_norm_delta": float((y.norm(dim=-1) - x.norm(dim=-1)).abs().max().item()),
                "metadata": rotation.to_metadata(),
            }
        )
    return rows


def _score_real_data_check(
    queries: torch.Tensor,
    keys: torch.Tensor,
    device: torch.device,
    max_keys: int,
    qjl_seeds: int,
) -> dict[str, Any]:
    dim = keys.shape[-1]
    k = keys[:max_keys].to(device=device, dtype=torch.float32).unsqueeze(0)
    query = queries[:1].to(device=device, dtype=torch.float32)
    rotation = build_rotation(dim, RotationMode.QR, device=device, dtype=torch.float32, seed=2040)

    quantizer = TorbuquantProd(
        dim,
        4,
        rotation,
        device=device,
        qjl_seed=2041,
        use_exact=True,
        qjl_for_attention=False,
    )
    q = quantizer.quantize(k)
    dense_scores = (query.unsqueeze(1) * k.squeeze(0)).sum(dim=-1)
    mse_scores = quantizer.attention_score(query, q)
    prod_scores = quantizer.attention_score(query, q, include_qjl=True)
    prod_recon = quantizer.dequantize(q)
    prod_recon_scores = (query.unsqueeze(1) * prod_recon.squeeze(0)).sum(dim=-1)

    mse_data = MSEData(q.mse_indices, q.norms, q.mse_bits, q.dim)
    mse_recon = quantizer.mse.dequantize(mse_data)
    mse_recon_scores = (query.unsqueeze(1) * mse_recon.squeeze(0)).sum(dim=-1)

    seed_errors = []
    for offset in range(qjl_seeds):
        seeded = TorbuquantProd(
            dim,
            4,
            rotation,
            device=device,
            qjl_seed=3000 + offset,
            use_exact=True,
            qjl_for_attention=True,
        )
        q_seed = seeded.quantize(k)
        seed_scores = seeded.attention_score(query, q_seed)
        seed_errors.append((seed_scores - dense_scores).detach().cpu())
    errors = torch.stack(seed_errors, dim=0)
    mean_error = errors.mean(dim=0)
    centered_variance = errors.var(dim=0, unbiased=False)
    query_norm2 = query.pow(2).sum(dim=-1, keepdim=True).detach().cpu()
    residual_norms = q.residual_norms.detach().cpu()
    variance_bound = (math.pi / (2.0 * dim)) * query_norm2 * residual_norms.pow(2)
    scaled = 1.0 / math.sqrt(dim)

    return {
        "keys": int(k.shape[1]),
        "bits": 4,
        "qjl_for_attention_default": bool(quantizer.qjl_for_attention),
        "mse_score_matches_mse_reconstruction_max_abs": float((mse_scores - mse_recon_scores).abs().max().item()),
        "prod_score_matches_prod_reconstruction_max_abs": float((prod_scores - prod_recon_scores).abs().max().item()),
        "dense_score_abs_mean": float(dense_scores.abs().mean().item()),
        "dense_score_std": float(dense_scores.std(unbiased=False).item()),
        "mse_score_rmse_vs_dense": float(torch.sqrt(torch.mean((mse_scores - dense_scores) ** 2)).item()),
        "prod_score_rmse_vs_dense_one_seed": float(torch.sqrt(torch.mean((prod_scores - dense_scores) ** 2)).item()),
        "attention_scaled_mse_rmse_vs_dense": float(torch.sqrt(torch.mean(((mse_scores - dense_scores) * scaled) ** 2)).item()),
        "attention_scaled_prod_rmse_vs_dense_one_seed": float(torch.sqrt(torch.mean(((prod_scores - dense_scores) * scaled) ** 2)).item()),
        "qjl_seed_count": int(qjl_seeds),
        "qjl_mean_signed_error": float(mean_error.mean().item()),
        "qjl_mean_abs_mean_error": float(mean_error.abs().mean().item()),
        "qjl_mean_abs_mean_error_over_dense_abs_mean": float(
            mean_error.abs().mean().item() / dense_scores.abs().mean().clamp(min=1e-12).item()
        ),
        "qjl_error_rmse_across_seeds": float(torch.sqrt(torch.mean(errors ** 2)).item()),
        "qjl_centered_variance_mean": float(centered_variance.mean().item()),
        "qjl_variance_bound_mean": float(variance_bound.mean().item()),
        "qjl_variance_under_bound_fraction": float((centered_variance <= variance_bound).float().mean().item()),
    }


def _codebook_reference_check(dim: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bits, expected in PAPER_MSE.items():
        cb = compute_lloyd_max(dim, bits, max_iter=300, tol=1e-13)
        total_mse = cb["mse_per_dim"] * dim
        rows.append(
            {
                "dim": dim,
                "bits": bits,
                "computed_total_mse": total_mse,
                "paper_reference_mse": expected,
                "absolute_delta": abs(total_mse - expected),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-texts", type=int, default=4)
    parser.add_argument("--max-vectors", type=int, default=2048)
    parser.add_argument("--score-keys", type=int, default=128)
    parser.add_argument("--qjl-seeds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", default="reports/phase2_real_data_check.json")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_vectors, k_vectors, meta = _collect_qwen_vectors(
        args.model,
        args.max_length,
        args.max_texts,
        args.layer,
        device,
        local_files_only=not args.allow_download,
    )

    head_dim = int(k_vectors.shape[-1])
    report = {
        "phase": 2,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "activation_meta": meta,
        "codebook_reference": _codebook_reference_check(head_dim),
        "rotation_real_data": _rotation_real_data_check(k_vectors, device, args.max_vectors),
        "mse_real_data_qr": _mse_real_data_check(
            k_vectors,
            [1, 2, 3, 4],
            RotationMode.QR,
            device,
            args.max_vectors,
        ),
        "mse_real_data_rht": _mse_real_data_check(
            k_vectors,
            [1, 2, 3, 4],
            RotationMode.RHT,
            device,
            args.max_vectors,
        ),
        "score_real_data": _score_real_data_check(
            q_vectors,
            k_vectors,
            device,
            args.score_keys,
            args.qjl_seeds,
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Phase 3 real-data checks for packing, layouts, and byte accounting."""

from __future__ import annotations

import argparse
import json
import platform
import sys
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
from torbuquant.core import RotationMode, TorbuquantMSE, build_rotation
from torbuquant.kv import (
    CacheGeometry,
    PackedKVLayout,
    dequantize_values,
    estimate_persistent_bytes,
    quantize_values,
    report_from_components,
    validate_v_format,
    value_data_nbytes,
    value_formula_nbytes,
)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _value_checks(values: torch.Tensor, *, group_size: int, max_vectors: int) -> list[dict[str, Any]]:
    x = values[:max_vectors].contiguous()
    rows: list[dict[str, Any]] = []
    for bits in (8, 4, 3, 2):
        allow_v3 = bits == 3
        q = quantize_values(x, bits=bits, group_size=group_size, allow_v3=allow_v3)
        out = dequantize_values(q)
        err = out.float() - x.float()
        actual = value_data_nbytes(q)
        formula = value_formula_nbytes(x.shape, bits=bits, group_size=group_size)
        rows.append(
            {
                "format": f"V{bits}",
                "vectors": int(x.shape[0]),
                "dim": int(x.shape[-1]),
                "group_size": q.group_size,
                "padded_dim": int(formula["padded_dim"]),
                "groups": int(formula["groups"]),
                "compressed_v_bytes_actual": actual["compressed_v_bytes"],
                "compressed_v_bytes_formula": formula["compressed_v_bytes"],
                "scales_bytes_actual": actual["scales_bytes"],
                "scales_bytes_formula": formula["scales_bytes"],
                "zeros_bytes_actual": actual["zeros_bytes"],
                "zeros_bytes_formula": formula["zeros_bytes"],
                "rmse": float(torch.sqrt(torch.mean(err.pow(2))).item()),
                "max_abs_error": float(err.abs().max().item()),
                "actual_matches_formula": bool(
                    actual["compressed_v_bytes"] == formula["compressed_v_bytes"]
                    and actual["scales_bytes"] == formula["scales_bytes"]
                    and actual["zeros_bytes"] == formula["zeros_bytes"]
                ),
            }
        )
    return rows


def _k4_check(keys: torch.Tensor, *, max_vectors: int, device: torch.device) -> dict[str, Any]:
    x = keys[:max_vectors].to(device=device, dtype=torch.float32).contiguous()
    dim = x.shape[-1]
    rotation = build_rotation(dim, RotationMode.RHT, device=device, dtype=torch.float32, seed=3100)
    quantizer = TorbuquantMSE(dim, 4, rotation, device=device, use_exact=True, norm_correction=True)
    q = quantizer.quantize(x)
    out = quantizer.dequantize(q)
    err = out.cpu().float() - keys[:max_vectors].float()
    rotation_tensors = {
        "signs1": rotation.signs1,
        "signs2": rotation.signs2,
    }
    return {
        "format": "K4",
        "vectors": int(x.shape[0]),
        "dim": dim,
        "compressed_k_bytes": _tensor_bytes(q.indices),
        "norms_bytes": _tensor_bytes(q.norms),
        "centroids_bytes": _tensor_bytes(quantizer.centroids) + _tensor_bytes(quantizer.decision_bnd),
        "rotation_bytes": sum(_tensor_bytes(t) for t in rotation_tensors.values() if t is not None),
        "rmse": float(torch.sqrt(torch.mean(err.pow(2))).item()),
        "max_abs_error": float(err.abs().max().item()),
    }


def _memory_report_check(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    max_vectors: int,
    group_size: int,
    device: torch.device,
) -> dict[str, Any]:
    k = keys[:max_vectors].to(device=device, dtype=torch.float32).contiguous()
    v = values[:max_vectors].to(device=device, dtype=torch.float32).contiguous()
    dim = int(k.shape[-1])
    rotation = build_rotation(dim, RotationMode.RHT, device=device, dtype=torch.float32, seed=3200)
    k_quantizer = TorbuquantMSE(dim, 4, rotation, device=device, use_exact=True)
    kq = k_quantizer.quantize(k)
    vq = quantize_values(v, bits=4, group_size=group_size)
    layout = PackedKVLayout("K4", "V4", head_dim=dim, value_group_size=group_size)
    geometry = CacheGeometry(batch=1, num_kv_heads=1, tokens=max_vectors, head_dim=dim)
    estimate = estimate_persistent_bytes(geometry, layout)
    metadata = json.dumps(layout.to_dict(), sort_keys=True).encode("utf-8")
    rotation_tensors = {
        "signs1": rotation.signs1,
        "signs2": rotation.signs2,
    }
    report = report_from_components(
        dense_kv_bytes=geometry.dense_kv_bytes,
        compressed_k=kq.indices,
        compressed_v=vq.data,
        scales=vq.scales,
        zeros=vq.zeros,
        norms=kq.norms,
        centroids={
            "centroids": k_quantizer.centroids,
            "decision_bnd": k_quantizer.decision_bnd,
        },
        rotation=rotation_tensors,
        metadata_bytes=len(metadata),
        formula_total_bytes=sum(
            estimate[field]
            for field in [
                "compressed_k_bytes",
                "compressed_v_bytes",
                "norms_bytes",
                "scales_bytes",
                "zeros_bytes",
            ]
        ),
    )
    data = report.as_dict()
    data["estimate"] = estimate
    data["layout"] = layout.to_dict()
    data["actual_matches_estimate_payloads"] = bool(
        data["compressed_k_bytes"] == estimate["compressed_k_bytes"]
        and data["compressed_v_bytes"] == estimate["compressed_v_bytes"]
        and data["scales_bytes"] == estimate["scales_bytes"]
        and data["zeros_bytes"] == estimate["zeros_bytes"]
        and data["norms_bytes"] == estimate["norms_bytes"]
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-texts", type=int, default=4)
    parser.add_argument("--max-vectors", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", default="reports/phase3_real_data_check.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _q, k, v, meta = collect_qwen_qkv(
        model_id=args.model,
        max_length=args.max_length,
        max_texts=args.max_texts,
        layer_idx=args.layer,
        device=device,
        local_files_only=not args.allow_download,
    )
    max_vectors = min(args.max_vectors, k.shape[0], v.shape[0])

    v3_gate_ok = False
    try:
        validate_v_format("V3")
    except ValueError:
        v3_gate_ok = True

    report = {
        "phase": 3,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "activation_meta": meta,
        "v3_gate_blocks_default": v3_gate_ok,
        "value_checks": _value_checks(v, group_size=args.group_size, max_vectors=max_vectors),
        "k4_check": _k4_check(k, max_vectors=max_vectors, device=device),
        "memory_report_k4v4": _memory_report_check(
            k,
            v,
            max_vectors=max_vectors,
            group_size=args.group_size,
            device=device,
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

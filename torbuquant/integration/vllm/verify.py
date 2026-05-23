"""Model-level verification helpers for TurboQuant cache modes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from torbuquant.integration.hf.dynamic_cache import CompressedDynamicCache
from torbuquant.integration.vllm.metadata import load_metadata
from torbuquant.integration.vllm.registry import is_recipe_dtype


MODEL_FAMILIES = {
    "qwen2": "Qwen",
    "llama": "Llama",
    "mistral": "Mistral",
    "gemma": "Gemma",
    "gemma2": "Gemma",
    "gemma3": "Gemma",
    "gemma4": "Gemma",
    "phi3": "Phi",
    "phi": "Phi",
    "molmo2": "Molmo",
}


@dataclass(frozen=True)
class ModelCacheShape:
    model_type: str
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    num_layers: int
    num_shared_kv_layers: int
    layer_types: tuple[str, ...] | None

    @property
    def unique_cache_layers(self) -> int:
        return int(self.num_layers - self.num_shared_kv_layers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def detect_model_cache_shape(config: Any) -> ModelCacheShape:
    text_config = detect_text_config(config)
    model_type = str(getattr(config, "model_type", getattr(text_config, "model_type", "unknown")))
    hidden_size = getattr(text_config, "hidden_size", None)
    num_q_heads = int(getattr(text_config, "num_attention_heads", 0))
    raw_head_dim = getattr(text_config, "head_dim", None)
    if raw_head_dim is None:
        if hidden_size is None or num_q_heads <= 0:
            raise ValueError("cannot derive head_dim from config")
        head_dim = int(hidden_size) // num_q_heads
    else:
        head_dim = int(raw_head_dim)
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    num_kv_heads = int(getattr(text_config, "num_key_value_heads", num_q_heads))
    num_layers = int(getattr(text_config, "num_hidden_layers"))
    shared = int(getattr(text_config, "num_kv_shared_layers", 0) or 0)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        layer_types = tuple(str(item) for item in layer_types)
    return ModelCacheShape(
        model_type=model_type,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        num_shared_kv_layers=shared,
        layer_types=layer_types,
    )


def version_report() -> dict[str, str | None]:
    report: dict[str, str | None] = {}
    for name in ("torch", "transformers", "vllm", "triton"):
        try:
            report[name] = version(name)
        except PackageNotFoundError:
            report[name] = None
    return report


def parse_verify_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify TurboQuant cache behavior")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 5))
    parser.add_argument("--k-bits", type=int, choices=(2, 3, 4, 5))
    parser.add_argument("--v-bits", type=int, choices=(2, 3, 4, 5))
    parser.add_argument("--kv-cache-dtype", default="tq4")
    parser.add_argument("--metadata-path")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--token")
    args = parser.parse_args(argv)
    if args.bits is not None and (args.k_bits is not None or args.v_bits is not None):
        parser.error("--bits cannot be mixed with --k-bits or --v-bits")
    if (args.k_bits is None) != (args.v_bits is None):
        parser.error("--k-bits and --v-bits must appear together")
    if args.bits is None and args.k_bits is None:
        args.bits = 4
    if is_recipe_dtype(args.kv_cache_dtype) and args.metadata_path is None:
        parser.error("recipe cache dtype requires --metadata-path")
    return args


def verify_metadata_only(*, model: str, kv_cache_dtype: str, metadata_path: str, head_dim: int) -> dict[str, Any]:
    metadata = load_metadata(metadata_path)
    status = "PASS" if metadata.recipe == kv_cache_dtype and metadata.head_dim == head_dim else "FAIL"
    return {
        "model": model,
        "kv_cache_dtype": kv_cache_dtype,
        "status": status,
        "metadata": {
            "recipe": metadata.recipe,
            "head_dim": metadata.head_dim,
            "layers": len(metadata.layers),
        },
    }


def run_cache_cosine_verification(
    *,
    model_id: str,
    bits: int,
    threshold: float,
    k_bits: int | None = None,
    v_bits: int | None = None,
    trust_remote_code: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        token=token,
    )
    shape = detect_model_cache_shape(model.config)
    if shape.unique_cache_layers <= 0:
        raise ValueError("model has no unique cache layers")
    device = next(model.parameters()).device
    text_config = detect_text_config(config)
    ref_cache = DynamicCache(config=text_config)
    tq_cache = DynamicCache(config=text_config)
    seq_len = 128

    generator = torch.Generator(device=device).manual_seed(42)
    tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _layer_idx in range(shape.unique_cache_layers):
        key = torch.randn(
            (1, shape.num_kv_heads, seq_len, shape.head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        value = torch.randn(
            (1, shape.num_kv_heads, seq_len, shape.head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        tensors.append((key, value))

    for layer_idx, (key, value) in enumerate(tensors):
        ref_cache.update(key, value, layer_idx)

    wrapper = CompressedDynamicCache(
        tq_cache,
        head_dim=shape.head_dim,
        bits=bits,
        k_bits=k_bits,
        v_bits=v_bits,
        device=device,
        model_config=text_config,
    )
    for layer_idx, (key, value) in enumerate(tensors):
        tq_cache.update(key, value, layer_idx)

    cosines: list[float] = []
    for layer_idx in range(shape.unique_cache_layers):
        ref_layer = ref_cache.layers[layer_idx]
        tq_layer = tq_cache.layers[layer_idx]
        ref_key = getattr(ref_layer, "keys", None)
        ref_value = getattr(ref_layer, "values", None)
        tq_key = getattr(tq_layer, "keys", None)
        tq_value = getattr(tq_layer, "values", None)
        if ref_key is None or ref_value is None or tq_key is None or tq_value is None:
            raise RuntimeError(f"cache tensors missing for layer {layer_idx}")
        key_cos = torch.nn.functional.cosine_similarity(ref_key.flatten().float(), tq_key.flatten().float(), dim=0).item()
        value_cos = torch.nn.functional.cosine_similarity(ref_value.flatten().float(), tq_value.flatten().float(), dim=0).item()
        cosines.append(min(key_cos, value_cos))
    wrapper.restore()
    min_cosine = min(cosines)
    return {
        "model": model_id,
        "family": MODEL_FAMILIES.get(shape.model_type, "unknown"),
        "bits": bits,
        "k_bits": k_bits if k_bits is not None else bits,
        "v_bits": v_bits if v_bits is not None else bits,
        "status": "PASS" if min_cosine >= threshold else "FAIL",
        "threshold": threshold,
        "min_cosine": min_cosine,
        "per_layer_cosine": cosines,
        "shape": shape.to_dict(),
        "versions": version_report(),
        "compression": wrapper.compression_stats(),
    }


def format_verify_report(result: dict[str, Any]) -> str:
    lines = [f"Model: {result['model']}", f"Status: {result['status']}"]
    if "family" in result:
        lines.append(f"Family: {result['family']}")
    if "min_cosine" in result:
        lines.append(f"Min cosine: {result['min_cosine']:.6f}")
        lines.append(f"Threshold: {result['threshold']}")
    if "compression" in result:
        compression = result["compression"]
        lines.append(f"Compressed bytes: {compression.get('compressed_bytes', 0)}")
        lines.append(f"Baseline bytes: {compression.get('baseline_bytes', 0)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_verify_args(argv)
    if is_recipe_dtype(args.kv_cache_dtype):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, token=args.token)
        shape = detect_model_cache_shape(config)
        result = verify_metadata_only(
            model=args.model,
            kv_cache_dtype=args.kv_cache_dtype,
            metadata_path=args.metadata_path,
            head_dim=shape.head_dim,
        )
    else:
        result = run_cache_cosine_verification(
            model_id=args.model,
            bits=args.bits,
            threshold=args.threshold,
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            trust_remote_code=args.trust_remote_code,
            token=args.token,
        )
    if args.json_output:
        print(json.dumps(result, indent=2), file=sys.stdout)
        print(format_verify_report(result), file=sys.stderr)
    else:
        print(format_verify_report(result), file=sys.stdout)
    raise SystemExit(0 if result["status"] == "PASS" else 1)

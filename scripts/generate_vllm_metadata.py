#!/usr/bin/env python3
"""Generate TurboQuant metadata from real activation scores."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import torch

from torbuquant.integration.vllm.calibration import (
    ActivationAccumulator,
    build_calibrated_metadata,
    derive_model_shape,
    discover_projection_modules,
    load_prompts,
    prompts_sha256,
    resolve_layer_indices,
    validate_calibration_model_choice,
)
from torbuquant.integration.vllm.metadata import CalibrationMetadata, save_metadata


def _resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unknown dtype: {name}") from exc


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_model_and_tokenizer(model_name: str, *, dtype: str, trust_remote_code: bool):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_errors: list[str] = []
    for loader in (AutoModelForCausalLM, AutoModel):
        try:
            model = loader.from_pretrained(
                model_name,
                torch_dtype=_resolve_dtype(dtype),
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            return model, tokenizer
        except Exception as exc:  # pragma: no cover - depends on local HF stack
            model_errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError("could not load calibration model: " + " | ".join(model_errors))


def _ensure_padding_token(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token


@torch.inference_mode()
def collect_scores(
    *,
    model_name: str,
    prompts: list[str],
    layer_indices: list[int],
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    max_seq_len: int,
    dtype: str,
    device_name: str,
    trust_remote_code: bool,
) -> tuple[dict[tuple[int, str], torch.Tensor], int]:
    device = _resolve_device(device_name)
    model, tokenizer = _load_model_and_tokenizer(
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    _ensure_padding_token(tokenizer)

    modules = discover_projection_modules(model, layer_indices)
    accumulator = ActivationAccumulator(num_kv_heads=num_kv_heads, head_dim=head_dim)
    handles = [
        module.register_forward_hook(accumulator.hook(layer_idx, tensor_name))
        for (layer_idx, tensor_name), module in sorted(modules.items())
    ]
    observed_tokens = 0
    with ExitStack() as stack:
        for handle in handles:
            stack.callback(handle.remove)
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            attention_mask = encoded["attention_mask"].to(device)
            observed_tokens += int(attention_mask.sum().item())
            accumulator.set_attention_mask(attention_mask)
            model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=attention_mask,
                use_cache=False,
            )
            accumulator.clear_attention_mask()
    if observed_tokens <= 0:
        raise ValueError("calibration observed zero tokens")
    return accumulator.channel_scores, observed_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TurboQuant vLLM metadata")
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-model")
    parser.add_argument("--kv-cache-dtype", choices=("turboquant25", "turboquant35"), required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--layer-pattern", default="model.layers.{i}.self_attn.attn")
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from transformers import AutoConfig

    prompts = load_prompts(args.prompts_file, limit=args.max_prompts)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    shape = derive_model_shape(config)
    calibration_model = args.calibration_model or args.model
    validate_calibration_model_choice(
        target_model=args.model,
        calibration_model=calibration_model,
        quantization_config=shape.quantization_config,
    )
    layer_indices = resolve_layer_indices(shape.num_hidden_layers, shape.layer_types)
    scores, observed_tokens = collect_scores(
        model_name=calibration_model,
        prompts=prompts,
        layer_indices=layer_indices,
        num_kv_heads=shape.num_kv_heads,
        head_dim=shape.head_dim,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        device_name=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    metadata = build_calibrated_metadata(
        recipe=args.kv_cache_dtype,
        head_dim=shape.head_dim,
        model_name=args.model,
        num_hidden_layers=shape.num_hidden_layers,
        layer_types=shape.layer_types,
        layer_pattern=args.layer_pattern,
        num_kv_heads=shape.num_kv_heads,
        calibration_scores=scores,
        calibration=CalibrationMetadata(
            method="activation_energy_v1",
            objective="sum_squared_activation",
            num_prompts=len(prompts),
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            num_observed_tokens=observed_tokens,
            dtype=args.dtype,
            device=str(_resolve_device(args.device)),
            prompts_sha256=prompts_sha256(prompts),
        ),
    )
    save_metadata(metadata, Path(args.output))


if __name__ == "__main__":
    main()

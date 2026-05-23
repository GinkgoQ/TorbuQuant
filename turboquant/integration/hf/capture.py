"""HuggingFace Qwen capture helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from torbuquant.integration.common import KVQuantConfig, OptionalDependencyError


@dataclass(frozen=True)
class QwenCapture:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    logits: torch.Tensor
    attention_mask: torch.Tensor
    meta: dict[str, Any]

    def to_report(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("q")
        data.pop("k")
        data.pop("v")
        data.pop("logits")
        data.pop("attention_mask")
        data["shapes"] = {
            "q": tuple(self.q.shape),
            "k": tuple(self.k.shape),
            "v": tuple(self.v.shape),
            "logits": tuple(self.logits.shape),
            "attention_mask": tuple(self.attention_mask.shape),
        }
        return data


def _load_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise OptionalDependencyError("transformers is required for HF integration") from exc
    return AutoModelForCausalLM, AutoTokenizer


def capture_qwen_layer(
    *,
    config: KVQuantConfig,
    texts: list[str],
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    max_length: int | None = None,
) -> QwenCapture:
    if not texts:
        raise ValueError("texts must contain at least one prompt")
    AutoModelForCausalLM, AutoTokenizer = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        local_files_only=config.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        local_files_only=config.local_files_only,
    ).to(device)
    model.eval()

    model_config = model.config
    num_q_heads = int(model_config.num_attention_heads)
    num_kv_heads = int(model_config.num_key_value_heads)
    head_dim = int(model_config.hidden_size // model_config.num_attention_heads)
    layer = model.model.layers[config.layer_idx].self_attn
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def save(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured[name] = output.detach()

        return save

    hooks = [
        layer.q_proj.register_forward_hook(make_hook("q")),
        layer.k_proj.register_forward_hook(make_hook("k")),
        layer.v_proj.register_forward_hook(make_hook("v")),
    ]
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        out = model(**batch, use_cache=False)
    for hook in hooks:
        hook.remove()

    q_raw = captured["q"].float()
    k_raw = captured["k"].float()
    v_raw = captured["v"].float()
    q = q_raw.reshape(q_raw.shape[0], q_raw.shape[1], num_q_heads, head_dim).contiguous()
    k = k_raw.reshape(k_raw.shape[0], k_raw.shape[1], num_kv_heads, head_dim).contiguous()
    v = v_raw.reshape(v_raw.shape[0], v_raw.shape[1], num_kv_heads, head_dim).contiguous()
    meta = {
        "model_id": config.model_id,
        "layer_idx": config.layer_idx,
        "texts": len(texts),
        "tokens": int(batch["attention_mask"].sum().item()),
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "device": str(device),
    }
    return QwenCapture(
        q=q.cpu(),
        k=k.cpu(),
        v=v.cpu(),
        logits=out.logits.detach().float().cpu(),
        attention_mask=batch["attention_mask"].detach().cpu(),
        meta=meta,
    )


def capture_generated_tokens(
    *,
    config: KVQuantConfig,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    AutoModelForCausalLM, AutoTokenizer = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, local_files_only=config.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        local_files_only=config.local_files_only,
    ).to(device)
    model.eval()
    batch = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = generated[0, batch["input_ids"].shape[1] :].detach().cpu().tolist()
    return {
        "model_id": config.model_id,
        "prompt_tokens": int(batch["input_ids"].shape[1]),
        "new_tokens": new_tokens,
        "text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }


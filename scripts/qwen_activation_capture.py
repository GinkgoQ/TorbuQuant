"""Capture Qwen Q/K/V projection activations from real text."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        texts = [Path("paper/main.tex").read_text()[:4000]]
    return texts


def collect_qwen_qkv(
    *,
    model_id: str,
    max_length: int,
    max_texts: int,
    layer_idx: int,
    device: torch.device,
    local_files_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
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

    def make_hook(name: str):
        def save(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured[name] = output.detach()

        return save

    hooks = [
        layer.q_proj.register_forward_hook(make_hook("q")),
        layer.k_proj.register_forward_hook(make_hook("k")),
        layer.v_proj.register_forward_hook(make_hook("v")),
    ]

    texts = load_wikitext_texts(max_texts)
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

    mask = batch["attention_mask"].bool().cpu()
    q_raw = captured["q"].float().cpu()
    k_raw = captured["k"].float().cpu()
    v_raw = captured["v"].float().cpu()

    q = q_raw.reshape(q_raw.shape[0], q_raw.shape[1], num_q_heads, head_dim)
    k = k_raw.reshape(k_raw.shape[0], k_raw.shape[1], num_kv_heads, head_dim)
    v = v_raw.reshape(v_raw.shape[0], v_raw.shape[1], num_kv_heads, head_dim)

    q_vectors = q[mask].reshape(-1, num_q_heads, head_dim).reshape(-1, head_dim)
    k_vectors = k[mask].reshape(-1, num_kv_heads, head_dim).reshape(-1, head_dim)
    v_vectors = v[mask].reshape(-1, num_kv_heads, head_dim).reshape(-1, head_dim)

    pair_count = min(q_vectors.shape[0], k_vectors.shape[0], v_vectors.shape[0])
    q_vectors = q_vectors[:pair_count].contiguous()
    k_vectors = k_vectors[:pair_count].contiguous()
    v_vectors = v_vectors[:pair_count].contiguous()

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
    return q_vectors, k_vectors, v_vectors, meta

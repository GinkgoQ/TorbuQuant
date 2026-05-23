# Quickstart

## 1. Run The Test Suite

```bash
python -m pytest tests -q
```

The exact number of tests can change as the repository evolves. Treat the
command output as the source of truth for your checkout.

## 2. Quantize Vectors With MSE

```python
import torch

from torbuquant.core import RotationMode, build_rotation, TorbuquantMSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim = 128
rotation = build_rotation(dim, RotationMode.RHT, device=device, seed=42)
quantizer = TorbuquantMSE(dim, bits=4, rotation=rotation, device=device)

x = torch.randn(32, dim, device=device)
payload = quantizer.quantize(x)
restored = quantizer.dequantize(payload)
```

`payload.indices` is packed `uint8`; `payload.norms` stores vector norms.

Check reconstruction error:

```python
err = (x.float() - restored.float()).pow(2).mean().sqrt()
print(float(err))
```

This is a tensor reconstruction check, not a model-quality check.

## 3. Build A KV Cache Policy

```python
from torbuquant.integration import KVQuantConfig, policy_from_config

config = KVQuantConfig(
    preset="k8_v4",
    backend="hf",
    mode="diagnostic",
    context_length=4096,
    num_q_heads=16,
    num_kv_heads=2,
    head_dim=128,
)

policy = policy_from_config(config)
print(policy.k_format, policy.v_format)
```

The policy object is the bridge between user configuration and cache format
selection. It is used by cache owners and integration adapters.

## 4. Use The HuggingFace Diagnostic Cache Wrapper

```python
from transformers import DynamicCache
from torbuquant.integration.hf import CompressedDynamicCache

cache = DynamicCache()
wrapper = CompressedDynamicCache(cache, head_dim=128, bits=4)

# Pass `cache` to model code. The wrapper patches cache.update().
# This path reconstructs dense tensors for HuggingFace attention.
stats = wrapper.compression_stats()
wrapper.restore()
```

This route is diagnostic because HuggingFace attention receives dense tensors.

## 5. Create vLLM Metadata

```bash
python scripts/generate_vllm_metadata.py \
  --model Qwen/Qwen2.5-3B \
  --kv-cache-dtype turboquant35 \
  --prompts-file prompts.txt \
  --output turboquant_kv.json \
  --dtype bfloat16 \
  --device cuda
```

The script loads prompts, registers projection hooks, accumulates activation
energy for K/V projections, and writes metadata JSON.

Inspect metadata before using recipe mode:

```python
from torbuquant.integration.vllm import load_metadata

metadata = load_metadata("turboquant_kv.json")
print(metadata.recipe, metadata.head_dim, len(metadata.layers))
```

## 6. Run Qwen A/B Evaluation

```bash
python scripts/qwen_ab_eval.py \
  --model Qwen/Qwen2.5-3B \
  --output-root reports/qwen_ab
```

This script keeps baseline and TurboQuant artifacts in separate directories and
records raw outputs and resource metrics.

Read the output report before interpreting a run. If TurboQuant is slower, the
report should say that plainly; memory reduction alone is not a speed result.

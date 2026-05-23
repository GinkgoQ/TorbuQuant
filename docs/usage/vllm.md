# vLLM Usage

The vLLM package area currently provides:

- cache dtype recognition,
- recipe layout helpers,
- metadata JSON loading and validation,
- page geometry and packed page storage,
- runtime path selection,
- verification helpers,
- PyTorch TQ row packing and paged decode reference.

It does not yet register a live vLLM attention backend in this repository.

## Implementation Map

| Capability | Module | Status |
| --- | --- | --- |
| Cache dtype parser | `registry.py`, `recipe.py` | Implemented. |
| Recipe layout | `recipe.py` | Implemented. |
| Recipe vector packing | `vector.py` | Implemented and tested. |
| Metadata JSON | `metadata.py` | Implemented and tested. |
| Calibration helper | `calibration.py`, `scripts/generate_vllm_metadata.py` | Implemented. |
| Page geometry | `page_cache.py` | Implemented and tested. |
| TQ row writer | `triton/tq4_update.py` | PyTorch contract path implemented. |
| TQ paged decode | `triton/tq4_decode.py` | PyTorch contract path implemented. |
| Live vLLM backend | not wired | Open work. |
| Recipe Triton update/decode | not present | Open work. |

## Cache Dtypes

Recognized TQ4 names:

```text
tq4
tq4_kv
k4v4
```

Recipe names:

```text
turboquant25
turboquant35
```

Recipe modes require metadata.

## TQ Byte Layout

For one component, one token, and one KV head:

```text
index_bytes = ceil(head_dim * bits / 8)
row_bytes = index_bytes + 4
```

The last four bytes are fp32 norm bytes. For head dimension `128`:

| Bits | Index bytes | Row bytes |
| ---: | ---: | ---: |
| 2 | 32 | 36 |
| 3 | 48 | 52 |
| 4 | 64 | 68 |
| 5 | 80 | 84 |

K and V are stored independently. A K4/V2 token-head pair uses:

```text
68 + 36 = 104 bytes
```

before page padding.

## Metadata Loading

```python
from turboquant.integration.vllm import load_metadata

metadata = load_metadata("turboquant_kv.json")
layer = metadata.get_layer("model.layers.0.self_attn")
```

Metadata lookup accepts a small set of layer-name aliases, such as removing a
trailing `.attn` or a leading `language_model.` prefix. It does not infer
missing layers.

## Metadata Validation Rules

The metadata loader and group-index path validate:

- metadata version,
- known recipe,
- integer head size,
- layer object shape,
- sorted unique high-group indices,
- index range,
- expected high-group count for the recipe.

Runtime metadata resolution also checks recipe and head-dimension match.

## Runtime Gate

```python
from turboquant.integration.vllm import validate_runtime_gate

info = validate_runtime_gate(
    cache_dtype="turboquant35",
    enabled=True,
    capability=(8, 6),
)
```

Supported CUDA capability checks are currently defined by
`recipe_cuda_allowed`.

## Runtime Dispatch

`select_decode_path` maps:

| Cache dtype | Decode path |
| --- | --- |
| `tq4`, `tq4_kv`, `k4v4` | `tq4_paged` |
| `turboquant25`, `turboquant35` | `recipe_paged` |

`select_prefill_path` chooses:

| Inputs | Path |
| --- | --- |
| no cache | dense live K/V |
| cache plus mixed prefill/decode | packed cache |
| cache plus INT8 prefill gate | int8 query path |

The INT8 prefill kernel is not yet present in this repository.

## Page Geometry

```python
from turboquant.integration.vllm import VLLMRuntimeConfig, build_page_geometry

config = VLLMRuntimeConfig(
    cache_dtype="tq4",
    enabled=True,
    model_name_or_path=None,
    metadata_path=None,
    num_q_heads=16,
    num_kv_heads=2,
    head_dim=128,
    block_size=16,
    k_bits=4,
    v_bits=4,
)

geometry = build_page_geometry(config, num_blocks=1024)
print(geometry.cache_shape())
```

For recipe mode, the page geometry uses recipe row byte overrides. For TQ4
mode, row bytes use the TQ formula and optional padding.

## Slot Mapping

`PackedPageCache.write_rows` and `write_tq4_kv` use flat slots. Negative slots
are ignored. A slot outside allocated pages raises.

```python
import torch

from turboquant.integration.vllm import PackedPageCache, PageGeometry

geometry = PageGeometry(
    num_blocks=2,
    block_size=4,
    num_kv_heads=2,
    head_dim=128,
    k_bits=4,
    v_bits=2,
)

cache = PackedPageCache(geometry=geometry, device=torch.device("cuda"))
```

## TQ Packed Page Contract

```python
import torch

from turboquant.core import RotationMode, build_rotation
from turboquant.core.codebook import get_codebook_tensors
from turboquant.triton import write_tq4_kv, decode_tq4_paged, tq_row_bytes

dim = 128
rotation = build_rotation(dim, RotationMode.RHT, torch.device("cuda"), seed=42)
centroids, _ = get_codebook_tensors(dim, 4, torch.device("cuda"))
row_bytes = tq_row_bytes(dim, 4)

pages = torch.zeros(32, 16, 2, row_bytes, dtype=torch.uint8, device="cuda")
raw_k = torch.randn(64, 2, dim, device="cuda")
slots = torch.arange(64, device="cuda")

write_tq4_kv(raw_k, pages, slots, rotation=rotation, centroids=centroids)
```

The current `write_tq4_kv` and `decode_tq4_paged` are contract/reference
implementations. Live vLLM backend registration is still tracked in
[vLLM Audit](../vllm_reference_audit.md).

## Verification Path

`scripts/verify_vllm_integration.py` can check:

- argument consistency,
- model cache shape detection,
- recipe metadata presence,
- recipe/head-dimension match,
- HF diagnostic cache cosine for TQ4-style modes.

Example:

```bash
python scripts/verify_vllm_integration.py \
  --model Qwen/Qwen2.5-3B \
  --bits 4 \
  --threshold 0.99 \
  --trust-remote-code
```

Recipe metadata check:

```bash
python scripts/verify_vllm_integration.py \
  --model Qwen/Qwen2.5-3B \
  --kv-cache-dtype turboquant35 \
  --metadata-path turboquant_kv.json
```

## Live Backend Gap

The reference quality target includes:

- vLLM attention backend class,
- custom cache spec,
- update kernel call from attention forward,
- decode kernel call through block tables,
- CUDA graph buffer ownership,
- live serving tests.

This repository has contracts for those pieces, but not all live vLLM hooks.
The audit page tracks the gap without masking it.

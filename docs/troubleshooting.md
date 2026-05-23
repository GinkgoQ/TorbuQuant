# Troubleshooting

Use this page to identify whether a failure is an installation issue, a shape
contract issue, a metadata issue, or a claim-boundary issue.

## Import Errors

If `transformers`, `triton`, or `vllm` is missing, install the needed package in
the active environment.

```bash
python -m pip install -e .
```

For docs:

```bash
python -m pip install -e ".[docs]"
```

## Metadata Errors

Recipe modes require metadata. If `resolve_metadata` raises a file error, pass
`metadata_path` or place `turboquant_kv.json` beside the model directory.

If metadata head dimension differs from runtime head dimension, regenerate
metadata for that model/config pair.

If group indices fail validation, check that each KV head has the correct
recipe count:

| Recipe | High-group count at head dim 128 |
| --- | ---: |
| `turboquant25` | 32 |
| `turboquant35` | 64 |

## Shape Errors

Most KV tensors use:

```text
[batch, heads, tokens, head_dim]
```

The packed-page TQ helpers use:

```text
[tokens, kv_heads, head_dim]
[blocks, block_size, kv_heads, row_bytes]
```

Check the function docstring before calling a path.

## Slot Mapping Errors

Negative slots are ignored. Non-negative slots outside allocated pages raise
`IndexError`.

For block size `B`:

```text
block = slot // B
offset = slot % B
```

If `block >= num_blocks`, allocate more pages or fix the caller's slot mapping.

## CUDA Errors

Some tests skip when CUDA or Triton kernels are unavailable. CPU reference tests
still validate layout and math contracts.

## Quality Drops

Try:

- higher key precision,
- K16V4 or K8V4 before K4V4,
- larger recent dense window,
- boundary-token preservation,
- real logits and retrieval checks,
- per-layer analysis.

Also check whether the route is compressing full-attention layers in a
sliding/global architecture. The HF wrapper can bypass full-attention layers
when `model_config.layer_types` exposes those labels.

## Documentation Build Issues

If MkDocs cannot find modules, run from the repository root. `mkdocs.yml` sets
mkdocstrings search paths to `"."`.

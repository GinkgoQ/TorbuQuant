# Configuration Reference

This page lists runtime configuration objects, metadata files, environment
variables, and script arguments that are currently implemented.

## `KVQuantConfig`

Defined in `turboquant.integration.common.config`.

| Field | Meaning |
| --- | --- |
| `preset` | Named K/V format selection such as `k16_v4`, `k8_v4`, or `k4_v4`. |
| `backend` | Runtime label such as `hf`, `vllm`, or `diagnostic`. |
| `mode` | `diagnostic` or `production`. |
| `context_length` | Target context length for reporting and policy construction. |
| `num_q_heads` | Query head count. |
| `num_kv_heads` | KV head count. |
| `head_dim` | Attention head dimension. |
| `recent_window` | Dense recent-token window size. |
| `fallback_count` | Number of dense fallback events to report. |
| `seed` | Transform and test seed. |
| `model_id` | HuggingFace model id for capture or generation scripts. |
| `layer_idx` | Layer index for capture helpers. |
| `local_files_only` | Pass-through flag for HuggingFace loaders. |

`policy_from_config` converts `KVQuantConfig` into a `KVQuantPolicy`. The policy
defines K format, V format, recent window, fallback behavior, and backend
capability.

## Known Preset Pattern

The test suite and scripts use presets such as:

| Preset | Intent |
| --- | --- |
| `k16_v4` | Dense/high precision keys and 4-bit values. |
| `k8_v4` | q8-style keys and 4-bit values. |
| `k4_v4` | 4-bit keys and 4-bit values. |

Unsupported presets should fail during policy construction or format
validation.

## vLLM Runtime Config

`VLLMRuntimeConfig` lives in `turboquant.integration.vllm.runtime`.

| Field | Meaning |
| --- | --- |
| `cache_dtype` | `tq4`, `tq4_kv`, `k4v4`, `turboquant25`, or `turboquant35`. |
| `enabled` | Must be true for TurboQuant cache dtypes. |
| `model_name_or_path` | Used for metadata discovery. |
| `metadata_path` | Optional path to `turboquant_kv.json`. |
| `num_q_heads` | Query head count. |
| `num_kv_heads` | KV head count. |
| `head_dim` | Head dimension. |
| `block_size` | vLLM page block size. |
| `k_bits` | TQ key bits for TQ4-style mode. |
| `v_bits` | TQ value bits for TQ4-style mode. |

## Recipe Config

Recipe constants live in `turboquant.integration.vllm.recipe`.

| Name | Value |
| --- | --- |
| `turboquant25` bits | `2.5` |
| `turboquant35` bits | `3.5` |
| group alignment | `16` |
| vector norm bytes | `2` |
| residual norm bytes | `2` |
| QJL scale | `sqrt(pi / 2)` |

Recipe group layout for `head_dim=128`:

| Recipe | High dim | Low dim | Packed row bytes |
| --- | ---: | ---: | ---: |
| `turboquant25` | 32 | 96 | 44 |
| `turboquant35` | 64 | 64 | 64 |

## Environment Variables

| Variable | Parser | Values |
| --- | --- | --- |
| `TQ4_K_BITS` | `parse_tq_bits_env` | `2`, `3`, `4`, `5`; default `4`. |
| `TQ4_V_BITS` | `parse_tq_bits_env` | `2`, `3`, `4`, `5`; default `4`. |
| `TQ4_USE_FUSED_PAGED` | `parse_tq_gate_env` | `1`, `true`, `yes` enable the gate. |
| `TQ4_USE_INT8_PREFILL` | `parse_tq_gate_env` | `1`, `true`, `yes` enable the gate. |

## Metadata JSON

Recipe metadata contains:

```json
{
  "version": 1,
  "recipe": "turboquant35",
  "head_size": 128,
  "model_name": "Qwen/Qwen2.5-3B",
  "transform_version": "structured_hadamard_v1",
  "codebook_version": "lloyd_beta_v1",
  "layers": {
    "model.layers.0.self_attn": {
      "key_high_precision_indices": [[0, 1, 2]],
      "value_high_precision_indices": [[0, 1, 2]]
    }
  },
  "calibration": {
    "method": "activation_energy_v1",
    "objective": "sum_squared_activation",
    "num_prompts": 4,
    "max_seq_len": 512,
    "batch_size": 1,
    "num_observed_tokens": 2048,
    "dtype": "bfloat16",
    "device": "cuda",
    "prompts_sha256": "..."
  }
}
```

The example truncates index lists. Real metadata must contain the recipe-defined
number of indices for each KV head.

## Metadata Discovery

`discover_metadata_path(model_name_or_path, explicit_path)` follows this order:

1. return the explicit path when provided,
2. if `model_name_or_path` is a local file, use its parent directory,
3. if it is a local directory, look for `turboquant_kv.json`,
4. otherwise return `None`.

Remote HuggingFace model ids do not imply a metadata file.

## Tensor Parallel Slicing

`slice_layer_metadata_for_tp` supports:

- partitioned metadata when total metadata KV heads divide TP size,
- replicated metadata when TP size divides metadata head count.

Invalid rank, size, or KV head count combinations raise `ValueError`.

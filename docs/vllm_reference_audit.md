# vLLM Reference Audit

This audit maps the local vLLM-related reference work to `torbuquant`.
Each later vLLM phase must update the affected row before the phase is treated as done.

Status values:

- `present`: implemented and tested in this repository.
- `partial`: some behavior exists, but a listed runtime path or test group is absent.
- `missing`: no implementation exists here yet.
- `rejected`: not being adopted, with a reason.

## File Map

| Reference file | Feature name | Data structures | Public functions/classes | Runtime path | Tests | Our target | Status |
|---|---|---|---|---|---|---|---|
| `turboquant-vllm/src/turboquant_vllm/lloyd_max.py` | Lloyd-Max codebooks | codebook centroids, boundaries | `solve_lloyd_max`, `LloydMaxCodebook` | quantizer setup | `test_lloyd_max.py` | `torbuquant/core/codebook.py` | present |
| `turboquant-vllm/src/turboquant_vllm/quantizer.py` | MSE and QJL codecs | indices, norms, signs | `TurboQuantMSE`, `TurboQuantProd` | HF and TQ4 diagnostics | `test_quantizer.py` | `torbuquant/core/` | present |
| `turboquant-vllm/src/turboquant_vllm/compressors.py` | key/value compressors | compressed keys, compressed values | key and value compressor classes | diagnostic compression | `test_compressors.py` | `torbuquant/kv/keys.py`, `torbuquant/kv/values.py` | partial |
| `turboquant-vllm/src/turboquant_vllm/kv_cache.py` | HF dynamic cache wrapper | compressed layers, lifecycle state | dynamic cache wrapper, stats methods | HF diagnostic path | `test_kv_cache_*.py` | `torbuquant/integration/hf/dynamic_cache.py` | partial |
| `turboquant-vllm/src/turboquant_vllm/verify.py` | model verification | model shape record, per-layer metrics | config detector, verification runner | CLI quality check | `test_verify*.py` | `torbuquant/integration/vllm/verify.py` | partial |
| `turboquant-vllm/src/turboquant_vllm/benchmark.py` | A/B measurement | run result dicts | model load, patched run, benchmark runner | local benchmark | `test_benchmark.py` | `scripts/qwen_ab_eval.py` | partial |
| `turboquant-vllm/src/turboquant_vllm/triton/tq4_compress.py` | TQ4 pack kernel | packed index bytes, fp32 norms | `tq4_compress` | packed cache write | `test_triton_multi_dim.py` | `torbuquant/triton/tq4_update.py` | partial |
| `turboquant-vllm/src/turboquant_vllm/triton/tq4_decompress.py` | TQ4 unpack kernel | packed index bytes, norms | `tq4_decompress` | diagnostic readback | `test_tq4_out_parameter.py` | `torbuquant/triton/tq4_decode.py` | partial |
| `turboquant-vllm/src/turboquant_vllm/triton/fused_qk_attention.py` | packed score kernel | packed key rows | `fused_qk_scores` | score-only diagnostic | kernel tests | `torbuquant/triton/tq4_decode.py` | missing |
| `turboquant-vllm/src/turboquant_vllm/triton/fused_paged_tq4_attention.py` | TQ4 paged decode | block table, packed pages | `fused_paged_tq4_decode` | decode attention | `test_fused_paged_tq4_attention.py` | `torbuquant/triton/tq4_decode.py` | partial |
| `turboquant-vllm/src/turboquant_vllm/triton/fused_paged_tq4_int8_prefill.py` | INT8-Q prefill | int8 query scale, packed pages | prefill wrapper | prefill experiment | `test_fused_paged_tq4_int8.py` | `torbuquant/triton/int8_prefill.py` | missing |
| `turboquant-vllm/src/turboquant_vllm/vllm/tq4_backend.py` | plugin-style backend | TQ4 spec, backend, impl | backend registration, page shape, dispatch | vLLM plugin path | `test_vllm_*.py` | `torbuquant/integration/vllm/` | partial |
| `vllm-turboquant/vllm/v1/attention/ops/turboquant_kv_cache.py` | recipe layout and vector codec | group layout, kernel meta | layout, transforms, pack/unpack | native vLLM recipe path | `test_turboquant.py` | `torbuquant/integration/vllm/recipe.py`, `vector.py` | partial |
| `vllm-turboquant/vllm/v1/attention/ops/turboquant_metadata.py` | metadata JSON | tensor/layer/calibration metadata | load, save, discovery, TP slicing | native vLLM recipe path | `test_turboquant.py` | `torbuquant/integration/vllm/metadata.py` | partial |
| `vllm-turboquant/vllm/v1/attention/ops/triton_turboquant_kv_update.py` | recipe update kernel | packed recipe page rows | `turboquant_write_packed_kv` | KV cache update | `test_turboquant.py` | `torbuquant/triton/recipe_update.py` | missing |
| `vllm-turboquant/vllm/v1/attention/ops/triton_turboquant_decode.py` | recipe decode kernel | block table, token metadata, packed pages | decode forward wrapper | paged attention | `test_turboquant.py` | `torbuquant/triton/recipe_decode.py` | missing |
| `vllm-turboquant/vllm/v1/attention/backends/triton_attn.py` | backend integration | metadata builder, layer impl state | update, decode, fallback gates | vLLM runtime | `test_turboquant.py` | `torbuquant/integration/vllm/runtime.py` | missing |
| `vllm-turboquant/vllm/config/cache.py` | cache config gate | cache dtype, gate, metadata path | config validators | vLLM config | config tests | `torbuquant/integration/vllm/registry.py` | missing |
| `vllm-turboquant/vllm/engine/arg_utils.py` | CLI flags | engine args | enable flag, metadata path | vLLM CLI | config tests | `scripts/verify_vllm_integration.py` | missing |
| `vllm-turboquant/benchmarks/generate_turboquant_metadata.py` | activation calibration | activation accumulator, metadata builder | metadata generator CLI | recipe metadata build | `test_turboquant.py` | `torbuquant/integration/vllm/calibration.py`, `scripts/generate_vllm_metadata.py` | missing |

## Feature Checklist

| Feature group | Our target | Status | Next action |
|---|---|---|---|
| Lloyd-Max codebooks | `torbuquant/core/codebook.py` | present | keep exact Beta tests |
| QR/RHT/block-Hadamard transforms | `torbuquant/core/rotation.py`, `torbuquant/integration/vllm/vector.py` | partial | expose query/group helpers |
| QJL residual | `torbuquant/core/qjl.py`, `torbuquant/integration/vllm/vector.py` | partial | add recipe decode contract |
| bit packing | `torbuquant/packing/bits.py` | present | keep 1/2/3/4/5-bit tests |
| TQ4 cache wrapper | `torbuquant/integration/hf/dynamic_cache.py` | partial | add Transformers model tests |
| TQ4 packed page layout | `torbuquant/integration/vllm/page_cache.py` | present | add kernel-backed writer |
| asymmetric K/V bits | `torbuquant/integration/vllm/recipe.py`, page cache | partial | add page bytes and tests |
| `turboquant25/35` recipe layout | `torbuquant/integration/vllm/recipe.py` | partial | add public aliases from plan |
| metadata JSON | `torbuquant/integration/vllm/metadata.py` | partial | validate recipe and runtime shape |
| activation calibration | `torbuquant/integration/vllm/calibration.py` | present | add model-family tests |
| TP metadata slicing | `torbuquant/integration/vllm/metadata.py` | present | expand tests |
| vLLM config hooks | `torbuquant/integration/vllm/registry.py` | missing | add gate checks |
| vLLM cache dtype registry | `torbuquant/integration/vllm/registry.py` | missing | map dtype to uint8 |
| vLLM attention backend selection | `torbuquant/integration/vllm/runtime.py` | missing | add dispatch layer |
| packed KV update kernel | `torbuquant/triton/tq4_update.py`, `recipe_update.py` | missing | add reference then Triton |
| packed paged decode kernel | `torbuquant/triton/tq4_decode.py`, `recipe_decode.py` | missing | add reference then Triton |
| INT8-Q prefill | `torbuquant/triton/int8_prefill.py` | missing | add gated experiment |
| CUDA graph buffers | `torbuquant/integration/vllm/runtime.py` | missing | add buffer owner |
| SWA/global-layer behavior | HF and vLLM adapters | missing | add config tests |
| model verification | `torbuquant/integration/vllm/verify.py` | partial | add real model smoke |
| Qwen A/B benchmark | `scripts/qwen_ab_eval.py` | partial | add vLLM variants |

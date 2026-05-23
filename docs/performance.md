# Performance Notes

Performance in this project is not a single number. The repository separates
storage bytes, kernel time, end-to-end generation time, and capacity under a
memory budget.

## Claim Rules

Do not claim speed or memory gains without measured evidence. A valid report
names:

- baseline path,
- TurboQuant path,
- model and revision,
- GPU and software versions,
- prompt/context lengths,
- warmup and repeat counts,
- CUDA synchronization policy,
- peak memory,
- raw outputs for quality comparisons.

## What Compression Can Improve

Packed KV storage can increase the number of tokens, requests, or context
length that fit under a memory budget. That does not guarantee lower
single-request latency.

Latency depends on:

- packed row decode cost,
- memory traffic reduction,
- attention kernel structure,
- batch size,
- context length,
- whether historical K/V is materialized.

## External vLLM Evidence

The public vLLM TurboQuant study reports a clear distinction between capacity
and per-token speed:

- FP8 KV cache stores KV in FP8 and uses FP8 attention compute on supported
  hardware.
- TurboQuant stores lower-bit KV, then dequantizes for BF16 attention in the
  vLLM path described by the study.
- FP8 is the recommended default for standard serving.
- `turboquant_4bit_nc` is the main TurboQuant candidate when more KV capacity
  is required than FP8 provides.
- TurboQuant can reduce P99 TTFT when BF16 queues requests because KV memory is
  saturated, but it usually worsens TPOT and throughput versus FP8.

Google's public TurboQuant speed claim is about attention-logit computation,
not whole-model generation. The benchmarked sub-operation is:

```text
query x cached keys -> attention scores
```

That sub-operation is important at long context, but generation also includes
projection layers, softmax, value accumulation, MLP blocks, normalization,
sampling, scheduling, and runtime overhead.

For real vLLM comparisons in this project, use
[vLLM FP8 And TurboQuant Benchmarking](usage/vllm-fp8-turboquant.md).

## Measurement Layers

| Layer | Measures | Cannot prove |
| --- | --- | --- |
| Tensor reconstruction | MSE, cosine, max error. | Model quality. |
| Attention reference | Output cosine, score KL, runtime of local path. | End-to-end serving behavior. |
| Generation A/B | Raw text, token match, resource metrics. | Throughput under continuous batching unless batched serving is measured. |
| Kernel benchmark | Kernel time and memory traffic. | User-visible latency alone. |
| Serving benchmark | Requests/sec, tokens/sec, max batch/context under budget. | Mathematical quality by itself. |

## Current Measured Evidence In This Repository

`VALIDATION_REPORT.md` records Qwen/Qwen2.5-3B activation-level measurements for
K16V4, K8V4, and K4V4. Those results are layer-level and attention-output
measurements, not end-to-end serving throughput proof.

`reports/qwen_ab/` contains Qwen A/B report artifacts produced by the local
evaluation script.

## Diagnostic Overhead

HuggingFace diagnostic wrappers reconstruct dense K/V. They can measure quality
impact and byte estimates, but they are not valid evidence for production
serving throughput.

## Memory Formulas

Dense KV for one component:

```text
tokens * kv_heads * head_dim * element_bytes
```

TQ row storage for one component:

```text
tokens * kv_heads * (ceil(head_dim * bits / 8) + norm_bytes)
```

Value group storage:

```text
payload bytes + scale bytes + zero bytes
```

A valid report also includes metadata and workspace. A storage-only ratio is a
layout number; it is not a CUDA peak-memory result.

## vLLM Serving Route

The intended serving route is packed paged cache update plus packed paged decode
attention. The live vLLM backend is still tracked as open work in
[vLLM Audit](vllm_reference_audit.md).

## Interpreting A Slow Result

If a compressed path is slower than dense, the result is still useful:

- it may reduce memory while increasing latency,
- it may support longer context under a memory budget,
- it may support larger batch size,
- it may expose a kernel bottleneck.

The report should say this plainly and avoid converting memory reduction into a
speed claim.

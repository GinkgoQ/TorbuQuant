# Limits And Assumptions

This page is part of the measurement contract. It exists so users can tell the
difference between a supported API, a diagnostic check, and an open serving
piece.

## Mathematical Limits

The paper-level vector guarantees assume a random rotation model and scalar
quantization of rotated unit-vector coordinates. They do not prove:

- logits preservation,
- perplexity preservation,
- generated text similarity,
- long-context retrieval quality,
- throughput gains.

## Current Implementation Limits

- HuggingFace cache wrappers reconstruct dense K/V for attention.
- vLLM live backend registration is not wired end to end in this repository.
- TQ packed page update/decode includes a PyTorch contract path; selected older
  non-paged kernels exist under `turboquant.triton.kernels`.
- Recipe update/decode Triton kernels are not present.
- INT8-Q prefill is not present.
- CUDA graph buffer ownership is not present.

## Claim Boundaries

| Claim | Required evidence |
| --- | --- |
| Storage reduction | Packed bytes plus norms, scales, zeros, metadata, recent window, and workspace. |
| Peak-memory reduction | CUDA peak memory from a controlled run. |
| Model quality | Dense-reference logits, generated text, perplexity, or retrieval task. |
| Serving gain | End-to-end serving run with named dense and q8/FP8 baselines when available. |
| Kernel gain | Kernel timing only; not an end-to-end claim. |

## Quality Risks

- Keys are more sensitive than values.
- Low-bit key compression can shift attention routing.
- Softmax can amplify score perturbations.
- Value error can change the token distribution even when score error looks
  small.
- Short prompts can hide long-context failures.
- QJL raw inner-product benefits may not translate to attention quality.

## Operational Assumptions

- Qwen/Qwen2.5-3B is the primary model target.
- CUDA GPU execution is the main runtime target.
- vLLM-style packed paged decode is the intended serving path.
- Reports must include fallback counters and byte accounting.

## Unclear Or Environment-Dependent Areas

- Whether a given model family tolerates K4 keys at long context is empirical.
- Whether FP8/q8 baselines are available depends on the serving stack.
- CUDA graph support depends on runtime buffer ownership and backend wiring.
- Remote model metadata discovery is not implemented; metadata is local.

# Usage

TorbuQuant has four practical usage layers:

1. Core vector quantization for tensor experiments.
2. KV cache construction and byte accounting.
3. Diagnostic HuggingFace cache wrappers and capture scripts.
4. vLLM-oriented metadata and packed page contracts.

The current serving-critical warning is simple: HuggingFace wrappers reconstruct
dense K/V for attention. They are useful for correctness checks, not for
production throughput claims.

See:

- [HuggingFace](huggingface.md)
- [vLLM](vllm.md)
- [Qwen A/B](qwen-ab.md)


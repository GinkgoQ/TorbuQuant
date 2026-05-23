# Qwen Context A/B Report

- Run id: `qwen_ab_20260522_k8v4_8192`
- Model: `Qwen/Qwen2.5-3B`
- TurboQuant preset: `k8_v4`
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- PyTorch: 2.9.1+cu128
- CUDA: 12.8
- Settings: `{"do_sample": false, "max_new_tokens": 2, "pad_token_id": 151643, "return_dict_in_generate": false, "use_cache": true}`

| Scenario | Input tokens | New tokens | Output match | HF runtime s | TQ runtime s | HF tok/s | TQ tok/s | HF peak CUDA | TQ peak CUDA | TQ cache ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context_8192 | 8192 | 2 | True | 1.306561 | 6.071994 | 1.5307 | 0.3294 | 7276948480 | 7107079168 | 2.2770 |

## context_8192

- HF generated text: ` \item`
- TurboQuant generated text: ` \item`
- Token match ratio: 1.000000
- Runtime ratio TQ over HF: 4.647310
- Throughput ratio TQ over HF: 0.215178
- CUDA peak ratio TQ over HF: 0.976657
- Cache storage ratio dense over TQ: 2.276970
- HF CPU mean/max: 99.15384615384616/117.5 percent
- TQ CPU mean/max: 101.33305084745764/134.0 percent
- HF RAM RSS max: 2346184704 bytes
- TQ RAM RSS max: 3211345920 bytes
- HF GPU use mean/max: 63.84615384615385/100 percent
- TQ GPU use mean/max: 48.11864406779661/56 percent

## Outcome

- Output tokens matched exactly.
- TurboQuant cache ledger shows storage reduction.
- This Hugging Face diagnostic path does not show runtime gain.
- CUDA peak allocation changes are small relative to model weights because dense tensors are returned to Hugging Face attention.

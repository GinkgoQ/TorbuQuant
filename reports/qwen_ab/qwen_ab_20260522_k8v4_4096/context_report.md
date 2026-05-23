# Qwen Context A/B Report

- Run id: `qwen_ab_20260522_k8v4_4096`
- Model: `Qwen/Qwen2.5-3B`
- TurboQuant preset: `k8_v4`
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- PyTorch: 2.9.1+cu128
- CUDA: 12.8
- Settings: `{"do_sample": false, "max_new_tokens": 4, "pad_token_id": 151643, "return_dict_in_generate": false, "use_cache": true}`

| Scenario | Input tokens | New tokens | Output match | HF runtime s | TQ runtime s | HF tok/s | TQ tok/s | HF peak CUDA | TQ peak CUDA | TQ cache ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context_4096 | 4096 | 4 | True | 0.863363 | 5.194287 | 4.6330 | 0.7701 | 6786083840 | 6701149184 | 2.2762 |

## context_4096

- HF generated text: `kvquant, z`
- TurboQuant generated text: `kvquant, z`
- Token match ratio: 1.000000
- Runtime ratio TQ over HF: 6.016344
- Throughput ratio TQ over HF: 0.166214
- CUDA peak ratio TQ over HF: 0.987484
- Cache storage ratio dense over TQ: 2.276189
- HF CPU mean/max: 96.88235294117646/130.8 percent
- TQ CPU mean/max: 101.0019801980198/122.4 percent
- HF RAM RSS max: 2352754688 bytes
- TQ RAM RSS max: 3075571712 bytes
- HF GPU use mean/max: 41.88235294117647/100 percent
- TQ GPU use mean/max: 41.2970297029703/53 percent

## Outcome

- Output tokens matched exactly.
- TurboQuant cache ledger shows storage reduction.
- This Hugging Face diagnostic path does not show runtime gain.
- CUDA peak allocation changes are small relative to model weights because dense tensors are returned to Hugging Face attention.

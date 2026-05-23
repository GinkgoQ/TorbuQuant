# Qwen A/B Comparison Report

## Run Setup

- Run id: `qwen_ab_20260522_k8v4`
- Model: `Qwen/Qwen2.5-3B`
- TurboQuant preset: `k8_v4`
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- Python: 3.10.12
- PyTorch: 2.9.1+cu128
- CUDA: 12.8
- Generation settings: `{"do_sample": false, "max_new_tokens": 16, "pad_token_id": 151643, "return_dict_in_generate": false, "use_cache": true}`

The Hugging Face run uses the standard generation path. The TurboQuant run uses the Hugging Face diagnostic adapter, which stores compressed cache state but returns dense K/V tensors to Hugging Face attention. This run can measure output behavior and cache-ledger storage, but observed CUDA peak memory is still dominated by the dense HF path.

## Scenario Results

| Scenario | Input tokens | New tokens | Output match | Token match | HF runtime s | TQ runtime s | HF tok/s | TQ tok/s | HF peak CUDA bytes | TQ peak CUDA bytes | TQ cache ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_prompt | 64 | 16 | True | 1.0000 | 0.678952 | 4.114642 | 23.5657 | 3.8886 | 6302888960 | 6301561856 | 2.0828 |
| medium_context | 256 | 16 | True | 1.0000 | 0.282133 | 4.237414 | 56.7108 | 3.7759 | 6325898240 | 6320589824 | 2.2180 |
| long_context | 1024 | 16 | True | 1.0000 | 0.334677 | 6.598022 | 47.8073 | 2.4250 | 6419508224 | 6398274560 | 2.2614 |

## short_prompt

- Model: `Qwen/Qwen2.5-3B`
- Input length: 64 tokens
- Settings: `{"do_sample": false, "max_new_tokens": 16, "pad_token_id": 151643, "return_dict_in_generate": false, "use_cache": true}`
- HF raw output file: `short_prompt/hf/raw_output.txt`
- TurboQuant raw output file: `short_prompt/tq/raw_output.txt`
- HF generated text: ` paragraphs why KV-cache compression can help serving large language models when context length grows.

`
- TurboQuant generated text: ` paragraphs why KV-cache compression can help serving large language models when context length grows.

`
- GPU use mean/max HF: 25.714285714285715/56 percent
- GPU use mean/max TQ: 39.27160493827161/56 percent
- CPU mean/max HF: 91.87857142857142/117.5 percent
- CPU mean/max TQ: 100.4172839506173/134.4 percent
- RAM RSS max HF/TQ: 2353524736 / 3208740864 bytes
- CUDA peak HF/TQ: 6302888960 / 6301561856 bytes
- Runtime HF/TQ: 0.678952 / 4.114642 s
- Latency HF/TQ: 0.042435 / 0.257165 s per generated token
- Throughput HF/TQ: 23.565726 / 3.888553 generated tokens/s
- Cache storage ratio dense over TQ: 2.082797
- Runtime ratio TQ over HF: 6.060282
- CUDA peak ratio TQ over HF: 0.999789
- Observed issues: none in output tokens; TQ path is slower in HF diagnostic mode

## medium_context

- Model: `Qwen/Qwen2.5-3B`
- Input length: 256 tokens
- Settings: `{"do_sample": false, "max_new_tokens": 16, "pad_token_id": 151643, "return_dict_in_generate": false, "use_cache": true}`
- HF raw output file: `medium_context/hf/raw_output.txt`
- TurboQuant raw output file: `medium_context/tq/raw_output.txt`
- HF generated text: `package{tikz}
\usepackage{tikz-cd}
`
- TurboQuant generated text: `package{tikz}
\usepackage{tikz-cd}
`
- GPU use mean/max HF: 56.5/64 percent
- GPU use mean/max TQ: 42.40963855421687/59 percent
- CPU mean/max HF: 82.93333333333332/116.2 percent
- CPU mean/max TQ: 100.86265060240966/122.0 percent
- RAM RSS max HF/TQ: 3337687040 / 3649581056 bytes
- CUDA peak HF/TQ: 6325898240 / 6320589824 bytes
- Runtime HF/TQ: 0.282133 / 4.237414 s
- Latency HF/TQ: 0.017633 / 0.264838 s per generated token
- Throughput HF/TQ: 56.710787 / 3.775887 generated tokens/s
- Cache storage ratio dense over TQ: 2.217955
- Runtime ratio TQ over HF: 15.019194
- CUDA peak ratio TQ over HF: 0.999161
- Observed issues: none in output tokens; TQ path is slower in HF diagnostic mode

## long_context

- Model: `Qwen/Qwen2.5-3B`
- Input length: 1024 tokens
- Settings: `{"do_sample": false, "max_new_tokens": 16, "pad_token_id": 151643, "return_dict_in_generate": false, "use_cache": true}`
- HF raw output file: `long_context/hf/raw_output.txt`
- TurboQuant raw output file: `long_context/tq/raw_output.txt`
- HF generated text: ` \usepackage{tikz}
% \usetikzlibrary{ar`
- TurboQuant generated text: ` \usepackage{tikz}
% \usetikzlibrary{ar`
- GPU use mean/max HF: 71.28571428571429/94 percent
- GPU use mean/max TQ: 42.78294573643411/50 percent
- CPU mean/max HF: 123.38571428571429/244.1 percent
- CPU mean/max TQ: 101.1100775193799/136.4 percent
- RAM RSS max HF/TQ: 3724640256 / 3770105856 bytes
- CUDA peak HF/TQ: 6419508224 / 6398274560 bytes
- Runtime HF/TQ: 0.334677 / 6.598022 s
- Latency HF/TQ: 0.020917 / 0.412376 s per generated token
- Throughput HF/TQ: 47.807334 / 2.424969 generated tokens/s
- Cache storage ratio dense over TQ: 2.261389
- Runtime ratio TQ over HF: 19.714614
- CUDA peak ratio TQ over HF: 0.996692
- Observed issues: none in output tokens; TQ path is slower in HF diagnostic mode

## Outcome

- Real cache-ledger memory reduction: yes, inside the TurboQuant diagnostic cache state.
- Observed CUDA peak memory reduction: not meaningful in this HF diagnostic path because dense tensors are returned to HF attention.
- Output preservation: yes for all measured scenarios; generated token sequences matched exactly.
- Runtime gain: no. TurboQuant diagnostic generation was slower in all scenarios.
- Large-context behavior: the 1024-token scenario preserved output and showed about 2.26x cache-ledger storage reduction, but runtime was worse in HF diagnostic mode.

## Artifact Isolation

HF and TurboQuant artifacts are stored in separate directories under each scenario. The combined manifest is in `report.json`.

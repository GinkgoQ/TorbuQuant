# vLLM FP8 And TurboQuant Benchmarking

This page defines how to compare the upstream vLLM FP8 KV cache path with the
upstream vLLM TurboQuant KV cache path. It is written for real serving tests,
not tensor-only checks.

## Source Reading

The current public vLLM study separates FP8 and TurboQuant by the execution
path, not only by storage size. FP8 stores KV in FP8 and uses FP8 attention
compute on supported hardware. TurboQuant stores KV in a lower-bit packed form
and, in the vLLM implementation discussed by the study, dequantizes back to
BF16 for attention compute. That means TurboQuant can increase KV capacity, but
it can also add decode overhead. See the vLLM study and API docs:

- [vLLM TurboQuant study, May 11 2026](https://vllm-project.github.io/2026/05/11/turboquant.html)
- [vLLM TurboQuant API docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/)

Google's public TurboQuant article reports up to 8x improvement for attention
logit computation on H100, measured relative to an optimized JAX baseline. That
number describes a sub-operation inside attention, not full generation
throughput. See:

- [Google Research TurboQuant article](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## What The Speed Claim Means

TurboQuant has shown speed-related gains in three different contexts. Only one
is close to LLM serving.

| Context | What improves | Whole-model generation? |
| --- | --- | --- |
| Google benchmark | Attention-logit computation from cached keys. | No. |
| Vector-search benchmark | Vector compression / index setup time versus PQ-style methods. | No. |
| vLLM burst serving | P99 TTFT when BF16 runs out of KV cache and queues requests. | Not per-token decode. |
| vLLM normal serving | Usually no gain; TPOT and throughput are worse than BF16/FP8 in the vLLM study. | No. |

Attention-logit computation is the operation:

```text
current query vector x historical key vectors -> attention scores
```

At long context, this operation reads many cached key vectors. Compressed keys
can reduce memory traffic for that sub-operation. This does not mean the whole
decoder has lower latency, because generation also includes projections, softmax,
value accumulation, MLP blocks, normalization, sampling, scheduling, and
runtime overhead.

## Current vLLM Interpretation

The public vLLM study reports the following practical pattern:

- FP8 KV cache is the default recommendation for standard serving because it
  doubles KV capacity and avoids the low-bit dequantization overhead seen in
  TurboQuant.
- `turboquant_4bit_nc` is the main TurboQuant candidate when FP8 does not
  provide enough KV capacity.
- `turboquant_k8v4` gives only modest extra capacity over FP8 in that study.
- `turboquant_k3v4_nc` and `turboquant_3bit_nc` need workload-specific quality
  checks before use.

The honest claim to test is:

```text
TurboQuant may reduce queueing delay when KV memory is saturated, but it can
slow per-token decode.
```

Do not convert storage compression into a speed claim.

## First Evaluation Target

The first repo-level external benchmark should compare:

| Role | vLLM setting |
| --- | --- |
| FP8 baseline | `--kv-cache-dtype fp8` |
| TurboQuant candidate | `--kv-cache-dtype turboquant_4bit_nc` |
| Optional TurboQuant reference | `--kv-cache-dtype turboquant_k8v4` |
| Avoid in first pass | `turboquant_k3v4_nc`, `turboquant_3bit_nc` |

Use one model at a time. Keep the model, tokenizer, tensor parallel size,
prompt set, context length, generation settings, scheduler settings, and
hardware unchanged across runs.

## Quality Tasks

Start with two 8k tasks:

| Task | Target length | Why it matters |
| --- | ---: | --- |
| LongBench 8k bucket | 8k-ish inputs | Broader long-context behavior. |
| Needle-in-a-Haystack | 8k tokens | Direct retrieval stress check. |

For NIAH, record:

- input token length,
- needle text,
- insertion depth,
- generated answer,
- pass/fail rule,
- exact token budget,
- output token budget,
- random seed.

For LongBench, record:

- dataset subset,
- task name,
- prompt template,
- context token length,
- reference answer,
- metric script and version,
- per-example output,
- aggregate score.

## Serving Tasks

Run serving separately from quality scoring. Use `vllm bench serve` or a custom
request driver against `vllm serve`.

Measure three load regimes:

| Regime | Purpose |
| --- | --- |
| Low request rate | Shows per-request overhead without queue pressure. |
| Moderate request rate | Shows scheduler behavior before saturation. |
| Burst / unlimited rate | Shows whether compressed KV reduces queueing. |

The vLLM study reports the most meaningful TurboQuant serving gain in burst
conditions where BF16 KV memory is saturated. Against FP8, TurboQuant must be
measured rather than assumed.

## Run Isolation

Each variant gets a separate output directory:

```text
runs/vllm_fp8_tq/
  manifest.json
  fp8/
    server.log
    bench.json
    outputs.jsonl
    env.json
  turboquant_4bit_nc/
    server.log
    bench.json
    outputs.jsonl
    env.json
```

The manifest must contain:

- model id and revision,
- vLLM version or git commit,
- CUDA, driver, PyTorch, and Triton versions,
- GPU name and count,
- tensor parallel size,
- `--max-model-len`,
- `--gpu-memory-utilization`,
- `--kv-cache-dtype`,
- request dataset path,
- generation settings,
- run command.

Do not reuse output files between variants. Do not merge server logs.

## Server Commands

FP8:

```bash
vllm serve MODEL_ID \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --tensor-parallel-size TP_SIZE \
  --gpu-memory-utilization 0.90
```

TurboQuant:

```bash
vllm serve MODEL_ID \
  --kv-cache-dtype turboquant_4bit_nc \
  --max-model-len 8192 \
  --tensor-parallel-size TP_SIZE \
  --gpu-memory-utilization 0.90
```

If the model needs remote code or a specific dtype, add the same flags to both
commands.

## Throughput Command

Use the same request shape for both variants:

```bash
vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model MODEL_ID \
  --dataset-name random \
  --input-len 8192 \
  --output-len 256 \
  --num-prompts 200 \
  --request-rate 2 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result \
  --save-detailed
```

Repeat with:

```text
--request-rate 8
--request-rate inf
```

If the selected vLLM version supports a JSON output flag, write one JSON file
per variant and request rate. Otherwise capture stdout plus server logs and
normalize them into a local report file.

## Repo Runner For 8k

The repository includes a runner for the first 8k FP8-versus-TurboQuant pass:

```bash
python scripts/vllm_fp8_tq_bench.py \
  --model MODEL_ID \
  --variants fp8,tq4 \
  --context-len 8192 \
  --output-len 256 \
  --num-prompts 64 \
  --request-rates 1,4,inf \
  --tensor-parallel-size TP_SIZE \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16
```

Run it from the Python environment that contains the intended vLLM build. The
runner invokes vLLM through that same Python executable, starts one server per
KV-cache dtype, keeps separate output directories, runs `vllm bench serve`,
queries `/metrics`, samples process and GPU resources, and writes:

`--context-len` is the total model context budget. Unless `--input-len` is set,
the runner sends `context_len - output_len` input tokens to `vllm bench serve`
so each request fits the server `--max-model-len`.

```text
reports/vllm_fp8_tq/RUN_ID/
  manifest.json
  report.json
  fp8/
    server.log
    variant_report.json
    bench_*.stdout.txt
    bench_*.stderr.txt
  tq4/
    server.log
    variant_report.json
    bench_*.stdout.txt
    bench_*.stderr.txt
```

By default the runner also executes an 8k Needle-in-a-Haystack streaming probe
against each server and stores the raw output. Disable it only when measuring
serving throughput in isolation:

```bash
python scripts/vllm_fp8_tq_bench.py \
  --model MODEL_ID \
  --variants fp8,tq4 \
  --context-len 8192 \
  --no-run-niah
```

The runner checks `vllm serve --help=kv-cache-dtype` before starting servers.
If the installed vLLM build lists `fp8` but not `turboquant_4bit_nc`, install a
vLLM build that contains the upstream TurboQuant cache dtypes before running
the comparison. Do not bypass this check unless you are testing a fork whose
CLI help is known to be out of date:

```bash
python scripts/vllm_fp8_tq_bench.py \
  --model MODEL_ID \
  --variants fp8,tq4 \
  --skip-cli-check
```

The report contains measured values only. If a vLLM build does not expose KV
memory bytes directly through metrics or logs, the runner preserves the raw
cache metrics and server log fields instead of inventing a byte number.

## Run Report Generator

After a run finishes, build a Markdown report with plots from the saved
artifacts:

```bash
python scripts/vllm_run_report.py reports/vllm_fp8_tq/RUN_ID
```

The script reads:

- `manifest.json`,
- `report.json`,
- per-variant `variant_report.json` files,
- per-rate benchmark JSON files referenced by the report,
- quality probe JSON files embedded in the report,
- server-log capacity fields already normalized by the runner,
- Prometheus snapshots saved before and after the benchmark,
- resource summaries saved while the server was running.

It writes:

```text
reports/vllm_fp8_tq/RUN_ID/
  run_report.md
  plots/
    ttft_ms.svg
    tpot_ms.svg
    e2el_ms.svg
    output_tps.svg
    request_tps.svg
    kv_cache_tokens.svg
    peak_gpu_gib.svg
    kv_memory_gib.svg
```

Use `--output` or `--plots-dir` when publishing the report elsewhere:

```bash
python scripts/vllm_run_report.py reports/vllm_fp8_tq/RUN_ID \
  --output reports/vllm_fp8_tq/RUN_ID/run_report.md \
  --plots-dir reports/vllm_fp8_tq/RUN_ID/plots
```

Use `--reference` to compare every other variant against a non-FP8 reference:

```bash
python scripts/vllm_run_report.py reports/vllm_fp8_tq/RUN_ID \
  --reference bf16
```

The generated report includes run identity, environment, server commands,
cache capacity, latency, raw distribution checks from `ttfts` and `itls`,
throughput, per-variant ratios, resource samples, Prometheus cache metrics,
quality probes, generated-text excerpts, error counts, run limits, and an
artifact index.

## Metrics To Compare

| Metric | Interpretation |
| --- | --- |
| TTFT p50/p90/p99 | Queueing and first-token behavior. |
| TPOT p50/p90/p99 | Decode cost. |
| ITL p50/p90/p99 | Inter-token spacing. |
| Requests/sec | Serving throughput. |
| Output tokens/sec | Decode throughput. |
| Peak GPU memory | Actual memory pressure. |
| KV cache capacity | Reported or estimated KV capacity. |
| Error count | Runtime instability. |
| Quality score | Task behavior at the same context length. |

The comparison is valid only when the same prompt set and generation settings
are used.

## Report Rules

Use these rules in reports:

- If FP8 has lower latency and acceptable quality, report FP8 as the default result.
- If TurboQuant uses less KV memory but has worse TPOT, say so directly.
- If TurboQuant lowers P99 TTFT only at burst load, attribute the gain to
  queueing reduction, not lower token-compute cost.
- If a TurboQuant variant fails a model or context length, record the error and
  do not replace it with another variant without marking the change.
- Do not claim a speed result from a quality-only run.
- Do not claim a quality result from synthetic timing requests.

## First-Pass Decision Table

| Result | Interpretation |
| --- | --- |
| FP8 wins quality and serving | Use FP8. |
| FP8 and TurboQuant match quality, FP8 wins TPOT | Use FP8 unless extra capacity is required. |
| TurboQuant lowers P99 TTFT at burst but worsens TPOT | TurboQuant is useful only under KV memory pressure. |
| TurboQuant fails NIAH or LongBench 8k | Do not proceed to higher context until the failure is understood. |
| TurboQuant needs lower request rate to avoid errors | Record it as a runtime limitation. |

## Next Context Lengths

After the 8k pass:

```text
16k -> 32k -> 64k -> model maximum
```

Increase only after both variants finish the lower length with valid outputs,
resource metrics, and no mixed artifacts.

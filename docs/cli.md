# CLI

## Metadata Generation

Script:

```bash
scripts/generate_vllm_metadata.py
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `--model` | Target HuggingFace model. |
| `--calibration-model` | Optional model used for activation collection. |
| `--kv-cache-dtype` | `turboquant25` or `turboquant35`. |
| `--prompts-file` | Text file with one prompt per line. |
| `--output` | Metadata JSON path. |
| `--trust-remote-code` | Passed to HuggingFace loaders. |
| `--layer-pattern` | Layer-name template, default uses `{i}`. |
| `--max-prompts` | Prompt limit. |
| `--max-seq-len` | Token truncation limit. |
| `--batch-size` | Calibration batch size. |
| `--dtype` | `auto`, `float32`, `float16`, or `bfloat16`. |
| `--device` | PyTorch device string. |

The prompts file is hashed and the hash is stored in metadata. Empty lines are
ignored. `--max-prompts` can be used to bound calibration cost while preserving
deterministic prompt ordering.

The script rejects using the same quantized checkpoint as the calibration source
when the target config reports quantization metadata. Use a non-quantized
calibration checkpoint when available.

## vLLM/HF Verification

Script:

```bash
scripts/verify_vllm_integration.py
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `--model` | HuggingFace model id or path. |
| `--bits` | Shared K/V bit width. |
| `--k-bits` and `--v-bits` | Per-component bits; must appear together. |
| `--kv-cache-dtype` | `tq4`, `k4v4`, `turboquant25`, or `turboquant35`. |
| `--metadata-path` | Required for recipe modes. |
| `--threshold` | Minimum cosine for pass status. |
| `--json` | Write JSON to stdout and text to stderr. |
| `--trust-remote-code` | Passed to HuggingFace loaders. |
| `--token` | Passed to HuggingFace loaders. |

Argument rules:

- `--bits` cannot be mixed with `--k-bits` or `--v-bits`.
- `--k-bits` and `--v-bits` must appear together.
- recipe cache dtypes require `--metadata-path`.
- JSON mode returns nonzero when the verification status is `FAIL`.

## Qwen A/B Evaluation

Script:

```bash
scripts/qwen_ab_eval.py
```

This script runs baseline and TurboQuant diagnostic scenarios and writes
separate artifacts for each run.

The script is designed for controlled A/B runs: same model, same prompts, same
generation settings, separate artifact directories.

## Phase Check Scripts

The repository includes scripts named:

```text
phase2_real_data_check.py
phase3_real_data_check.py
phase4_real_data_check.py
phase5_real_data_check.py
phase6_real_data_check.py
phase7_real_data_check.py
```

These are development validation scripts tied to the project phases.

They are not a replacement for end-to-end serving benchmarks, but they are
useful for finding regressions in layout, memory accounting, and attention
reference behavior.

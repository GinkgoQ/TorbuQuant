# Qwen A/B Evaluation

`scripts/qwen_ab_eval.py` compares a HuggingFace baseline against a TurboQuant
diagnostic path for the same model and scenarios.

## Run

```bash
python scripts/qwen_ab_eval.py \
  --model Qwen/Qwen2.5-3B \
  --output-root reports/qwen_ab \
  --dtype bfloat16
```

Use local model files when needed:

```bash
python scripts/qwen_ab_eval.py \
  --model Qwen/Qwen2.5-3B \
  --local-files-only
```

## Recorded Fields

The script records:

- scenario name,
- prompt and input length,
- generation settings,
- raw generated text,
- prompt/decode/total timing,
- tokens per second,
- CUDA peak allocation and reservation when CUDA is available,
- CPU and RAM samples when `psutil` is available,
- dense KV byte estimate,
- compressed byte estimate for the TurboQuant adapter,
- output comparison notes,
- artifact manifests.

## Artifact Isolation

Runs use a run id and separate directories for:

- baseline outputs,
- TurboQuant outputs,
- shared scenario manifest,
- metrics JSON,
- raw JSONL outputs.

The script is intended for controlled A/B checks, not for production serving
claims.


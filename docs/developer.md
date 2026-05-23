# Developer Guide

This guide focuses on changing the code without confusing diagnostic behavior
with serving behavior.

## Setup

```bash
python -m pip install -e ".[dev,docs]"
```

## Test Commands

```bash
python -m pytest tests -q
python -m pytest tests/test_vllm_recipe.py -q
python -m pytest tests/test_hf_dynamic_cache.py -q
```

## Documentation Commands

```bash
python -m mkdocs serve
python -m mkdocs build
```

## Development Rules

- Keep diagnostic and serving paths separate.
- Count all metadata and workspace bytes.
- Do not report speed gains without a named baseline.
- Do not use random-tensor tests as model-quality evidence.
- Keep reference implementations outside runtime imports.
- Add tests for byte layout, shape, and failure behavior.

## Review Checklist

Before merging a runtime change:

- Does the changed path reconstruct dense historical K/V?
- Does the report label diagnostic reconstruction?
- Are all byte categories represented?
- Is the baseline named?
- Are unsupported formats rejected?
- Are shape and dtype assumptions tested?
- Does the code avoid importing from `other_implemnetations/`?

## Adding A KV Format

1. Add format description in `turboquant.kv.formats`.
2. Add packing/unpacking helpers if needed.
3. Add byte accounting in `turboquant.kv.memory`.
4. Add cache policy mapping.
5. Add attention reference behavior.
6. Add Triton path only after reference tests pass.
7. Add quality and byte-report tests.

## Adding A Metadata Field

1. Add the field to the dataclass.
2. Add JSON serialization and parsing.
3. Add validation in the loader or runtime resolver.
4. Add a round-trip test.
5. Add a compatibility note if older files omit the field.

## Adding A vLLM Route

1. Define cache dtype and gate behavior.
2. Define page shape and slot byte math.
3. Add packed update contract.
4. Add packed decode contract.
5. Add kernel wrapper and tests.
6. Add runtime selection and fallback errors.
7. Update [vLLM Audit](vllm_reference_audit.md).

## Test Tiers

| Tier | Example |
| --- | --- |
| Byte layout | `tests/test_packing_bits.py`, `tests/test_vllm_page_cache.py` |
| Math | `tests/test_core_codebook.py`, `tests/test_core_mse.py` |
| Runtime contract | `tests/test_vllm_tq4_update.py`, `tests/test_vllm_tq4_decode.py` |
| Integration | `tests/test_hf_dynamic_cache.py`, `tests/test_integration_phase7.py` |
| Quality | `tests/test_quality_metrics.py`, retrieval and trajectory tests |

Run the narrow test first, then the suite:

```bash
python -m pytest tests/test_vllm_tq4_decode.py -q
python -m pytest tests -q
```

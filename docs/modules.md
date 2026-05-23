# Internal Modules

This page is a module-oriented map. Use the API reference for signatures and
docstrings; use this page to understand where code belongs.

## `turboquant.core`

Core math for rotations, codebooks, MSE quantization, QJL, and shared payload
types.

Important files:

- `codebook.py`: Beta-sphere and Gaussian codebook generation.
- `rotation.py`: QR, RHT, inverse transforms, and QJL matrix generation.
- `mse.py`: MSE payload creation and reconstruction.
- `qjl.py`: sign packing and residual sketch estimator.
- `polar.py`: product quantizer built from MSE and QJL.
- `types.py`: shared named tuples and serializable specs.

## `turboquant.packing`

Bitstream utilities for unsigned indices and sign bits. Used by MSE, QJL,
values, recipe packing, and TQ page rows.

The packer uses little-endian bit order inside bytes. Tests cover 1/2/3/4/5-bit
payloads.

## `turboquant.kv`

KV cache format descriptions, key/value quantization, compressed cache owners,
recent-window handling, policy construction, and byte reports.

Key distinction:

- `cache.py` is the phase-oriented cache owner with reports and recent windows.
- `store.py` contains a flatter store abstraction and value chunk helpers.
- `compressors.py` contains compressor wrappers modeled after the reference
  implementation family.

## `turboquant.attention`

Reference attention paths:

- dense attention,
- SDPA attention,
- diagnostic dequant attention,
- direct-QK score path,
- packed value accumulation,
- hybrid exact/compressed attention.

`reference.py` favors clarity and shape coverage. `hybrid.py` adds exact-token
and compressed-token mixing for cache stores.

## `turboquant.triton`

Contains both Triton kernels and PyTorch reference contracts for packed page
behavior. Check each function docstring before using it for timing claims.

Files with live Triton JIT code and files with reference contracts live in the
same package so callers can share imports. Documentation pages identify which is
which.

## `turboquant.integration.common`

Shared runtime config, backend capability detection, and integration report
objects.

This package is intentionally small. It is the bridge from user-facing config to
cache policy and reporting.

## `turboquant.integration.hf`

HuggingFace capture and diagnostic cache wrapper code.

The HF path is primarily for:

- Qwen activation capture,
- generated-text comparison,
- DynamicCache lifecycle checks,
- diagnostic compression reports.

## `turboquant.integration.vllm`

vLLM-oriented recipe layouts, metadata JSON, calibration, page cache, runtime
dispatch, registry, and verification helpers.

The vLLM package should remain independent from HuggingFace cache wrappers,
except for verification code that intentionally uses HF as a diagnostic model
loader.

## `turboquant.quality`

Quality and evaluation utilities:

- tensor error,
- KL and logits metrics,
- timing stats,
- retrieval prompt generation,
- trajectory drift metrics,
- hardware-profile replay parsers.

## `turboquant.weights`

Experimental weight compression helpers. These are separate from the KV-cache
path.

These modules are documented because they are present in the package, but they
are not evidence for KV-cache serving behavior.

## `turboquant.search`

Vector search helper code using compressed vector representations.

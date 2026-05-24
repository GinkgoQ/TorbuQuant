<p align="center">
  <img src="docs/assets/images/banner.png" alt="TorbuQuant banner" width="100%">
</p>

<h1 align="center">TorbuQuant</h1>

<p align="center">
  <strong>LLM KV-cache compression, vector quantization, cache accounting, and serving-path experiments.</strong>
</p>

<p align="center">
  A <strong>Ginkgo<span style="color:#79863c">Q</span></strong> research implementation for measuring when compressed KV storage helps real long-context inference workloads.
</p>

---

## Purpose

TorbuQuant is a Python package for studying and implementing TurboQuant-style
LLM KV-cache compression. The repository keeps the research path and serving
path separate:

| Path            | Role                                                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Diagnostic path | Reconstructs or compares dense tensors so logits, attention outputs, and generation behavior can be inspected.            |
| Serving path    | Stores historical K/V in packed byte layouts and measures whether that storage can improve capacity under a memory bound. |

The project is built around three measurable goals:

1. Reduce KV-cache storage bytes while counting metadata and workspace.
2. Preserve model behavior under explicit quality gates.
3. Improve practical serving capacity when KV memory, context length, or batch size is the bottleneck.

## What Is In This Repository

| Area                    | Status                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| Codebooks               | Exact Beta-sphere Lloyd-Max codebooks and Gaussian approximation.                           |
| Rotation                | QR and RHT modes.                                                                           |
| Vector quantization     | MSE direction quantization with packed index storage.                                       |
| QJL                     | Residual sign projection for vector and inner-product experiments.                          |
| KV formats              | K16, K8, K4 and V8, V4, V3, V2 helpers.                                                     |
| Cache accounting        | Packed bytes, norms, scales, zeros, metadata, recent windows, and workspace reports.        |
| Attention               | Dense, dequantized, direct-QK, packed-V, and selected Triton decode paths.                  |
| vLLM integration        | Metadata, page geometry, page cache, registry, runtime selection, and verification helpers. |
| HuggingFace integration | Diagnostic DynamicCache wrapper and Qwen evaluation scripts.                                |
| Qwen A/B                | Controlled baseline-vs-TurboQuant evaluation script with isolated run directories.          |

## Install

Create and activate a Python environment, then install the repository from the
project root:

### Install from source

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Install from PyPI

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ginkgoq-turboquant
```

## Quick Checks

Run the repository test suite:

```bash
python -m pytest tests -q
```

Build the documentation site:

```bash
python -m mkdocs build --strict
```

Run the Qwen A/B evaluator with explicit scenario settings before making any
memory, quality, or speed statement:

```bash
python scripts/qwen_ab_eval.py --help
```

## Repository Map

```text
turboquant/
  core/          codebooks, rotations, MSE quantization, QJL
  packing/       bit packing and byte accounting helpers
  kv/            compressed cache formats and memory reports
  attention/     dense, diagnostic, and packed-reference attention paths
  triton/        update/decode kernels and reference wrappers
  integration/   HuggingFace and vLLM integration layers
  quality/       comparison, retrieval, and generation metrics
scripts/         metadata, verification, benchmark, and Qwen A/B entry points
docs/            MkDocs Material publication site
tests/           unit and integration tests
```

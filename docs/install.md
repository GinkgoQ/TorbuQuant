# Install

## Requirements

The project package metadata declares:

- Python `>=3.9`
- `torch`
- `triton`
- `numpy`
- `scipy`
- `transformers`

Recommended environment:

- Python virtual environment or conda environment
- PyTorch with CUDA support
- Triton
- Transformers

## Environment

Create an environment before installing the package:

=== "venv"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```

=== "conda"

    ```bash
    conda create -n turboquant python=3.10
    conda activate turboquant
    python -m pip install --upgrade pip
    ```

## Package Install

From the repository root, install the package in editable mode:

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[dev]"
```

For documentation:

```bash
python -m pip install -e ".[docs]"
```

## Build The Documentation Site

```bash
python -m mkdocs serve
python -m mkdocs build --strict
```

The generated site is written to `site/`.

## Optional Runtime Packages

Some integration paths require optional packages at runtime:

| Path | Package |
| --- | --- |
| HuggingFace model capture | `transformers` |
| vLLM helpers | `vllm` |
| CUDA kernels | CUDA-capable PyTorch and Triton |

If an optional package is absent, helper code raises `OptionalDependencyError`
or a standard import/configuration error depending on the entry point.

"""TorbuQuant package.

TorbuQuant is a high-performance KV cache compression library for
large language model inference. It provides:

- Core quantization algorithms (MSE, Polar/QJL)
- KV cache compression with configurable policies
- HuggingFace and vLLM integrations
- Triton GPU kernels for fused attention
- Quality metrics and benchmarking tools
- Weight quantization (optional)
- Vector search (optional)

Example:
    ```python
    from torbuquant.core import TorbuquantMSE, TorbuquantProd
    from torbuquant.kv import CompressedKVCache
    from torbuquant.attention import compute_hybrid_attention
    ```
"""

__all__ = [
    "__version__",
    # Submodules
    "core",
    "kv",
    "attention",
    "triton",
    "quality",
    "integration",
    "weights",
    "search",
]

__version__ = "0.1.0"

# Lazy imports for optional submodules
def __getattr__(name: str):
    if name == "weights":
        from torbuquant import weights
        return weights
    elif name == "search":
        from torbuquant import search
        return search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""HuggingFace integration package."""

from torbuquant.integration.hf.capture import QwenCapture, capture_generated_tokens, capture_qwen_layer
from torbuquant.integration.hf.config import HFQwenSettings, hf_qwen_settings
from torbuquant.integration.hf.dynamic_cache import CompressedDynamicCache, CompressedLayer
from torbuquant.integration.hf.qwen import (
    DynamicCachePatch,
    HFDiagnosticCacheAdapter,
    build_layer_cache_from_capture,
)

__all__ = [
    "CompressedDynamicCache",
    "CompressedLayer",
    "DynamicCachePatch",
    "HFDiagnosticCacheAdapter",
    "HFQwenSettings",
    "QwenCapture",
    "build_layer_cache_from_capture",
    "capture_generated_tokens",
    "capture_qwen_layer",
    "hf_qwen_settings",
]

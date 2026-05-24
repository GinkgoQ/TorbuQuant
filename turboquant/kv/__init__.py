"""KV cache data structures and accounting."""

from turboquant.kv.cache import (
    BoundaryPolicy,
    CompressedKVBlock,
    CompressedKVCache,
    CompressedKVPage,
    DiagnosticDenseKV,
    ProductionKVHandle,
    SparseOutlierStore,
)
from turboquant.kv.capture import (
    KVCaptureEngine,
    RingBuffer,
)
from turboquant.kv.compressors import (
    CompressedKeys,
    CompressedValues,
    MSECompressor,
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
    TurboQuantV3,
)
from turboquant.kv.formats import (
    K_FORMATS,
    V_FORMATS,
    KVFormatSpec,
    get_k_format,
    get_v_format,
    validate_k_format,
    validate_v_format,
)
from turboquant.kv.keys import (
    DenseKeyData,
    KeyPayload,
    QuantizedKeyData,
    TurboKeyData,
    build_k4_quantizer,
    dequantize_k8,
    key_payload_nbytes,
    quantize_k8,
)
from turboquant.kv.layout import CacheGeometry, PackedKVLayout, estimate_persistent_bytes
from turboquant.kv.memory import (
    ByteLedger,
    MemoryReport,
    bytes_from_tensors,
    report_from_components,
)
from turboquant.kv.policy import (
    AutoPolicyDecision,
    AutoPolicyInput,
    BackendCapability,
    KVQuantPolicy,
    choose_auto_kv_policy,
    estimate_dense_kv_bytes,
    estimate_policy_kv_bytes,
    qwen25_3b_policy,
)
from turboquant.kv.recent import RecentWindow
from turboquant.kv.store import (
    CompressedKVStore,
    FlatCache,
    TurboQuantKVCache,
    ValueQuantized,
    dequantize_value_chunk,
    quantize_value_chunk,
    unpack_value_data,
)
from turboquant.kv.values import (
    dequantize_values,
    padded_dim,
    quantize_values,
    value_data_nbytes,
    value_formula_nbytes,
)

__all__ = [
    # Cache structures
    "BackendCapability",
    "AutoPolicyDecision",
    "AutoPolicyInput",
    "BoundaryPolicy",
    "ByteLedger",
    "CacheGeometry",
    "CompressedKVBlock",
    "CompressedKVCache",
    "CompressedKVPage",
    "CompressedKVStore",
    "DenseKeyData",
    "DiagnosticDenseKV",
    "FlatCache",
    "K_FORMATS",
    "KVCaptureEngine",
    "KVFormatSpec",
    "KVQuantPolicy",
    "KeyPayload",
    "MemoryReport",
    "PackedKVLayout",
    "ProductionKVHandle",
    "QuantizedKeyData",
    "RecentWindow",
    "RingBuffer",
    "SparseOutlierStore",
    "TurboKeyData",
    "TurboQuantKVCache",
    "V_FORMATS",
    "ValueQuantized",
    # Compressors
    "CompressedKeys",
    "CompressedValues",
    "MSECompressor",
    "TurboQuantCompressorMSE",
    "TurboQuantCompressorV2",
    "TurboQuantV3",
    # Functions
    "bytes_from_tensors",
    "build_k4_quantizer",
    "choose_auto_kv_policy",
    "dequantize_k8",
    "dequantize_value_chunk",
    "dequantize_values",
    "estimate_persistent_bytes",
    "estimate_dense_kv_bytes",
    "estimate_policy_kv_bytes",
    "get_k_format",
    "get_v_format",
    "key_payload_nbytes",
    "padded_dim",
    "quantize_k8",
    "quantize_value_chunk",
    "quantize_values",
    "qwen25_3b_policy",
    "report_from_components",
    "unpack_value_data",
    "validate_k_format",
    "validate_v_format",
    "value_data_nbytes",
    "value_formula_nbytes",
]

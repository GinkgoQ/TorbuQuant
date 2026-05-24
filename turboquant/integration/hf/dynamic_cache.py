"""HuggingFace DynamicCache wrapper with compressed storage."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch

from turboquant.core.mse import TorbuquantMSE
from turboquant.core.rotation import RotationMode, build_rotation
from turboquant.core.types import MSEData


def _bits_to_mode(bits: int) -> int:
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    return int(bits)


@dataclass
class CompressedLayer:
    indices: torch.Tensor
    norms: torch.Tensor
    bits: int
    dim: int

    @property
    def seq_len(self) -> int:
        return int(self.norms.shape[-1])

    @property
    def byte_count(self) -> int:
        return int(
            self.indices.numel() * self.indices.element_size()
            + self.norms.numel() * self.norms.element_size()
        )

    @property
    def baseline_bytes(self) -> int:
        batch, heads, tokens = self.norms.shape
        return int(batch * heads * tokens * self.dim * 2)


class HeadDimCodecs:
    def __init__(
        self,
        *,
        k_bits: int,
        v_bits: int,
        seed: int,
        device: torch.device,
        rotation_mode: RotationMode,
    ):
        self.k_bits = _bits_to_mode(k_bits)
        self.v_bits = _bits_to_mode(v_bits)
        self.seed = int(seed)
        self.device = device
        self.rotation_mode = rotation_mode
        self.key: dict[int, TorbuquantMSE] = {}
        self.value: dict[int, TorbuquantMSE] = {}

    def for_dim(self, dim: int) -> tuple[TorbuquantMSE, TorbuquantMSE]:
        if dim not in self.key:
            k_rotation = build_rotation(
                dim,
                self.rotation_mode,
                device=self.device,
                dtype=torch.float32,
                seed=self.seed,
            )
            v_rotation = build_rotation(
                dim,
                self.rotation_mode,
                device=self.device,
                dtype=torch.float32,
                seed=self.seed,
            )
            self.key[dim] = TorbuquantMSE(
                dim,
                self.k_bits,
                k_rotation,
                self.device,
                norm_correction=True,
            )
            self.value[dim] = TorbuquantMSE(
                dim,
                self.v_bits,
                v_rotation,
                self.device,
                norm_correction=True,
            )
        return self.key[dim], self.value[dim]


class CompressedDynamicCache:
    """Patch a Transformers cache and store compressed K/V rows.

    The wrapper dequantizes for HuggingFace attention output, so this is a
    diagnostic route. Production decode should use packed page kernels.
    """

    def __init__(
        self,
        cache: Any,
        *,
        head_dim: int,
        bits: int | None = 4,
        k_bits: int | None = None,
        v_bits: int | None = None,
        seed: int = 42,
        device: torch.device | None = None,
        rotation_mode: RotationMode = RotationMode.RHT,
        model_config: Any | None = None,
    ):
        resolved_k = k_bits if k_bits is not None else bits
        resolved_v = v_bits if v_bits is not None else bits
        if resolved_k is None or resolved_v is None:
            raise ValueError("provide bits or both k_bits and v_bits")
        self.cache = cache
        self.head_dim = int(head_dim)
        self.k_bits = _bits_to_mode(int(resolved_k))
        self.v_bits = _bits_to_mode(int(resolved_v))
        self.bits = self.k_bits
        self.seed = int(seed)
        self.device = device or torch.device("cpu")
        self.enabled = True
        self.diagnostic_label = "hf_dequant"
        self._original_update = cache.update
        self._original_get_seq_length = getattr(cache, "get_seq_length", None)
        self._key_layers: list[CompressedLayer | None] = []
        self._value_layers: list[CompressedLayer | None] = []
        self._dequant_keys: list[torch.Tensor | None] = []
        self._dequant_values: list[torch.Tensor | None] = []
        self._dtype = torch.float16
        self._codecs = HeadDimCodecs(
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            seed=self.seed,
            device=self.device,
            rotation_mode=rotation_mode,
        )
        self._bypass_layers = _full_attention_layers(model_config)

        if hasattr(cache.update, "__self__") and isinstance(cache.update.__self__, CompressedDynamicCache):
            warnings.warn("cache is already wrapped by TurboQuant", UserWarning, stacklevel=2)

        cache.update = self._update
        if self._original_get_seq_length is not None:
            cache.get_seq_length = self._get_seq_length

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def restore(self) -> None:
        self.cache.update = self._original_update
        if self._original_get_seq_length is not None:
            self.cache.get_seq_length = self._original_get_seq_length

    def __enter__(self) -> "CompressedDynamicCache":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.restore()
        return False

    def _update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled or layer_idx in self._bypass_layers:
            return self._original_update(key_states, value_states, layer_idx, cache_kwargs)
        _check_hf_kv(key_states, value_states)
        self._dtype = key_states.dtype
        dim = int(key_states.shape[-1])
        key_codec, value_codec = self._codecs.for_dim(dim)
        new_key = _compress(key_codec, key_states.to(self.device))
        new_value = _compress(value_codec, value_states.to(self.device))
        _pad_layers(self._key_layers, self._value_layers, layer_idx)
        _pad_dequant(self._dequant_keys, self._dequant_values, layer_idx)
        self._key_layers[layer_idx] = _cat_layer(self._key_layers[layer_idx], new_key)
        self._value_layers[layer_idx] = _cat_layer(self._value_layers[layer_idx], new_value)
        key_dense = key_codec.dequantize(_to_mse(new_key)).to(device=key_states.device, dtype=key_states.dtype)
        value_dense = value_codec.dequantize(_to_mse(new_value)).to(device=value_states.device, dtype=value_states.dtype)
        self._dequant_keys[layer_idx] = _cat_dense(self._dequant_keys[layer_idx], key_dense)
        self._dequant_values[layer_idx] = _cat_dense(self._dequant_values[layer_idx], value_dense)
        out_key = self._dequant_keys[layer_idx]
        out_value = self._dequant_values[layer_idx]
        assert out_key is not None
        assert out_value is not None
        _write_cache_layer(self.cache, layer_idx, out_key, out_value, key_states, value_states)
        return out_key, out_value

    def _get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.enabled or layer_idx in self._bypass_layers:
            return int(self._original_get_seq_length(layer_idx))
        if layer_idx >= len(self._key_layers):
            return 0
        layer = self._key_layers[layer_idx]
        if layer is None:
            return 0
        return layer.seq_len

    def get_compressed(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self._key_layers):
            raise ValueError(f"layer {layer_idx} has no compressed cache")
        key = self._key_layers[layer_idx]
        value = self._value_layers[layer_idx]
        if key is None or value is None:
            raise ValueError(f"layer {layer_idx} has no compressed cache")
        return key.indices, key.norms, value.indices, value.norms

    def vram_bytes(self) -> int:
        return sum(layer.byte_count for layer in [*self._key_layers, *self._value_layers] if layer is not None)

    def baseline_vram_bytes(self) -> int:
        return sum(layer.baseline_bytes for layer in [*self._key_layers, *self._value_layers] if layer is not None)

    def compression_stats(self) -> dict[str, Any]:
        compressed = self.vram_bytes()
        baseline = self.baseline_vram_bytes()
        layers = [layer for layer in self._key_layers if layer is not None]
        ratio = baseline / compressed if compressed else 0.0
        seq_len = layers[0].seq_len if layers else 0
        heads = int(layers[0].norms.shape[1]) if layers else 0
        return {
            "layers": len(layers),
            "seq_len": seq_len,
            "heads": heads,
            "head_dim": self.head_dim,
            "bits": self.bits,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "compressed_bytes": compressed,
            "baseline_bytes": baseline,
            "compression_ratio": ratio,
            "context_bytes": {
                str(length): _context_bytes(len(layers), heads, self.head_dim, self.k_bits, self.v_bits, length)
                for length in (4096, 16384, 32768)
            },
            "label": self.diagnostic_label,
        }


def _compress(codec: TorbuquantMSE, tensor: torch.Tensor) -> CompressedLayer:
    data = codec.quantize(tensor.float())
    return CompressedLayer(
        indices=data.indices,
        norms=data.norms.float(),
        bits=data.bits,
        dim=data.dim,
    )


def _to_mse(layer: CompressedLayer) -> MSEData:
    return MSEData(indices=layer.indices, norms=layer.norms, bits=layer.bits, dim=layer.dim)


def _cat_layer(old: CompressedLayer | None, new: CompressedLayer) -> CompressedLayer:
    if old is None:
        return new
    if old.bits != new.bits or old.dim != new.dim:
        raise ValueError("cannot append compressed layers with different layouts")
    return CompressedLayer(
        indices=torch.cat([old.indices, new.indices], dim=-2),
        norms=torch.cat([old.norms, new.norms], dim=-1),
        bits=old.bits,
        dim=old.dim,
    )


def _cat_dense(old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    if old is None:
        return new
    return torch.cat([old, new], dim=-2)


def _pad_layers(key_layers: list[CompressedLayer | None], value_layers: list[CompressedLayer | None], layer_idx: int) -> None:
    while len(key_layers) <= layer_idx:
        key_layers.append(None)
        value_layers.append(None)


def _pad_dequant(key_layers: list[torch.Tensor | None], value_layers: list[torch.Tensor | None], layer_idx: int) -> None:
    while len(key_layers) <= layer_idx:
        key_layers.append(None)
        value_layers.append(None)


def _check_hf_kv(key_states: torch.Tensor, value_states: torch.Tensor) -> None:
    if key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("key/value tensors must have shape [batch, heads, tokens, dim]")
    if key_states.shape != value_states.shape:
        raise ValueError("key/value shapes differ")


def _write_cache_layer(
    cache: Any,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_init: torch.Tensor,
    value_init: torch.Tensor,
) -> None:
    layers = getattr(cache, "layers", None)
    if layers is None or layer_idx >= len(layers):
        return
    layer = layers[layer_idx]
    if hasattr(layer, "is_initialized") and not layer.is_initialized:
        try:
            layer.lazy_initialization(key_init, value_init)
        except TypeError:
            layer.lazy_initialization(key_init)
    if hasattr(layer, "keys"):
        layer.keys = key_states
    if hasattr(layer, "values"):
        layer.values = value_states


def _full_attention_layers(model_config: Any | None) -> set[int]:
    layer_types = getattr(model_config, "layer_types", None)
    if not layer_types:
        return set()
    if not any("sliding" in str(item) for item in layer_types):
        return set()
    return {idx for idx, item in enumerate(layer_types) if "full" in str(item)}


def _context_bytes(layers: int, heads: int, dim: int, k_bits: int, v_bits: int, tokens: int) -> int:
    k_row = ((dim * k_bits + 7) // 8) + 4
    v_row = ((dim * v_bits + 7) // 8) + 4
    return int(layers * heads * tokens * (k_row + v_row))

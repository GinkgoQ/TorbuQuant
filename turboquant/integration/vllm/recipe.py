"""TurboQuant cache recipes used by vLLM integration code."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from turboquant.packing.bits import packed_length


RecipeName = Literal["turboquant25", "turboquant35"]
RecipeBits = float

RECIPE_BITS = {
    "turboquant25": 2.5,
    "turboquant35": 3.5,
}

RECIPE_OUTLIER_RATIOS = {
    "turboquant25": 0.25,
    "turboquant35": 0.50,
}

RECIPE_GROUP_BITS = {
    "turboquant25": (3, 2),
    "turboquant35": (4, 3),
}

VECTOR_NORM_BYTES = 2
RESIDUAL_NORM_BYTES = 2
NORM_BYTES = VECTOR_NORM_BYTES + RESIDUAL_NORM_BYTES
GROUP_ALIGNMENT = 16
SEED = 20250428
QJL_SEED_OFFSET = 10_000
QJL_SCALE = math.sqrt(math.pi / 2.0)
CODEBOOK_GRID_POINTS = 32768
CODEBOOK_EPS = 1e-6
CUDA_CAPABILITIES = frozenset(((8, 6), (12, 1)))
CUDA_NOTE = "TurboQuant vLLM kernels require CUDA SM86 or SM121."
TQ_BITS = 4
TQ_VALID_BITS = frozenset({2, 3, 4, 5})
TQ_NORM_BYTES = 4


@dataclass(frozen=True)
class KernelMeta:
    decode_block_n: int
    decode_num_warps: int
    update_tile: int
    update_num_warps: int
    postprocess_num_warps: int


RecipeKernelMeta = KernelMeta


@dataclass(frozen=True)
class RecipeGroupLayout:
    dim: int
    bits: int
    mse_bits: int
    mse_payload_bytes: int
    qjl_payload_bytes: int
    qjl_offset: int
    vector_norm_offset: int
    residual_norm_offset: int
    packed_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "dim": self.dim,
            "bits": self.bits,
            "mse_bits": self.mse_bits,
            "mse_payload_bytes": self.mse_payload_bytes,
            "qjl_payload_bytes": self.qjl_payload_bytes,
            "qjl_offset": self.qjl_offset,
            "vector_norm_offset": self.vector_norm_offset,
            "residual_norm_offset": self.residual_norm_offset,
            "packed_bytes": self.packed_bytes,
        }


@dataclass(frozen=True)
class RecipeLayout:
    recipe: str
    head_dim: int
    groups: tuple[RecipeGroupLayout, RecipeGroupLayout]
    packed_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "head_dim": self.head_dim,
            "packed_dim": self.packed_dim,
            "groups": [group.to_dict() for group in self.groups],
        }


def is_turboquant_dtype(cache_dtype: str | None) -> bool:
    return cache_dtype in RECIPE_BITS


def get_recipe_bits(cache_dtype: str) -> float:
    try:
        return RECIPE_BITS[cache_dtype]
    except KeyError as exc:
        raise ValueError(f"unknown TurboQuant cache dtype: {cache_dtype}") from exc


def recipe_bits(name: RecipeName | str) -> RecipeBits:
    return get_recipe_bits(name)


def canonical_recipe(bits_or_dtype: float | int | str) -> str:
    if isinstance(bits_or_dtype, str):
        if not is_turboquant_dtype(bits_or_dtype):
            raise ValueError(f"unknown TurboQuant cache dtype: {bits_or_dtype}")
        return bits_or_dtype
    bits = float(bits_or_dtype)
    if bits == 2.5:
        return "turboquant25"
    if bits == 3.5:
        return "turboquant35"
    raise ValueError(f"unknown TurboQuant bit width: {bits}")


def normalize_cuda_capability(capability: Any | None) -> tuple[int, int] | None:
    if capability is None:
        return None
    if hasattr(capability, "major") and hasattr(capability, "minor"):
        return (int(capability.major), int(capability.minor))
    if isinstance(capability, (tuple, list)) and len(capability) >= 2:
        return (int(capability[0]), int(capability[1]))
    raise TypeError(f"unknown CUDA capability value: {capability!r}")


def is_turboquant_cuda(capability: Any | None) -> bool:
    return normalize_cuda_capability(capability) in CUDA_CAPABILITIES


def recipe_cuda_allowed(capability: Any | None) -> bool:
    return is_turboquant_cuda(capability)


def kernel_meta(capability: Any | None, head_dim: int) -> KernelMeta:
    normalized = normalize_cuda_capability(capability)
    if normalized == (8, 6):
        return KernelMeta(
            decode_block_n=8,
            decode_num_warps=2,
            update_tile=16,
            update_num_warps=2,
            postprocess_num_warps=2,
        )
    return KernelMeta(
        decode_block_n=8 if head_dim >= 256 else 16,
        decode_num_warps=4,
        update_tile=32,
        update_num_warps=4,
        postprocess_num_warps=4,
    )


def recipe_kernel_meta(capability: Any | None, head_dim: int) -> RecipeKernelMeta:
    return kernel_meta(capability, head_dim)


def env_flag(name: str, environ: dict[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return values.get(name, "").lower() in ("1", "true", "yes")


def parse_tq_bits_env(environ: dict[str, str] | None = None) -> tuple[int, int]:
    values = os.environ if environ is None else environ
    raw_k = values.get("TQ4_K_BITS", str(TQ_BITS))
    raw_v = values.get("TQ4_V_BITS", str(TQ_BITS))
    try:
        k_bits = int(raw_k)
        v_bits = int(raw_v)
    except ValueError as exc:
        raise ValueError(f"TQ4_K_BITS and TQ4_V_BITS must be integers: {raw_k!r}, {raw_v!r}") from exc
    for name, bits in (("TQ4_K_BITS", k_bits), ("TQ4_V_BITS", v_bits)):
        if bits not in TQ_VALID_BITS:
            raise ValueError(f"{name} must be one of {sorted(TQ_VALID_BITS)}, got {bits}")
    return k_bits, v_bits


def tq_index_bytes(head_dim: int, bits: int) -> int:
    if bits not in TQ_VALID_BITS:
        raise ValueError(f"bits must be one of {sorted(TQ_VALID_BITS)}, got {bits}")
    return packed_length(head_dim, bits)


def tq_bytes_per_token(head_dim: int, bits: int = TQ_BITS) -> int:
    return tq_index_bytes(head_dim, bits) + TQ_NORM_BYTES


def tq_bytes_per_token_kv(
    head_dim: int,
    *,
    k_bits: int = TQ_BITS,
    v_bits: int = TQ_BITS,
) -> int:
    return tq_bytes_per_token(head_dim, k_bits) + tq_bytes_per_token(head_dim, v_bits)


def padded_slot_bytes(head_dim: int, *, k_bits: int = TQ_BITS, v_bits: int = TQ_BITS) -> int:
    raw = tq_bytes_per_token_kv(head_dim, k_bits=k_bits, v_bits=v_bits)
    return 1 << (raw - 1).bit_length()


def outlier_count(head_dim: int, recipe: str) -> int:
    if head_dim % GROUP_ALIGNMENT != 0:
        raise ValueError("head_dim must be a multiple of 16 for TurboQuant recipes")
    if recipe not in RECIPE_OUTLIER_RATIOS:
        raise ValueError(f"unknown TurboQuant recipe: {recipe}")
    ratio = RECIPE_OUTLIER_RATIOS[recipe]
    count = int(round(head_dim * ratio / GROUP_ALIGNMENT) * GROUP_ALIGNMENT)
    if count <= 0 or count >= head_dim:
        raise ValueError(f"head_dim {head_dim} cannot use recipe {recipe}")
    return count


def group_dims(head_dim: int, recipe: str) -> tuple[int, int]:
    high = outlier_count(head_dim, recipe)
    return high, head_dim - high


def recipe_group_dims(head_dim: int, name: RecipeName | str) -> tuple[int, int]:
    return group_dims(head_dim, name)


@lru_cache(maxsize=None)
def recipe_layout(recipe: str, head_dim: int) -> RecipeLayout:
    dims = group_dims(head_dim, recipe)
    group_bits = RECIPE_GROUP_BITS[recipe]
    groups: list[RecipeGroupLayout] = []
    cursor = 0
    for dim, bits in zip(dims, group_bits, strict=True):
        mse_bits = bits - 1
        mse_payload_bytes = (dim * mse_bits + 7) // 8
        qjl_payload_bytes = (dim + 7) // 8
        qjl_offset = cursor + mse_payload_bytes
        vector_norm_offset = qjl_offset + qjl_payload_bytes
        residual_norm_offset = vector_norm_offset + VECTOR_NORM_BYTES
        packed_bytes = mse_payload_bytes + qjl_payload_bytes + NORM_BYTES
        groups.append(
            RecipeGroupLayout(
                dim=dim,
                bits=bits,
                mse_bits=mse_bits,
                mse_payload_bytes=mse_payload_bytes,
                qjl_payload_bytes=qjl_payload_bytes,
                qjl_offset=qjl_offset,
                vector_norm_offset=vector_norm_offset,
                residual_norm_offset=residual_norm_offset,
                packed_bytes=packed_bytes,
            )
        )
        cursor += packed_bytes
    return RecipeLayout(
        recipe=recipe,
        head_dim=head_dim,
        groups=(groups[0], groups[1]),
        packed_dim=cursor,
    )


def packed_dim(head_dim: int, bits_or_dtype: float | int | str) -> int:
    return recipe_layout(canonical_recipe(bits_or_dtype), head_dim).packed_dim


def recipe_packed_dim(head_dim: int, name: RecipeName | str) -> int:
    return packed_dim(head_dim, name)
